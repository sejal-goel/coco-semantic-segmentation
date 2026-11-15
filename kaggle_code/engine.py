"""Training and evaluation loops, shared by every experiment.

Checkpointing: after every epoch a checkpoint is written to Drive, and
train() resumes automatically if one exists - so a Colab disconnect
never loses more than one epoch.
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

import config as C
import metrics as M


def _device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _poly_lr(epoch, max_epochs):
    return (1 - epoch / max(max_epochs, 1)) ** C.POLY_POWER


def ckpt_path(tag):
    return os.path.join(C.CKPT_DIR, f'{tag}_ckpt.pt')


def history_path(tag):
    return os.path.join(C.RESULTS_DIR, f'{tag}_history.json')


def train(model, tag, train_loader, val_loader, loss_fn, epochs,
          lr=C.LR, seed=42, verbose=True):
    """Train with checkpoint-resume. Returns the history list.

    tag uniquely names the run (model/loss/seed/improvement combo), so
    every experiment keeps its own checkpoint and history.
    """
    C.ensure_dirs()
    C.set_seed(seed)
    dev = _device()
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: _poly_lr(e, epochs))
    scaler = torch.amp.GradScaler('cuda', enabled=(C.USE_AMP and dev == 'cuda'))
    accum = C.GRAD_ACCUM.get(tag.split('_')[0], 1)

    start_epoch, history = 0, []
    path = ckpt_path(tag)
    if os.path.exists(path):
        state = torch.load(path, map_location=dev, weights_only=False)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['opt'])
        sched.load_state_dict(state['sched'])
        scaler.load_state_dict(state['scaler'])
        start_epoch = state['epoch'] + 1
        history = state['history']
        if start_epoch < epochs:
            print(f'[resume] {tag}: continuing at epoch {start_epoch + 1}/{epochs}')
        else:
            print(f'[resume] {tag}: already trained for {epochs} epochs, skipping')

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        running, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(C.USE_AMP and dev == 'cuda')):
                loss = loss_fn(model(x), y) / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0 or step + 1 == len(train_loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            running += float(loss.detach()) * accum
            nb += 1
        sched.step()
        train_loss = running / max(nb, 1)
        val = evaluate(model, val_loader)
        secs = time.time() - t0
        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'val_miou': val['miou'], 'epoch_seconds': secs})
        if verbose:
            print(f'{tag} epoch {epoch + 1}/{epochs}  loss {train_loss:.4f}  '
                  f'val mIoU {val["miou"]:.4f}  ({secs:.0f}s)')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                    'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                    'epoch': epoch, 'history': history}, path)
        with open(history_path(tag), 'w') as f:
            json.dump(history, f, indent=2)
    return history


@torch.no_grad()
def evaluate(model, loader, threshold=0.0, tta=False, max_batches=None):
    """Full validation/test metric set (Part D formulas).

    threshold: pixels whose top softmax probability is below `threshold`
    are assigned to background. 0.0 = plain argmax (the default).
    tta: average softmax over the image and its horizontal flip.
    """
    dev = _device()
    model = model.to(dev).eval()
    cm = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), dtype=np.int64)
    bf1 = []
    for bi, (x, y) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = x.to(dev)
        with torch.amp.autocast('cuda', enabled=(C.USE_AMP and dev == 'cuda')):
            probs = F.softmax(model(x).float(), dim=1)
            if tta:
                probs = probs + torch.flip(F.softmax(
                    model(torch.flip(x, dims=[3])).float(), dim=1), dims=[3])
                probs = probs / 2
        conf, pred = probs.max(dim=1)
        pred = torch.where(conf < threshold, torch.zeros_like(pred), pred)
        pred_np = pred.cpu().numpy()
        y_np = y.numpy()
        for i in range(pred_np.shape[0]):
            cm += M.confusion_matrix(pred_np[i], y_np[i], C.NUM_CLASSES)
            if len(bf1) < 32:                        # boundary F1 on first 32 imgs (cost)
                bf1.append(M.boundary_f1(pred_np[i], y_np[i], C.NUM_CLASSES))
    out = M.metrics_from_confusion(cm)
    out['boundary_f1'] = float(np.mean(bf1)) if bf1 else float('nan')
    return out


@torch.no_grad()
def threshold_sweep(model, loader, grid=None):
    """One validation pass that accumulates a confusion matrix per candidate
    threshold - the search space the threshold agent optimises over."""
    grid = grid or C.THRESHOLD_GRID
    dev = _device()
    model = model.to(dev).eval()
    cms = {t: np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), dtype=np.int64) for t in grid}
    for x, y in loader:
        x = x.to(dev)
        with torch.amp.autocast('cuda', enabled=(C.USE_AMP and dev == 'cuda')):
            probs = F.softmax(model(x).float(), dim=1)
        conf, pred = probs.max(dim=1)
        for t in grid:
            p = torch.where(conf < t, torch.zeros_like(pred), pred).cpu().numpy()
            for i in range(p.shape[0]):
                cms[t] += M.confusion_matrix(p[i], y.numpy()[i], C.NUM_CLASSES)
    return [{'threshold': t, **M.metrics_from_confusion(cms[t])} for t in grid]


@torch.no_grad()
def save_worst_overlays(model, loader, class_names, out_dir, k=8):
    """Failure-case gallery: the k lowest-IoU images, saved as overlays."""
    from data_viz import overlay
    dev = _device()
    model = model.to(dev).eval()
    scored = []
    for bi, (x, y) in enumerate(loader):
        xd = x.to(dev)
        with torch.amp.autocast('cuda', enabled=(C.USE_AMP and dev == 'cuda')):
            pred = model(xd).float().argmax(dim=1).cpu().numpy()
        y_np = y.numpy()
        for i in range(pred.shape[0]):
            m = M.metrics_from_confusion(M.confusion_matrix(pred[i], y_np[i], C.NUM_CLASSES))
            scored.append((m['miou'], bi, i, x[i].numpy(), y_np[i], pred[i]))
    scored.sort(key=lambda r: r[0])
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for rank, (miou, bi, i, x_np, y_np, p_np) in enumerate(scored[:k]):
        p = os.path.join(out_dir, f'failure_{rank}_miou{miou:.3f}.png')
        overlay(x_np, y_np, p_np, class_names, p)
        paths.append(p)
    return paths