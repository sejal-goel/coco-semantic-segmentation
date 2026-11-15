"""Evaluation metrics - pure NumPy so they can be tested without a GPU.

Implements exactly the formulas in the project brief (Part D):
per-class IoU, mIoU, Dice/F1, Precision, Recall, Pixel Accuracy,
plus a boundary metric (Boundary F1 with a pixel tolerance) and a
pixel-level confusion matrix for the error analysis.
"""

import numpy as np
from scipy import ndimage


def confusion_matrix(pred, target, num_classes):
    """Pixel-level confusion matrix. Rows = true class, columns = predicted."""
    pred = np.asarray(pred).ravel().astype(np.int64)
    target = np.asarray(target).ravel().astype(np.int64)
    valid = (target >= 0) & (target < num_classes) & (pred >= 0) & (pred < num_classes)
    cm = np.bincount(num_classes * target[valid] + pred[valid],
                     minlength=num_classes ** 2)
    return cm.reshape(num_classes, num_classes)


def metrics_from_confusion(cm):
    """All Part-D metrics from one confusion matrix."""
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    def safe_div(a, b):
        out = np.full_like(a, np.nan, dtype=np.float64)
        np.divide(a, b, out=out, where=(b > 0))
        return out

    iou = safe_div(tp, tp + fp + fn)
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    pixel_acc = tp.sum() / cm.sum() if cm.sum() > 0 else float('nan')

    present = (tp + fn) > 0          # classes that actually occur in the ground truth
    return {
        'per_class_iou': iou.tolist(),
        'per_class_dice': dice.tolist(),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'miou': float(np.nanmean(iou[present])) if present.any() else float('nan'),
        'mean_dice': float(np.nanmean(dice[present])) if present.any() else float('nan'),
        'mean_precision': float(np.nanmean(precision[present])) if present.any() else float('nan'),
        'mean_recall': float(np.nanmean(recall[present])) if present.any() else float('nan'),
        'pixel_accuracy': float(pixel_acc),
    }


def _boundary(binary_mask):
    """1-pixel inner boundary of a binary mask."""
    eroded = ndimage.binary_erosion(binary_mask, border_value=0)
    return binary_mask & ~eroded


def boundary_f1(pred, target, num_classes, tolerance=2):
    """Boundary F1: a predicted boundary pixel counts as correct if it lies
    within `tolerance` pixels of a ground-truth boundary pixel (and vice
    versa). Averaged over classes present in target or prediction."""
    scores = []
    for c in range(num_classes):
        gt = target == c
        pr = pred == c
        if not gt.any() and not pr.any():
            continue
        gt_b, pr_b = _boundary(gt), _boundary(pr)
        if not gt_b.any() and not pr_b.any():
            continue
        if not gt_b.any() or not pr_b.any():
            scores.append(0.0)
            continue
        dist_gt = ndimage.distance_transform_edt(~gt_b)
        dist_pr = ndimage.distance_transform_edt(~pr_b)
        tp_p = (dist_gt[pr_b] <= tolerance).sum()   # pred boundary close to gt
        tp_r = (dist_pr[gt_b] <= tolerance).sum()   # gt boundary close to pred
        prec = tp_p / pr_b.sum()
        rec = tp_r / gt_b.sum()
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        scores.append(f1)
    return float(np.mean(scores)) if scores else float('nan')


def full_report(pred, target, num_classes, class_names, tolerance=2):
    """One image pair -> complete metric dictionary."""
    cm = confusion_matrix(pred, target, num_classes)
    m = metrics_from_confusion(cm)
    m['boundary_f1'] = boundary_f1(pred, target, num_classes, tolerance)
    m['class_names'] = list(class_names)
    return m


def mean_std(dicts, keys):
    """Mean +/- std over seed runs, as {'key': (mean, std)}."""
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if d.get(k) is not None and not np.isnan(d.get(k, np.nan))]
        if vals:
            out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out