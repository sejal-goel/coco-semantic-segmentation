"""Loss functions (Part C).

Six individual losses from different families, plus a composite loss
L_total = sum_k lambda_k * L_k  whose weights the agent tunes.

  ce          Cross-entropy            (distribution-based; the default)
  weighted_ce CE with median-frequency class weights (distribution; imbalance)
  focal       Focal loss               (distribution; focuses hard pixels)
  dice        Soft Dice loss           (region-based; matches Dice metric)
  tversky     Tversky loss             (region-based; FP/FN trade-off knob)
  lovasz      Lovasz-Softmax           (metric-based; directly optimises IoU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = -1


# ------------------------------------------------------------- helpers
def _one_hot(target, num_classes, ignore_index=IGNORE_INDEX):
    """(N,H,W) int -> (N,C,H,W) float one-hot, ignoring ignore_index."""
    valid = target != ignore_index
    t = target.clone()
    t[~valid] = 0
    oh = F.one_hot(t, num_classes).permute(0, 3, 1, 2).float()
    return oh * valid.unsqueeze(1).float(), valid


class CrossEntropy(nn.Module):
    def __init__(self, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, target):
        return self.ce(logits, target)


class WeightedCrossEntropy(nn.Module):
    """CE weighted per class, e.g. median-frequency weights from the
    class-pixel distribution (a data-side answer to imbalance)."""
    def __init__(self, class_weights, ignore_index=IGNORE_INDEX):
        super().__init__()
        w = torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer('w', w)
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, weight=self.w,
                               ignore_index=self.ignore_index)


class FocalLoss(nn.Module):
    """Down-weights easy pixels so the model focuses on hard ones
    (Lin et al., 2017). Good for dominant-background imbalance."""
    def __init__(self, gamma=2.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none',
                             ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        valid = target != self.ignore_index
        loss = ((1 - pt) ** self.gamma) * ce
        return loss[valid].mean()


class DiceLoss(nn.Module):
    """Soft Dice: 1 - mean_c (2|p_c ∩ y_c| + eps) / (|p_c| + |y_c| + eps).
    Directly optimises a soft version of the Dice/F1 metric."""
    def __init__(self, smooth=1.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        C = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        oh, valid = _one_hot(target, C, self.ignore_index)
        dims = (0, 2, 3)
        inter = (probs * oh).sum(dims)
        union = probs.sum(dims) + oh.sum(dims)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TverskyLoss(nn.Module):
    """Generalises Dice with separate FP/FN weights. beta > alpha
    penalises false negatives more (recall-leaning)."""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        C = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        oh, valid = _one_hot(target, C, self.ignore_index)
        dims = (0, 2, 3)
        tp = (probs * oh).sum(dims)
        fp = (probs * (1 - oh)).sum(dims)
        fn = ((1 - probs) * oh).sum(dims)
        tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tv.mean()


# ------------------------------------------------------------- Lovasz
def _lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension of the IoU w.r.t. sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_softmax_flat(probs, labels):
    """probs (P,C) softmaxed, labels (P,) int."""
    C = probs.shape[1]
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probs[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg[perm])))
    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


class LovaszSoftmax(nn.Module):
    """Lovasz-Softmax (Berman et al., 2018): a convex surrogate that
    optimises the IoU (Jaccard) metric directly."""
    def __init__(self, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        probs = F.softmax(logits, dim=1)
        probs = probs.permute(0, 2, 3, 1).reshape(-1, probs.shape[1])
        labels = target.reshape(-1)
        valid = labels != self.ignore_index
        return _lovasz_softmax_flat(probs[valid], labels[valid])


class CompositeLoss(nn.Module):
    """L_total = sum_k lambda_k * L_k. The agent (code/agent_search.py)
    chooses the lambdas on the validation set, never on test."""
    def __init__(self, components, weights):
        super().__init__()
        assert set(components) == set(weights)
        self.names = list(components)
        self.components = nn.ModuleDict(components)
        self.weights = {k: float(v) for k, v in weights.items()}

    def forward(self, logits, target):
        total = 0.0
        for k in self.names:
            total = total + self.weights[k] * self.components[k](logits, target)
        return total

    def describe(self):
        return {k: self.weights[k] for k in self.names}


def median_frequency_weights(counts, num_classes):
    """class weight = median(freq) / freq, freq = pixel share of the class.
    Rare classes get large weights. Background is included."""
    counts = np.asarray(counts, dtype=np.float64) + 1.0
    freq = counts / counts.sum()
    w = np.median(freq) / freq
    return (w / w.mean()).tolist()          # normalise so mean weight = 1


import numpy as np  # noqa: E402  (kept low so the torch API above reads first)

LOSS_REGISTRY = {
    'ce':          lambda C, w=None: CrossEntropy(),
    'weighted_ce': lambda C, w=None: WeightedCrossEntropy(w),
    'focal':       lambda C, w=None: FocalLoss(gamma=2.0),
    'dice':        lambda C, w=None: DiceLoss(),
    'tversky':     lambda C, w=None: TverskyLoss(alpha=0.3, beta=0.7),
    'lovasz':      lambda C, w=None: LovaszSoftmax(),
}


def build_loss(name, num_classes, class_weights=None):
    return LOSS_REGISTRY[name](num_classes, class_weights)