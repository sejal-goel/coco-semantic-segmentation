"""Small visualisation helpers (overlays, charts). Kept separate so the
heavy modules never import matplotlib."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


_PALETTE = np.array([
    [0, 0, 0], [220, 20, 60], [0, 0, 230], [255, 140, 0], [0, 180, 0],
    [148, 0, 211], [0, 191, 255], [255, 215, 0], [139, 69, 19], [255, 20, 147],
    [46, 139, 87], [70, 130, 180], [188, 143, 143]], dtype=np.uint8)


def colorize(mask):
    return _PALETTE[np.asarray(mask) % len(_PALETTE)]


def overlay(img_chw_float, gt, pred, class_names, path, title=''):
    """Denormalise an image tensor (C,H,W in ImageNet units) and save a
    3-panel figure: image, ground truth, prediction."""
    img = np.asarray(img_chw_float)
    img = img * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1) + np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    img = np.clip(img.transpose(1, 2, 0), 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img); axes[0].set_title('image')
    axes[1].imshow(colorize(gt)); axes[1].set_title('ground truth')
    axes[2].imshow(img); axes[2].imshow(colorize(pred), alpha=0.55)
    axes[2].set_title('prediction (overlay)')
    for ax in axes:
        ax.axis('off')
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def bar_chart(labels, values, path, title='', ylabel=''):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, values, color='#3b6ea5')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
