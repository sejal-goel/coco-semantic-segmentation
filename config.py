"""Central configuration for the whole project.

Every notebook imports this file, so all experiments share the same
classes, splits, sizes and hyperparameters (the brief requires an
identical pipeline for both models).

You are EXPECTED to change values here (see MAKE_IT_YOURS.md): input
size, epochs, learning rate, seeds, which losses the agent combines.
"""

import os
import json
import random
import numpy as np

# ---------------------------------------------------------------- paths (Kaggle)
WORK_DIR     = '/kaggle/working'
LOCAL_DS     = '/tmp/ds'                                     # fast local copy of images + masks
PROJECT_DIR  = os.path.join(WORK_DIR, 'project')             # everything saved here is kept as notebook output
CODE_DIR     = os.path.join(WORK_DIR, 'code')
RESULTS_DIR  = os.path.join(PROJECT_DIR, 'results')          # every metric lands here
FIGURES_DIR  = os.path.join(RESULTS_DIR, 'figures')
CKPT_DIR     = os.path.join(PROJECT_DIR, 'checkpoints')      # resume-safe checkpoints
MASKS_INDEX  = os.path.join(LOCAL_DS, 'masks_index.csv')     # photo -> mask mapping (absolute paths)
MASKS_DIR    = os.path.join(LOCAL_DS, 'masks')

# ---------------------------------------------------------------- data
CLASSES = ['person', 'car', 'dog', 'cat', 'chair', 'bottle',
           'bus', 'bicycle', 'horse', 'cup', 'laptop', 'bird']
NUM_CLASSES = 1 + len(CLASSES)          # 0 = background, 1..12 = classes above
CLASS_NAMES = ['background'] + CLASSES
IMG_SIZE = 512                          # all images resized to IMG_SIZE x IMG_SIZE
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------- training
# Identical pipeline for both models (brief requirement).
OPTIMIZER     = 'adamw'
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
LR_SCHEDULE   = 'poly'      # poly decay, power below; standard for segmentation
POLY_POWER    = 0.9
EPOCHS_MAIN   = 12          # full training runs
EPOCHS_SWEEP  = 6           # loss-comparison runs (same for every loss = fair)
EPOCHS_TRIAL  = 2           # each Optuna trial (cheap proxy; see agent_search docstring)
BATCH_SIZE    = {'deeplabv3plus': 8, 'segformer_b3': 4}
GRAD_ACCUM    = {'deeplabv3plus': 1, 'segformer_b3': 2}   # effective batch = BATCH*ACCUM
NUM_WORKERS   = 4
USE_AMP       = True        # mixed precision on GPU (T4-safe)

# ---------------------------------------------------------------- agent
AGENT_N_TRIALS    = 10        # compute budget for the composite-weight agent
AGENT_TRAIN_SUBSET = 1500     # images per trial (documented proxy, see report text)
THRESHOLD_GRID    = [round(0.05 * i, 2) for i in range(0, 14)]   # 0.00 .. 0.65
AGENT_SEED        = 42

# ---------------------------------------------------------------- seeds / final
SEEDS = [42, 1337, 2026]    # >= 3 seeds for mean +/- std (brief requirement)

MODELS = ['deeplabv3plus', 'segformer_b3']
LOSSES = ['ce', 'weighted_ce', 'focal', 'dice', 'tversky', 'lovasz']


def set_seed(seed):
    """Seed every RNG we use so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dirs():
    for d in (PROJECT_DIR, RESULTS_DIR, FIGURES_DIR, CKPT_DIR):
        os.makedirs(d, exist_ok=True)


def as_dict():
    """Full hyperparameter table, dumped into results/config.json (reproducibility)."""
    keys = [k for k in dir() if k.isupper()]
    return {k: globals()[k] for k in keys}


def save_config():
    ensure_dirs()
    with open(os.path.join(RESULTS_DIR, 'config.json'), 'w') as f:
        json.dump(as_dict(), f, indent=2, default=str)