"""Reading and writing the results/ folder. Every experiment saves a
small JSON here; the report and slides are generated from these files,
so no number in the report is typed by hand (or invented)."""

import os
import json
import glob
import numpy as np

import config as C


def save_result(name, payload, results_dir=None):
    results_dir = results_dir or C.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=_jsonable)
    return path


def load_result(name, results_dir=None):
    results_dir = results_dir or C.RESULTS_DIR
    path = os.path.join(results_dir, f'{name}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_all(results_dir=None):
    results_dir = results_dir or C.RESULTS_DIR
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        name = os.path.splitext(os.path.basename(p))[0]
        with open(p) as f:
            out[name] = json.load(f)
    return out


def mean_std_over_seeds(records, key):
    """records: list of per-seed dicts -> 'mean +/- std' (and the raw list)."""
    vals = [r[key] for r in records if r and r.get(key) is not None]
    if not vals:
        return None
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
            'n_seeds': len(vals), 'values': [float(v) for v in vals]}


def status(results_dir=None):
    """Which result files exist and which are still missing - printed by the
    notebooks and used to stop the report silently using placeholders."""
    results_dir = results_dir or C.RESULTS_DIR
    expected = (['models', 'dataset_stats', 'baseline_slic_rf', 'config']
                + [f'loss_sweep_{m}' for m in C.MODELS]
                + [f'agent_threshold_choice_{m}' for m in C.MODELS]
                + [f'agent_weight_choice_{m}' for m in C.MODELS]
                + [f'ablations_{m}' for m in C.MODELS]
                + [f'test_final_{m}' for m in C.MODELS])
    have = {os.path.splitext(os.path.basename(p))[0]
            for p in glob.glob(os.path.join(results_dir, '*.json'))}
    return {name: (name in have) for name in expected}


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
