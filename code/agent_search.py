"""Agent-based selection of the composite-loss weights and the confidence
threshold (Part C). This is the project's agent_search.py, adapted for Kaggle:

 * the Optuna study is stored in a small database file in results/, so a stopped
   session resumes instead of restarting
 * trial 0 is always the DEFAULT weights (equal weights), trained with exactly
   the same proxy budget as every agent-sampled trial, so "default vs agent
   selected" is measured under identical conditions. It counts toward the
   trial budget (config.AGENT_N_TRIALS trials in total, default included)
 * TPE uses n_startup_trials=4 random trials, then models the results. (Optuna's
   own default is 10 random trials, which would make a 10-trial budget pure
   random search.)
Only train and validation images are used. Test is never touched here.
"""
import os
import csv
import json
import time
import numpy as np

import config as C

N_STARTUP_TRIALS = 4


def threshold_agent(sweep_rows, model_name, results_dir=None):
    """Exhaustive search over config.THRESHOLD_GRID on validation mIoU."""
    results_dir = results_dir or C.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f'agent_threshold_trials_{model_name}.csv')
    keys = ['threshold', 'miou', 'mean_dice', 'mean_precision', 'mean_recall', 'pixel_accuracy']
    with open(log_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in sweep_rows:
            w.writerow({k: r.get(k) for k in keys})
    best = max(sweep_rows, key=lambda r: (r['miou'], -abs(r['threshold'])))
    default = next(r for r in sweep_rows if r['threshold'] == 0.0)
    choice = {'model': model_name, 'objective': 'val mIoU',
              'strategy': 'exhaustive grid search (agent)',
              'search_space': [r['threshold'] for r in sweep_rows],
              'n_trials': len(sweep_rows),
              'chosen_threshold': best['threshold'],
              'chosen_val_miou': best['miou'],
              'default_threshold': 0.0,
              'default_val_miou': default['miou'],
              'log_csv': log_path}
    with open(os.path.join(results_dir, f'agent_threshold_choice_{model_name}.json'), 'w') as f:
        json.dump(choice, f, indent=2)
    return choice


def _is_default(trial, names):
    vals = [trial.params.get(f'lambda_{k}') for k in names]
    return all(v is not None and abs(v - vals[0]) < 1e-12 for v in vals)


def composite_weight_agent(model_name, component_names, make_model_and_loaders,
                           results_dir=None, n_trials=None, seed=None, stop_check=None):
    import optuna
    import torch
    import engine
    import losses
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    TS = optuna.trial.TrialState
    results_dir = results_dir or C.RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    n_trials = n_trials or C.AGENT_N_TRIALS
    seed = C.AGENT_SEED if seed is None else seed
    db = os.path.join(results_dir, f'agent_weight_study_{model_name}.db')
    log_path = os.path.join(results_dir, f'agent_weight_trials_{model_name}.csv')
    names = list(component_names)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=N_STARTUP_TRIALS),
        study_name=f'composite_weights_{model_name}',
        storage=f'sqlite:///{db}', load_if_exists=True)
    for t in study.trials:                      # a trial cut off by a stopped session
        if t.state == TS.RUNNING:
            try:
                study.tell(t.number, state=TS.FAIL)
            except Exception as e:              # noqa: BLE001
                print('could not close stale trial', t.number, e)
    for leftover in os.listdir(C.CKPT_DIR):     # free disk from cut-off trials
        if f'{model_name}_agent_' in leftover and leftover.endswith('_ckpt.pt'):
            os.remove(os.path.join(C.CKPT_DIR, leftover))

    def run_trial(lambdas, tag, trial_seed):
        model, tl, vl = make_model_and_loaders()
        loss_fn = losses.CompositeLoss(
            {k: losses.build_loss(k, C.NUM_CLASSES) for k in names}, lambdas)
        t0 = time.time()
        history = engine.train(model, tag, tl, vl, loss_fn, epochs=C.EPOCHS_TRIAL,
                               seed=trial_seed, verbose=False)
        secs = time.time() - t0
        p = engine.ckpt_path(tag)
        if os.path.exists(p):
            os.remove(p)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return history[-1]['val_miou'], secs

    def n_done():
        return len([t for t in study.trials if t.state == TS.COMPLETE])

    # trial 0 = the default (equal) weights, same proxy budget as all other trials
    if stop_check and stop_check():
        print('[stop] not enough session time left to start the agent. Run again in a new session.')
        return None
    if not any(_is_default(t, names) for t in study.trials if t.state == TS.COMPLETE):
        lam = {k: 1.0 / len(names) for k in names}
        val, secs = run_trial(lam, f'{model_name}_agent_default', seed)
        dist = {f'lambda_{k}': optuna.distributions.FloatDistribution(0.0, 1.0) for k in names}
        study.add_trial(optuna.trial.create_trial(
            params={f'lambda_{k}': 1.0 for k in names}, distributions=dist, value=val,
            user_attrs={'seconds': secs}))
        print(f'{model_name} agent trial (default equal weights): val mIoU {val:.4f} ({secs:.0f}s)')

    def objective(trial):
        raw = {k: trial.suggest_float(f'lambda_{k}', 0.0, 1.0) for k in names}
        total = sum(raw.values()) or 1.0
        lam = {k: v / total for k, v in raw.items()}
        val, secs = run_trial(lam, f'{model_name}_agent_trial{trial.number}', seed + trial.number)
        trial.set_user_attr('seconds', secs)
        print(f'{model_name} agent trial {trial.number}: weights '
              f'{ {k: round(v, 3) for k, v in lam.items()} } val mIoU {val:.4f} ({secs:.0f}s)')
        return val

    while n_done() < n_trials:
        if stop_check and stop_check():
            print(f'[stop] not enough session time left; {n_done()}/{n_trials} trials done. Run again to continue.')
            break
        study.optimize(objective, n_trials=1)

    rows = []
    for t in study.trials:
        if t.state != TS.COMPLETE:
            continue
        raw = {k: t.params[f'lambda_{k}'] for k in names}
        total = sum(raw.values()) or 1.0
        rows.append({'trial': t.number, 'is_default': _is_default(t, names), 'val_miou': t.value,
                     **{f'weight_{k}': raw[k] / total for k in names},
                     'seconds': t.user_attrs.get('seconds')})
    if rows:
        with open(log_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if n_done() < n_trials:
        return None
    best = study.best_trial
    raw = {k: best.params[f'lambda_{k}'] for k in names}
    total = sum(raw.values()) or 1.0
    default_row = next(r for r in rows if r['is_default'])
    choice = {'model': model_name, 'objective': 'val mIoU (proxy: subset training, subset validation)',
              'strategy': f'Bayesian (Optuna TPE, seed {seed}, {N_STARTUP_TRIALS} random start-up trials)',
              'components': names,
              'n_trials': n_trials,
              'default_trial_included_in_budget': True,
              'compute_budget_per_trial': f'{C.AGENT_TRAIN_SUBSET} train images, {C.EPOCHS_TRIAL} epochs, '
                                          f'300 validation images',
              'stopping_rule': 'fixed trial budget',
              'chosen_trial': best.number,
              'chosen_weights': {k: v / total for k, v in raw.items()},
              'chosen_val_miou': best.value,
              'default_weights': {k: 1.0 / len(names) for k in names},
              'default_val_miou': default_row['val_miou'],
              'chosen_is_default': best.number == default_row['trial'],
              'log_csv': log_path}
    with open(os.path.join(results_dir, f'agent_weight_choice_{model_name}.json'), 'w') as f:
        json.dump(choice, f, indent=2)
    return choice
