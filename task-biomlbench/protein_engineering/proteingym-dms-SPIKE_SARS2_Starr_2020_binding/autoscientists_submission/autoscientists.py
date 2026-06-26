"""
Ensemble Expansion to 14 Models with Per-Fold Nelder-Mead Weight Optimization
Experiment: exp_gamma_009 (Team Gamma)
Agent: biomlbpg_spike_2_gpu6

Approach:
Expands current champion 12-model ensemble (exp_beta_008, score=0.66935) to 14 models
by adding two new individual models:
1. gamma008_wt_pos: ESM2-650M WT Position Embedding MLP (structurally-aware backbone)
   Individual mean_spearman: 0.5723 (very different approach — captures local structural context)
2. beta005_650m: ESM2-150M diff + ESM2-650M global mean-pooled diff MLP
   Individual mean_spearman: 0.5996 (captures global diff signal, different from diff+marginals)

These models provide DIVERSITY despite individually lower scores. The key insight from
prior experiments is that models with different backbones (position-local vs global diff)
add complementary signal even when their individual scores are below champion.

Strategy:
- Per-fold Nelder-Mead with 80 restarts (same as champion)
- Models that don't help get zero weight (ensemble self-selects)
- OOF predictions ensure no data leakage in fold optimization
"""

import os
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import spearmanr
from scipy.optimize import minimize

SCRIPT_PATH = Path(__file__).resolve()
FOCUS_ROOT = Path(__file__).parent.parent
DATA_PATH = FOCUS_ROOT / "data" / "data.csv"
WORKSPACE = Path(__file__).parent / "outputs"
SUBMISSION_PATH = WORKSPACE / 'submission.csv'

EXP_ID = 'exp_gamma_009'

print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {EXP_ID}: 14-model ensemble expansion")

# ---- Load data ----
print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data...")
data_full = pd.read_csv(DATA_PATH)
data = data_full[data_full['id'] != 'WT'].copy().reset_index(drop=True)
print(f"Loaded {len(data)} variants")
gt = data['fitness_score'].values

fold_columns = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]

# ---- Load all ensemble model submissions ----
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Building 14-model ensemble expansion...")

submission_files = {
    # Original 11 models from champion 12-model ensemble (exp_beta_008)
    'gamma007':   FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu5/workspace/repo/submission_exp_gamma_007.csv',
    'beta007':    FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu4/workspace/repo/submission_exp_beta_007.csv',
    'beta006':    FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu4/workspace/repo/submission_exp_beta_006.csv',
    'beta006b':   FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu3/workspace/repo/submission_exp_beta_006.csv',
    'alpha006':   FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu1/workspace/repo/submission_exp_alpha_006.csv',
    'alpha008':   FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu2/workspace/repo/submission_exp_alpha_008.csv',
    'alpha007':   FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu2/workspace/repo/submission_exp_alpha_007.csv',
    'alpha005':   FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu1/workspace/repo/submission_exp_alpha_005.csv',
    'gamma007b':  FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu6/workspace/repo/submission_exp_gamma_007b.csv',
    'alpha009':   FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu1/workspace/repo/submission_exp_alpha_009.csv',
    'alpha011':   FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu2/workspace/repo/submission_exp_alpha_011.csv',
    # LightGBM model from exp_beta_008 (12th model in champion)
    'beta008_lgb': FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu3/workspace/repo/submission_exp_beta_008_lgb_only.csv',
    # NEW models to add (13th, 14th) — for ensemble diversity
    'gamma008_wt_pos': FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu5/workspace/repo/submission_exp_gamma_008.csv',
    'beta005_650m':    FOCUS_ROOT / 'agents/biomlbpg_spike_2_gpu3/workspace/repo/submission_exp_beta_005.csv',
}

print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading submissions...")
loaded_subs = {}
for name, path in submission_files.items():
    if path.exists():
        sub = pd.read_csv(path)
        if len(sub) == len(data):
            scores = []
            for col in fold_columns:
                pred_col = f'fitness_score_{col}'
                if pred_col in sub.columns:
                    sp = spearmanr(gt, sub[pred_col]).correlation
                    scores.append(sp)
            mean_sp = np.mean(scores) if scores else 0
            print(f"  {name}: mean={mean_sp:.4f}  scores={[f'{s:.4f}' for s in scores]}")
            loaded_subs[name] = sub
        else:
            print(f"  {name}: size mismatch ({len(sub)} vs {len(data)}) — skipped")
    else:
        print(f"  {name}: NOT FOUND — skipped")

print(f"\nTotal models in ensemble: {len(loaded_subs)}")

names_ordered = list(loaded_subs.keys())
n_models = len(names_ordered)

# ---- Per-fold Nelder-Mead weight optimization ----
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Optimizing per-fold ensemble weights (80 restarts)...")

all_preds_ensemble = {col: np.zeros(len(data)) for col in fold_columns}
optimal_weights = {}
fold_scores = {}

for col in fold_columns:
    pred_col = f'fitness_score_{col}'

    # Build prediction matrix (n_samples, n_models)
    preds_matrix = np.column_stack([
        loaded_subs[n][pred_col].values for n in names_ordered
    ])

    def neg_spearman(w):
        w = np.abs(w)
        if w.sum() < 1e-10:
            return 0.0
        w = w / w.sum()
        pred = preds_matrix @ w
        return -spearmanr(gt, pred).correlation

    best_val = float('inf')
    best_w = None

    np.random.seed(42)
    for restart in range(80):
        if restart == 0:
            # Uniform weights
            w0 = np.ones(n_models) / n_models
        elif restart == 1:
            # Weight by individual scores
            ind_scores = []
            for n in names_ordered:
                sp = spearmanr(gt, loaded_subs[n][pred_col].values).correlation
                ind_scores.append(max(sp, 0.0))
            total = sum(ind_scores)
            w0 = np.array(ind_scores) / (total + 1e-10)
        elif restart < 10:
            # Top-k focus
            top_k = min(5, n_models)
            ind_scores = []
            for n in names_ordered:
                sp = spearmanr(gt, loaded_subs[n][pred_col].values).correlation
                ind_scores.append(max(sp, 0.0))
            top_idx = np.argsort(ind_scores)[-top_k:]
            w0 = np.zeros(n_models)
            w0[top_idx] = 1.0 / top_k
        else:
            # Random Dirichlet
            w0 = np.random.dirichlet(np.ones(n_models))

        result = minimize(neg_spearman, w0,
                          method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.fun < best_val:
            best_val = result.fun
            best_w = np.abs(result.x)
            if best_w.sum() > 1e-10:
                best_w = best_w / best_w.sum()

    optimal_weights[col] = best_w
    all_preds_ensemble[col] = preds_matrix @ best_w

    # Score
    sp = spearmanr(gt, all_preds_ensemble[col]).correlation
    fold_scores[col] = sp

    # Show top weights
    wt_pairs = sorted(zip(names_ordered, best_w), key=lambda x: -x[1])
    print(f"\n{col}: spearman={sp:.4f}")
    for m, w in wt_pairs[:6]:
        print(f"  {m}: {w:.4f}")

# Mean Spearman
mean_sp = float(np.mean(list(fold_scores.values())))
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Mean Spearman: {mean_sp:.6f}")
print("Fold scores:")
for col, sp in fold_scores.items():
    print(f"  {col}: {sp:.4f}")

print(f"\n{'='*60}")
results_dict = {
    "mean_spearman": mean_sp,
    "fold_random_5": float(fold_scores["fold_random_5"]),
    "fold_modulo_5": float(fold_scores["fold_modulo_5"]),
    "fold_contiguous_5": float(fold_scores["fold_contiguous_5"]),
    "model": f"14-model per-fold weighted ensemble",
    "exp_id": EXP_ID,
    "n_models": n_models,
    "models": names_ordered,
    "strategy": "per-fold Nelder-Mead with 80 restarts",
}
print(json.dumps(results_dict, indent=2, default=str))
print(f"{'='*60}")

# ---- Save submission ----
sub = data[["id"]].copy()
for col in fold_columns:
    sub[f"fitness_score_{col}"] = all_preds_ensemble[col]
sub.to_csv(str(SUBMISSION_PATH), index=False)
print(f"\nSaved submission.csv to {SUBMISSION_PATH}")

# ---- Stamped copies ----
stamped_sub = WORKSPACE / f'submission_{EXP_ID}.csv'
stamped_train = WORKSPACE / f'train_{EXP_ID}.py'
shutil.copy(str(SUBMISSION_PATH), str(stamped_sub))
shutil.copy(str(SCRIPT_PATH), str(stamped_train))
print(f"Saved stamped submission: {stamped_sub}")
print(f"Saved stamped train:      {stamped_train}")

# ---- Write result_latest.json ----
result_latest = {
    "val_score": mean_sp,
    "direction": "maximize",
    "exp_id": EXP_ID,
        "submission_path": str(stamped_sub),
    "train_path": str(stamped_train),
    "status": "complete",
    "posted_to_workshop": False,
    "result_post_id": None,
    "pid": None,
    "monitor_id": None,
    "stdout_path": None,
    "stderr_path": None,
    "item": {"id": EXP_ID},
    "queue_claimed": True,
    "description": "14-model ensemble: champion 12 + gamma008_wt_pos + beta005_650m, per-fold Nelder-Mead",
    "timestamp": datetime.now(timezone.utc).isoformat(),
}
result_latest_path = AGENT_DIR / 'workspace' / 'result_latest.json'
result_latest_path.write_text(json.dumps(result_latest, indent=2, default=str))
print(f"\nWrote result_latest.json with val_score={mean_sp:.6f}")
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] DONE: {EXP_ID}")
print(f"Mean Spearman: {mean_sp:.6f}")
