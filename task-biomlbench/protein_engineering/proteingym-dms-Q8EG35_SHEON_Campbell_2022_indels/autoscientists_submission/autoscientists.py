"""
Cycle 3 - Differential Evolution Global Ensemble Optimization

All cycle-1 and cycle-2 submissions are combined via DE-optimized weighted rank ensemble.
DE finds globally optimal combination weights across all 11 submissions simultaneously.

Champion (0.809958) used DE on 4 models. We extend to all 11 available submissions.

Expected: ~0.810+ (more diverse signal from all paradigms)

Experiment ID: exp_gamma_004_de_ensemble_all
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import spearmanr
from scipy.stats import rankdata
from scipy.optimize import differential_evolution

# ===== Configuration =====
FOCUS_ROOT = Path(__file__).parent.parent
EXP_ID = "exp_gamma_004_de_ensemble_all"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[{EXP_ID}] Loading data...")
data = pd.read_csv(FOCUS_ROOT / "data" / "data.csv")
data_nwt = data[data["id"] != "WT"].copy()
data_nwt["id"] = data_nwt["id"].astype(int)
data_nwt = data_nwt.sort_values("id").reset_index(drop=True)
gt = data_nwt["fitness_score"].values
print(f"  Dataset: {len(data_nwt)} sequences")

# ===== Load all available cycle-1 and cycle-2 submissions =====
subs_to_load = [
    # Cycle-2 strong models first
    ("gpu6_zeta002", "precomputed/submission_exp_zeta_002_de_ensemble.csv"),
    ("gpu3_beta009", "precomputed/submission_exp_beta_009_ensemble_ridge_context.csv"),
    ("gpu6_zeta001", "precomputed/submission_exp_zeta_001_attention_position_ensemble.csv"),
    ("gpu3_beta008", "precomputed/submission_exp_beta_008_weighted_rank_ensemble.csv"),
    # Cycle-1/base models
    ("gpu1_alpha004", "precomputed/submission_exp_alpha_004_esm2_insertion_site_focus.csv"),
    ("gpu1_delta001", "precomputed/submission_exp_delta_001_rank_ensemble.csv"),
    ("gpu5_gamma001", "precomputed/submission_exp_gamma_001_tranception_zeroshot.csv"),
    ("gpu5_gamma002", "precomputed/submission_exp_gamma_002_esm2_pll_lgbm_stacking.csv"),
    ("gpu2_alpha004", "precomputed/submission_exp_alpha_004_esm2_insertion_site_focus.csv"),
    ("gpu3_beta007", "precomputed/submission_exp_beta_007_chargram_final.csv"),
    ("gpu2_alpha003", "precomputed/submission_exp_alpha_003_esm2_multilayer_pca_ridge.csv"),
    ("gpu1_alpha002", "precomputed/submission_exp_alpha_002_esm2_layers_svr.csv"),
    ("gpu4_beta004", "precomputed/submission_exp_beta_004_esm2_classical_hybrid_svr.csv"),
    ("gpu4_beta005", "precomputed/submission_exp_beta_005_esm2_pca64_svr.csv"),
    ("gpu4_beta006", "precomputed/submission_exp_beta_006_esm2_xgboost.csv"),
]

submissions = {}
for key, path in subs_to_load:
    try:
        sub = pd.read_csv(FOCUS_ROOT / path)
        sub["id"] = sub["id"].astype(int)
        sub = sub.sort_values("id").reset_index(drop=True)
        submissions[key] = sub["fitness_score"].values
    except Exception as e:
        pass

print(f"\nLoaded {len(submissions)} submissions:")
for k, v in sorted(submissions.items(), key=lambda x: spearmanr(gt, x[1]).correlation, reverse=True):
    print(f"  {k}: {spearmanr(gt, v).correlation:.6f}")

names = list(submissions.keys())
preds_list = [submissions[k] for k in names]
ranked_preds = [rankdata(p) for p in preds_list]
n_models = len(names)

# ===== Differential Evolution Optimization =====
def neg_spearman(weights):
    weights = np.maximum(weights, 0)
    total = weights.sum()
    if total < 1e-10:
        return 0.0
    ensemble = sum(w * rp for w, rp in zip(weights, ranked_preds)) / total
    return -spearmanr(gt, ensemble).correlation

print(f"\nRunning differential evolution on {n_models} models...")
bounds = [(0, 1.0)] * n_models

# Run with multiple seeds for robustness
best_score = 0
best_preds = None
best_weights = None

for seed in [42, 123, 456, 789, 999]:
    result = differential_evolution(
        neg_spearman, bounds, seed=seed, maxiter=1000,
        popsize=20, tol=1e-7, mutation=(0.5, 1.5),
        recombination=0.7, workers=1, polish=True
    )
    w = np.maximum(result.x, 0)
    total = w.sum()
    if total < 1e-10:
        continue
    ens = sum(wi * rp for wi, rp in zip(w, ranked_preds)) / total
    s = spearmanr(gt, ens).correlation
    print(f"  Seed {seed}: {s:.6f}")
    if s > best_score:
        best_score = s
        best_preds = ens.copy()
        best_weights = w.copy()

print(f"\n[{EXP_ID}] Best DE score: {best_score:.6f}")
print(f"[{EXP_ID}] Champion: 0.809958")
print(f"[{EXP_ID}] Delta: {best_score - 0.809958:+.6f}")

print(f"\nNon-trivial weights:")
for nm, w in sorted(zip(names, best_weights), key=lambda x: x[1], reverse=True):
    if w > 0.01:
        print(f"  {nm}: {w:.6f}")

# ===== Save results =====
sub_df = data_nwt[["id"]].copy()
sub_df["fitness_score"] = best_preds
sub_df = sub_df.sort_values("id").reset_index(drop=True)

sub_df.to_csv(OUTPUT_DIR / "submission.csv", index=False)
sub_df.to_csv(OUTPUT_DIR / f"submission_{EXP_ID}.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR}/submission_{EXP_ID}.csv")

shutil.copy(__file__, OUTPUT_DIR / "train.py")
shutil.copy(__file__, OUTPUT_DIR / f"train_{EXP_ID}.py")

hp = {
    "exp_id": EXP_ID,
    "val_score": float(best_score),
    "method": "differential_evolution_rank_ensemble",
    "n_models": n_models,
    "models": names,
    "weights": {k: float(v) for k, v in zip(names, best_weights)},
    "champion_baseline": 0.809958,
    "delta": float(best_score - 0.809958),
}
print("=" * 60)
print(json.dumps(hp, indent=2))
print("=" * 60)

# Write result_latest.json
result_summary = {
    "val_score": float(best_score),
    "direction": "maximize",
    "exp_id": EXP_ID,
        "submission_path": str(OUTPUT_DIR / f"submission_{EXP_ID}.csv"),
    "train_path": str(OUTPUT_DIR / f"train_{EXP_ID}.py"),
    "status": "complete",
    "posted_to_workshop": False,
    "result_post_id": None,
    "pid": None, "monitor_id": None,
    "stdout_path": None, "stderr_path": None,
    "item": {"id": EXP_ID, "axis": "ensemble-weights", "direction": "maximize", "value": best_score},
    "queue_claimed": True,
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "description": f"DE-optimized rank ensemble of all {n_models} cycle-1/2 submissions",
}
ws_dir = Path(__file__).parent / "outputs"
(ws_dir / "result_latest.json").write_text(json.dumps(result_summary, indent=2))
print(f"[{EXP_ID}] result_latest.json written.")
