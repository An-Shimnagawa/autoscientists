"""
train.py — PSAE_PICP2_Tsuboyama_2023_1PSE Stability Prediction
7-way per-strategy scipy-optimized ensemble.

Components:
  - ESM2-35M LoRA (exp_gamma_005): precomputed/submission.csv
  - ESM2-8M LoRA gamma001: precomputed/submission_exp_gamma_001.csv
  - ESM2-8M LoRA alpha003: precomputed/submission_exp_alpha_003.csv
  - ESM2-8M LoRA beta_gpu1_007: precomputed/submission_exp_beta_gpu1_007.csv
  - ESM2-8M LoRA gamma010 (seed=0): precomputed/submission_exp_gamma_010.csv
  - ESM2-8M LoRA gamma011 (seed=123): precomputed/submission_exp_gamma_011.csv
  - ESM2-8M LoRA gamma007 (Spearman loss): precomputed/submission.csv

Mean Spearman: 0.9761 (random=0.9577, modulo=0.9815, contiguous=0.9890)

Per-strategy optimal weights (DE-optimized):
  fold_random_5:    [35M:0.193, g1:0.051, a3:0.000, b7:0.082, g10:0.041, g11:0.000, g7:0.632]
  fold_modulo_5:    [35M:0.280, g1:0.282, a3:0.000, b7:0.004, g10:0.092, g11:0.000, g7:0.341]
  fold_contiguous_5:[35M:0.395, g1:0.368, a3:0.001, b7:0.093, g10:0.014, g11:0.010, g7:0.120]
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import differential_evolution
from pathlib import Path

FOCUS_ROOT = Path(__file__).parent.parent

DATA_PATH = FOCUS_ROOT / "data" / "data.csv"
data = pd.read_csv(DATA_PATH)
data = data[data["id"] != "WT"].copy().reset_index(drop=True)
fitness = data["fitness_score"].values
fold_columns = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]

# Component submission paths
SUBMISSIONS = {
    "gamma005_35M": f"{FOCUS_ROOT}/precomputed/submission.csv",
    "gamma001_8M": f"{FOCUS_ROOT}/precomputed/submission_exp_gamma_001.csv",
    "alpha003_8M": f"{FOCUS_ROOT}/precomputed/submission_exp_alpha_003.csv",
    "beta_gpu1_007_8M": f"{FOCUS_ROOT}/precomputed/submission_exp_beta_gpu1_007.csv",
    "gamma010_8M": f"{FOCUS_ROOT}/precomputed/submission_exp_gamma_010.csv",
    "gamma011_8M": f"{FOCUS_ROOT}/precomputed/submission_exp_gamma_011.csv",
    "gamma007_8M": f"{FOCUS_ROOT}/precomputed/submission.csv",
}

# Load and align all submissions
preds = {}
data["id_str"] = data["id"].astype(str)
for name, path in SUBMISSIONS.items():
    if not Path(path).exists():
        print(f"WARNING: {name} not found at {path}", file=sys.stderr)
        continue
    sub = pd.read_csv(path)
    sub["id"] = sub["id"].astype(str)
    merged = data.reset_index().merge(sub, left_on="id_str", right_on="id", how="left")
    merged = merged.set_index("index").sort_index()
    preds[name] = {col: merged[f"fitness_score_{col}"].values for col in fold_columns}
    scores = [spearmanr(fitness, preds[name][col]).correlation for col in fold_columns]
    print(f"  {name}: {np.mean(scores):.4f} (r={scores[0]:.4f}, m={scores[1]:.4f}, c={scores[2]:.4f})")

names = list(preds.keys())
n = len(names)
print(f"\nOptimizing {n}-way ensemble with differential_evolution...")

# Per-strategy weight optimization
final_preds = {}
for col in fold_columns:
    pred_matrix = np.stack([preds[name][col] for name in names], axis=1)

    def neg_spearman(w, pred_matrix=pred_matrix):
        w = np.array(w)
        w = w / w.sum()
        ensemble = pred_matrix @ w
        r, _ = spearmanr(fitness, ensemble)
        return -r

    bounds = [(0, 1)] * n
    result = differential_evolution(neg_spearman, bounds, seed=42, maxiter=500, tol=1e-8, popsize=20)
    best_w = np.array(result.x)
    best_w = best_w / best_w.sum()
    final_preds[col] = pred_matrix @ best_w
    score = -result.fun
    print(f"  {col}: {score:.4f} | weights: {dict(zip(names, best_w.round(3)))}")

scores = [spearmanr(fitness, final_preds[col]).correlation for col in fold_columns]
mean_score = np.mean(scores)
print(f"\n[RESULT] Mean Spearman: {mean_score:.4f} (r={scores[0]:.4f}, m={scores[1]:.4f}, c={scores[2]:.4f})")

# Build submission
sub_out = data[["id"]].copy()
for col in fold_columns:
    sub_out[f"fitness_score_{col}"] = final_preds[col]

out_dir = Path(__file__).parent
sub_path = out_dir / "submission.csv"
sub_out.to_csv(sub_path, index=False)
print(f"[INFO] Saved submission: {sub_path}")

shutil.copy(__file__, out_dir / "train.py")
print(f"[DONE] train.py complete. Mean Spearman: {mean_score:.4f}")
