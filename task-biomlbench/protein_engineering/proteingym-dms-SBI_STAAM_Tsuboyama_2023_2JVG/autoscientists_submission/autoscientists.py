"""
Weighted ensemble of ESM-2 3B stacked (gpu6) and ESM-2 3B supervised (gpu3) submissions.
Weights: gpu6=0.66, gpu3=0.34
Champion: esm2_3b_embed_stacked_001 (0.8190)
This ensemble: 0.8224

Experiment: ensemble_weighted_gpu6_gpu3_001
"""

# This experiment averages predictions from two submissions:
# - gpu6: submission_esm2_3b_embed_stacked_001.csv (val_score=0.8190)
# - gpu3: submission_esm2_3b_embed_supervised_001.csv (val_score=0.7559)
# Optimal weights discovered by grid search: w6=0.66, w3=0.34

import pandas as pd
import numpy as np

W6 = 0.66
W3 = 0.34

df6 = pd.read_csv("submission_esm2_3b_embed_stacked_001.csv")
df3 = pd.read_csv("submission_esm2_3b_embed_supervised_001.csv")

score_cols = [c for c in df6.columns if c != "id"]
df_best = df6[["id"]].copy()
for col in score_cols:
    df_best[col] = W6 * df6[col].values + W3 * df3[col].values

df_best.to_csv("submission.csv", index=False)
print("Ensemble submission saved.")
