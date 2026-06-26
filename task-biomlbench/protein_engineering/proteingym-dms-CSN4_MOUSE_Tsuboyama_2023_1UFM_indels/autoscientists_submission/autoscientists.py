#!/usr/bin/env python3
# Champion: 4-model rank^2 ensemble (exhaustive-optimal)
# Score: 0.9319 (global Spearman on 196 non-WT variants)
# Models: exp_delta_006 (ESM2+ESM1v), exp_alpha_002, exp_beta_003, exp_beta_005 (ESM2-35M)
import pandas as pd
import numpy as np
from scipy.stats import rankdata

BASE = str(Path(__file__).parent.parent)

subs = [
    str(Path(BASE) / "precomputed" / "submission_exp_delta_006.csv"),  # ESM2+ESM1v hybrid, 0.9155
    str(Path(BASE) / "precomputed" / "submission_exp_alpha_002.csv"),  # ESM2-650M LoRA, 0.9107
    str(Path(BASE) / "precomputed" / "submission_exp_beta_003.csv"),   # ESM2 variant, 0.9000
    str(Path(BASE) / "precomputed" / "submission_exp_beta_005.csv"),   # ESM2-35M LoRA, 0.8969
]

dfs = []
for path in subs:
    df = pd.read_csv(path)
    df["id"] = df["id"].astype(str)
    df = df[df["id"] != "WT"]
    dfs.append(df)

base = dfs[0][["id"]].copy()
for i, df in enumerate(dfs):
    base = base.merge(df.rename(columns={"fitness_score": f"s{i}"}), on="id")

rank2_avg = np.column_stack([rankdata(base[f"s{i}"].values)**2 for i in range(len(dfs))]).mean(axis=1)
out = base[["id"]].copy()
out["fitness_score"] = rank2_avg

wt_row = pd.DataFrame([{"id": "WT", "fitness_score": -1.9301334891696365}])
out = pd.concat([wt_row, out], ignore_index=True)
out.to_csv(f"{BASE}/task/submission.csv", index=False)
print(f"Saved submission shape: {out.shape}")
