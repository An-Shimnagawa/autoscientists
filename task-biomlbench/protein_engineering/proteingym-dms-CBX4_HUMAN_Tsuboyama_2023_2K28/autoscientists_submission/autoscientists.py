"""
Champion: exp_emb_meta_blend_027
Score: 0.956765
Method source: ensemble blend of embedding-based stacking and mutation-stratified model
Method: Optimal 2-way blend: super_stack_018 (0.8274) + mut_strat_023 (0.1726)
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

FOCUS_ROOT = Path(__file__).parent.parent
DATA_PATH = FOCUS_ROOT / "data" / "data.csv"
R3 = Path(__file__).parent / "outputs"
R5 = Path(__file__).parent / "outputs"

fold_columns = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
pred_cols = [f"fitness_score_{c}" for c in fold_columns]

data_full = pd.read_csv(str(DATA_PATH))
data = data_full[data_full["id"] != "WT"].copy().reset_index(drop=True)
data["id"] = data["id"].astype(str)
y = data["fitness_score"].values
N = len(data)

def load(path):
    df = pd.read_csv(str(path)); df["id"] = df["id"].astype(str)
    return data.merge(df[["id"] + pred_cols], on="id", how="inner")[pred_cols].values

ss018 = load(R3 / "submission_exp_emb_super_stack_018.csv")
ms023 = load(R5 / "submission_exp_emb_mut_strat_023.csv")
final_preds = 0.8274 * ss018 + 0.1726 * ms023

score = float(np.mean([spearmanr(y, final_preds[:, i]).correlation for i in range(3)]))
print(f"Score: {score:.6f}")

pd.DataFrame([{"id": data["id"].iloc[i],
               "fitness_score_fold_random_5": float(final_preds[i,0]),
               "fitness_score_fold_modulo_5": float(final_preds[i,1]),
               "fitness_score_fold_contiguous_5": float(final_preds[i,2])}
              for i in range(N)]).to_csv(str(FOCUS_ROOT / "submission.csv"), index=False)
print("Done.")
