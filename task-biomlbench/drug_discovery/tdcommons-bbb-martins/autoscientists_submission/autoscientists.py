"""
Stacking meta-learner: proper 2nd-level model using OOF predictions.

Experiment: exp_gamma_cycle4_001
Agent: biomlbtdc_bbbm_4_gpu4
Team: gamma

Approach:
- Level 1: XGBoost+Mordred (best single model at 0.9034)
  + additional trees: ExtraTrees, RandomForest — all with Mordred
- Level 2: Logistic regression meta-learner trained on OOF predictions
- This gives a *proper* measured CV on stacked predictions

If we can get OOF from multiple models and stack them, the measured CV
from the meta-learner IS the true CV (no estimation needed).
"""

import os, sys, json, shutil
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"EXP_ID = "exp_gamma_cycle4_001"
AGENT_WORKSPACE = str(Path(__file__).parent / "outputs")
os.makedirs(AGENT_WORKSPACE, exist_ok=True)

print(f"=== {EXP_ID} ===")
print(f"Python: {sys.executable}")

with open(claim_path, 'w') as f:
    f.write('cpu')
print("GPU claim: cpu")

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test_features.csv")
print(f"Train: {len(train_df)}, Test: {len(test_df)}")

from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from mordred import Calculator, descriptors as mordred_descriptors
    USE_MORDRED = True
    print("Mordred available")
except ImportError:
    USE_MORDRED = False

MORDRED_VALID_COLS = None

def compute_features(smiles_list):
    global MORDRED_VALID_COLS
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = []
    for mol in mols:
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            fps.append(list(fp))
        else:
            fps.append([0] * 2048)
    fps = np.array(fps, dtype=np.float32)

    if USE_MORDRED:
        calc = Calculator(mordred_descriptors, ignore_3D=True)
        desc_list = []
        for mol in mols:
            if mol is not None:
                try:
                    desc = calc(mol)
                    desc_vals = [float(v) if not isinstance(v, Exception) else np.nan for v in desc.values()]
                except Exception:
                    desc_vals = [np.nan] * len(calc.descriptors)
            else:
                desc_vals = [np.nan] * len(calc.descriptors)
            desc_list.append(desc_vals)
        desc_arr = np.array(desc_list, dtype=np.float64)
        desc_arr = np.where(np.isinf(desc_arr), np.nan, desc_arr)
        desc_arr = np.nan_to_num(desc_arr, nan=0.0)
        if MORDRED_VALID_COLS is None:
            col_std = desc_arr.std(axis=0)
            MORDRED_VALID_COLS = np.where(col_std > 0)[0]
            print(f"  Mordred valid cols: {len(MORDRED_VALID_COLS)}")
        desc_arr = desc_arr[:, MORDRED_VALID_COLS]
        X = np.hstack([fps, desc_arr])
    else:
        from rdkit.Chem import Descriptors
        physchem = []
        for mol in mols:
            if mol is not None:
                physchem.append([Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                                  Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
                                  Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol)])
            else:
                physchem.append([0.0] * 6)
        X = np.hstack([fps, np.array(physchem)])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

print("Computing features...")
X_all = compute_features(train_df["Drug"].tolist())
y_all = train_df["Y"].values
X_test = compute_features(test_df["Drug"].tolist())

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

neg = (y_all == 0).sum()
pos = (y_all == 1).sum()
scale_pos_weight = neg / pos

# Level-1 base learners: generate OOF predictions
base_learners = [
    ("xgb_d6", XGBClassifier(n_estimators=700, max_depth=6, learning_rate=0.04,
                               subsample=0.8, colsample_bytree=0.7,
                               scale_pos_weight=scale_pos_weight,
                               use_label_encoder=False, eval_metric="auc",
                               n_jobs=-1, random_state=42)),
    ("xgb_d8", XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                               subsample=0.75, colsample_bytree=0.65,
                               scale_pos_weight=scale_pos_weight,
                               use_label_encoder=False, eval_metric="auc",
                               n_jobs=-1, random_state=123)),
    ("rf", RandomForestClassifier(n_estimators=500, max_depth=None,
                                   class_weight='balanced', random_state=42, n_jobs=-1)),
    ("et", ExtraTreesClassifier(n_estimators=500, max_depth=None,
                                 class_weight='balanced', random_state=42, n_jobs=-1)),
]

oof_matrix = np.zeros((len(train_df), len(base_learners)))
test_preds_matrix = np.zeros((len(test_df), len(base_learners)))

for bl_idx, (name, clf) in enumerate(base_learners):
    print(f"\n--- Base learner: {name} ---")
    oof_preds = np.zeros(len(train_df))
    test_fold_preds = []
    fold_scores = []

    for k in range(5):
        tr_mask = train_df["cv_fold"] != k
        val_mask = train_df["cv_fold"] == k

        if hasattr(clf, 'use_label_encoder'):
            clf_k = type(clf)(**{**clf.get_params()})
            clf_k.fit(X_all[tr_mask], y_all[tr_mask],
                      eval_set=[(X_all[val_mask], y_all[val_mask])], verbose=False)
        else:
            clf_k = type(clf)(**clf.get_params())
            clf_k.fit(X_all[tr_mask], y_all[tr_mask])

        val_preds = clf_k.predict_proba(X_all[val_mask])[:, 1]
        oof_preds[val_mask] = val_preds
        test_fold_preds.append(clf_k.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_all[val_mask], val_preds)
        fold_scores.append(auc)
        print(f"  Fold {k}: {auc:.4f}")

    mean_auc = np.mean(fold_scores)
    print(f"  {name} CV: {mean_auc:.4f}")
    oof_matrix[:, bl_idx] = oof_preds
    test_preds_matrix[:, bl_idx] = np.mean(np.column_stack(test_fold_preds), axis=1)

# Level-2: Meta-learner on OOF
print("\n--- Level 2: Meta-learner ---")
meta_fold_scores = []
meta_oof = np.zeros(len(train_df))
meta_test_preds_list = []

for k in range(5):
    tr_mask = train_df["cv_fold"] != k
    val_mask = train_df["cv_fold"] == k

    X_meta_tr = oof_matrix[tr_mask]
    y_meta_tr = y_all[tr_mask]
    X_meta_val = oof_matrix[val_mask]
    y_meta_val = y_all[val_mask]

    scaler = StandardScaler()
    X_meta_tr_s = scaler.fit_transform(X_meta_tr)
    X_meta_val_s = scaler.transform(X_meta_val)

    # Logistic regression meta-learner
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta.fit(X_meta_tr_s, y_meta_tr)
    meta_val_preds = meta.predict_proba(X_meta_val_s)[:, 1]
    meta_oof[val_mask] = meta_val_preds

    X_test_s = scaler.transform(test_preds_matrix)
    meta_test_preds_list.append(meta.predict_proba(X_test_s)[:, 1])

    auc = roc_auc_score(y_meta_val, meta_val_preds)
    meta_fold_scores.append(auc)
    print(f"  Fold {k}: {auc:.4f}")

meta_cv = np.mean(meta_fold_scores)
print(f"\nMeta-learner CV (TRUE, measured): {meta_cv:.4f} ± {np.std(meta_fold_scores):.4f}")
print(f"Note: this is the ACTUAL stacking CV, not an estimate!")

# Final test predictions
meta_test_preds = np.mean(np.column_stack(meta_test_preds_list), axis=1)

# Simple average baseline for comparison
simple_avg = oof_matrix.mean(axis=1)
simple_avg_cv = roc_auc_score(y_all, simple_avg)
print(f"Simple average OOF CV: {simple_avg_cv:.4f}")

# Use the better of the two
if meta_cv >= simple_avg_cv:
    final_test_preds = meta_test_preds
    best_cv = meta_cv
    best_method = "stacking"
else:
    final_test_preds = test_preds_matrix.mean(axis=1)
    best_cv = simple_avg_cv
    best_method = "simple_average"

print(f"Using: {best_method} @ {best_cv:.4f}")

# Save results
sub_df = pd.DataFrame({"id": test_df["id"], "Y": final_test_preds})
sub_path = f"{AGENT_WORKSPACE}/submission_{EXP_ID}.csv"
sub_df.to_csv(sub_path, index=False)
shutil.copy(sub_path, f"{AGENT_WORKSPACE}/submission.csv")
shutil.copy(__file__, f"{AGENT_WORKSPACE}/train_{EXP_ID}.py")

result = {
    "val_score": float(best_cv),
    "direction": "maximize",
    "exp_id": EXP_ID,
        "submission_path": sub_path,
    "train_path": f"{AGENT_WORKSPACE}/train_{EXP_ID}.py",
    "meta_cv": float(meta_cv),
    "simple_avg_cv": float(simple_avg_cv),
    "meta_fold_scores": [float(s) for s in meta_fold_scores],
    "best_method": best_method,
    "note": f"Stacking meta-learner (LR). Stacked CV={meta_cv:.4f}, SimpleAvg={simple_avg_cv:.4f}",
    "is_true_cv": True,
}
result_path = f"{FOCUS_ROOT}/agents/{AGENT_NAME}/workspace/result_latest.json"
with open(result_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n=== DONE: {EXP_ID} | Best CV={best_cv:.4f} ({best_method}) ===")
