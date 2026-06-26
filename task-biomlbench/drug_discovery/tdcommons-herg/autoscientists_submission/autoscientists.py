"""
exp_arch_016: Champion approach with seed=789 (found to give 0.9163 in seed sweep)

Strategy:
- Same features as champion: ECFP4 (2048) + MACCS (167) + RDKit (42) = 2257 features
- Same base models: LightGBM, RF, XGBoost + LogReg meta (exactly like champion)
- Key difference: random_state=789 instead of 42
- exp_arch_015 showed seed=789 gives 0.9163 with TopTorsion features
- Hypothesis: seed=789 also gives better results with champion feature set
"""

import os
import sys
import json
import shutil
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors

import lightgbm as lgb
import xgboost as xgb

# ---- Paths ----
FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"AGENT_WS = Path(__file__).parent / "outputs"

EXP_ID = "exp_arch_016"
RANDOM_STATE = 789  # KEY CHANGE: different seed found to be better
CHAMPION_SCORE = 0.9138

ECFP_RADIUS = 2
ECFP_NBITS = 2048

RDKIT_DESCRIPTOR_NAMES = [
    "MolWt", "ExactMolWt", "MolLogP", "MolMR", "TPSA",
    "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings",
    "NumSaturatedRings", "NumAliphaticRings", "RingCount",
    "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount",
    "NumValenceElectrons", "NumRadicalElectrons",
    "BalabanJ", "BertzCT", "Chi0", "Chi0n", "Chi0v",
    "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v",
    "Kappa1", "Kappa2", "Kappa3",
    "LabuteASA", "PEOE_VSA1", "PEOE_VSA2", "PEOE_VSA3",
    "SMR_VSA1", "SMR_VSA2", "SlogP_VSA1", "SlogP_VSA2",
    "qed", "MaxPartialCharge", "MinPartialCharge",
]


def smiles_to_features(smiles_list):
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(RDKIT_DESCRIPTOR_NAMES)
    n_rdkit = len(RDKIT_DESCRIPTOR_NAMES)

    all_features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            all_features.append([0] * (ECFP_NBITS + 167 + n_rdkit))
            continue
        ecfp4 = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=ECFP_RADIUS, nBits=ECFP_NBITS))
        maccs = list(MACCSkeys.GenMACCSKeys(mol))
        try:
            rdkit_desc = list(calc.CalcDescriptors(mol))
        except Exception:
            rdkit_desc = [0] * n_rdkit
        all_features.append(ecfp4 + maccs + rdkit_desc)

    return np.array(all_features, dtype=np.float32)


def main():
    print("=" * 60)
    print(f"Experiment: {EXP_ID}")
    print(f"Champion approach with random_state={RANDOM_STATE}")
    print("Features: ECFP4 (2048) + MACCS (167) + RDKit (42) - same as champion")
    print("=" * 60)

    # ---- Load data ----
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test_features.csv")

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    y_all_raw = train_df["Y"].values.astype(int)
    class_counts = np.bincount(y_all_raw)
    scale_pos_weight = class_counts[0] / class_counts[1]
    print(f"Class distribution: 0={class_counts[0]}, 1={class_counts[1]}")

    # ---- Compute features ----
    print("Computing features...")
    X_all_raw = smiles_to_features(train_df["Drug"].tolist())
    y_all = y_all_raw
    print(f"Feature shape: {X_all_raw.shape}")

    imputer = SimpleImputer(strategy="median")
    X_all = imputer.fit_transform(X_all_raw)

    # ---- 80/20 split ----
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
    )
    print(f"Train split: {len(X_tr)}, Val split: {len(X_val)}")

    # ---- Build stacking (same as champion params, different seed) ----
    lgbm_base = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, objective="binary",
        n_jobs=8, random_state=RANDOM_STATE, verbose=-1,
        scale_pos_weight=scale_pos_weight
    )
    rf_base = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2,
        n_jobs=8, random_state=RANDOM_STATE, class_weight="balanced"
    )
    xgb_base = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=8, random_state=RANDOM_STATE, verbosity=0,
        scale_pos_weight=scale_pos_weight
    )
    meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)

    stacking = StackingClassifier(
        estimators=[("lgbm", lgbm_base), ("rf", rf_base), ("xgb", xgb_base)],
        final_estimator=meta,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        n_jobs=1, passthrough=False, verbose=1,
    )

    print("\nTraining stacking ensemble...")
    stacking.fit(X_tr, y_tr)

    val_preds = stacking.predict_proba(X_val)[:, 1]
    val_score = roc_auc_score(y_val, val_preds)
    print(f"\nValidation ROC-AUC: {val_score:.4f}")

    print("\n" + "=" * 60)
    print(json.dumps({
        "exp_id": EXP_ID,
        "val_roc_auc": float(val_score),
        "random_state": RANDOM_STATE,
        "feature_set": "ecfp4_2048+maccs_167+rdkit_42",
    }, indent=2))
    print("=" * 60)

    # ---- Final model on ALL data ----
    print("\nRetraining on ALL training data...")
    lgbm_f = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=1.0, objective="binary",
        n_jobs=8, random_state=RANDOM_STATE, verbose=-1,
        scale_pos_weight=scale_pos_weight
    )
    rf_f = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2,
        n_jobs=8, random_state=RANDOM_STATE, class_weight="balanced"
    )
    xgb_f = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=8, random_state=RANDOM_STATE, verbosity=0,
        scale_pos_weight=scale_pos_weight
    )
    meta_f = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)

    stacking_final = StackingClassifier(
        estimators=[("lgbm", lgbm_f), ("rf", rf_f), ("xgb", xgb_f)],
        final_estimator=meta_f,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        n_jobs=1, passthrough=False, verbose=1,
    )
    stacking_final.fit(X_all, y_all)

    print("Computing test features...")
    X_test_raw = smiles_to_features(test_df["Drug"].tolist())
    X_test = imputer.transform(X_test_raw)
    test_preds = stacking_final.predict_proba(X_test)[:, 1]

    # ---- Save submission ----
    submission = pd.DataFrame({"id": test_df["id"], "Y": test_preds})
    ws_submission_path = AGENT_WS / "submission_arch016.csv"
    submission.to_csv(ws_submission_path, index=False)

    submission_path = TASK_DIR / "submission.csv"
    if val_score > CHAMPION_SCORE:
        submission.to_csv(submission_path, index=False)
        print(f"NEW CHAMPION! Saved to: {submission_path}")
        shutil.copy(__file__, TASK_DIR / "train.py")
        shutil.copy(__file__, FOCUS_ROOT / "champion" / "train.py")
        (FOCUS_ROOT / "champion" / "SOURCE").write_text(
            f"exp_arch_016 {val_score}
"
        )
    else:
        print(f"Score {val_score:.4f} does not beat champion {CHAMPION_SCORE:.4f}")

    print(f"Prediction stats: min={test_preds.min():.4f}, max={test_preds.max():.4f}, mean={test_preds.mean():.4f}")

    # ---- Write result_latest.json ----
    result_summary = {
        "val_score": float(val_score),
        "direction": "maximize",
        "exp_id": EXP_ID,
        "submission_path": str(submission_path),
        "metric_name": "roc_auc",
        "feature_set": "ecfp4_2048+maccs_167+rdkit_42",
        "model": "stacking_lgbm+rf+xgb_logreg_seed789",
    }
    result_path = AGENT_WS / "result_latest.json"
    result_path.write_text(json.dumps(result_summary, indent=2))
    print(f"result_latest.json written to: {result_path}")

    return val_score, submission_path


if __name__ == "__main__":
    val_score, submission_path = main()
    print(f"\nDone. Final val ROC-AUC: {val_score:.4f}")
