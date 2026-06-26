"""
Experiment: exp_alpha_010
Approach: beta_007 cross-SHAP features (160-dim) + SVM-RBF (calibrated) as 5th base model
          + LGB+XGB+CAT+ET stacking with LR meta-learner
Task: CYP2D6 substrate binary classification (PR-AUC, higher is better)
Team: alpha

Cycle 3 context: Champion 0.7367 (beta_007). Our best was 0.7350 (alpha_008).

Strategy:
- Use the same 160-dim cross-SHAP feature set (Mordred-50+ECFP4-30+ECFP6-30+MACCS-20+AP-15+TT-15)
- Add SVM with RBF kernel as 5th base model (calibrated for probabilities)
- SVM captures different decision boundaries than tree-based models
- class_weight='balanced' handles imbalance
- CalibratedClassifierCV (cv=5, method='isotonic') for proper probability estimation
- LR meta-learner on 5-model OOF (proven better than GBM meta in exp_alpha_009)
- Scale features for SVM (StandardScaler)

Hypothesis: SVM-RBF brings orthogonal signal to the tree-based ensemble,
improving stacking performance beyond the champion's 4-model stack.
"""

import json
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem, rdMolDescriptors
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"AGENT_WORKSPACE = Path(__file__).parent / "outputs"
EXP_ID = "exp_alpha_010"

print(f"=== {EXP_ID}: beta_007 cross-SHAP features (160-dim) + SVM-RBF (5th base) + LR meta ===")
print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

# --- Load data ---
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test_features.csv")

print(f"Train: {len(train_df)} rows, {train_df['Y'].sum()} positives ({train_df['Y'].mean()*100:.1f}%)")
print(f"Test: {len(test_df)} rows")

y_all = train_df["Y"].values
n_folds = 5
folds = range(n_folds)
fold_assignments = train_df["cv_fold"].values


# ============================================================
# Helper: compute molecules
# ============================================================
def smiles_to_mols(smiles_series):
    return [Chem.MolFromSmiles(s) for s in smiles_series]


# ============================================================
# Helper: SHAP-based feature selection
# ============================================================
def shap_select_features(X_np, y, col_names, top_k, fold_assignments, seed=42):
    """Run 5-fold OOF LGB, accumulate SHAP importance, return top-k column indices."""
    shap_importance = np.zeros(X_np.shape[1])
    for k in range(5):
        tr_mask = fold_assignments != k
        val_mask = fold_assignments == k
        X_tr, X_val = X_np[tr_mask], X_np[val_mask]
        y_tr = y[tr_mask]
        pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        clf = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=seed,
            verbose=-1,
            n_jobs=8,
        )
        clf.fit(X_tr, y_tr)
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_val)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_importance += np.abs(shap_vals).mean(axis=0)

    top_indices = np.argsort(shap_importance)[::-1][:top_k]
    print(f"    SHAP selected top-{top_k} from {X_np.shape[1]} features")
    return top_indices


# ============================================================
# Feature extraction (same as exp_alpha_009)
# ============================================================
print("\n--- Feature extraction ---")
train_mols = smiles_to_mols(train_df["Drug"])
test_mols = smiles_to_mols(test_df["Drug"])

# 1. Mordred descriptors
print("Computing Mordred descriptors...")
calc = Calculator(descriptors, ignore_3D=True)
mordred_tr_raw = pd.DataFrame(calc.pandas(train_mols)).select_dtypes(include=[float, int])
mordred_te_raw = pd.DataFrame(calc.pandas(test_mols)).select_dtypes(include=[float, int])

common_cols = list(set(mordred_tr_raw.columns) & set(mordred_te_raw.columns))
mordred_tr_raw = mordred_tr_raw[common_cols]
mordred_te_raw = mordred_te_raw[common_cols]

mordred_tr_raw = mordred_tr_raw.replace([np.inf, -np.inf], np.nan)
mordred_te_raw = mordred_te_raw.replace([np.inf, -np.inf], np.nan)
valid_cols = mordred_tr_raw.columns[mordred_tr_raw.isnull().sum() < (len(mordred_tr_raw) * 0.2)]
mordred_tr_raw = mordred_tr_raw[valid_cols].fillna(mordred_tr_raw[valid_cols].median())
mordred_te_raw = mordred_te_raw[valid_cols].fillna(mordred_tr_raw[valid_cols].median())

print(f"  Mordred: {mordred_tr_raw.shape[1]} descriptors after cleaning")
mordred_col_names = list(mordred_tr_raw.columns)

print("  SHAP-selecting top-50 Mordred features...")
mordred_idx = shap_select_features(mordred_tr_raw.values, y_all, mordred_col_names, 50, fold_assignments)
X_mordred_tr = mordred_tr_raw.values[:, mordred_idx]
X_mordred_te = mordred_te_raw.values[:, mordred_idx]

# 2. ECFP4
print("Computing ECFP4 fingerprints...")
ecfp4_tr = np.array([
    AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) if m else np.zeros(2048)
    for m in train_mols
], dtype=float)
ecfp4_te = np.array([
    AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) if m else np.zeros(2048)
    for m in test_mols
], dtype=float)
ecfp4_col_names = [f"ECFP4_{i}" for i in range(2048)]

print("  SHAP-selecting top-30 ECFP4 bits...")
ecfp4_idx = shap_select_features(ecfp4_tr, y_all, ecfp4_col_names, 30, fold_assignments)
X_ecfp4_tr = ecfp4_tr[:, ecfp4_idx]
X_ecfp4_te = ecfp4_te[:, ecfp4_idx]

# 3. ECFP6
print("Computing ECFP6 fingerprints...")
ecfp6_tr = np.array([
    AllChem.GetMorganFingerprintAsBitVect(m, radius=3, nBits=2048) if m else np.zeros(2048)
    for m in train_mols
], dtype=float)
ecfp6_te = np.array([
    AllChem.GetMorganFingerprintAsBitVect(m, radius=3, nBits=2048) if m else np.zeros(2048)
    for m in test_mols
], dtype=float)
ecfp6_col_names = [f"ECFP6_{i}" for i in range(2048)]

print("  SHAP-selecting top-30 ECFP6 bits...")
ecfp6_idx = shap_select_features(ecfp6_tr, y_all, ecfp6_col_names, 30, fold_assignments)
X_ecfp6_tr = ecfp6_tr[:, ecfp6_idx]
X_ecfp6_te = ecfp6_te[:, ecfp6_idx]

# 4. MACCS keys
print("Computing MACCS keys...")
maccs_tr = np.array([
    list(MACCSkeys.GenMACCSKeys(m)) if m else [0] * 167
    for m in train_mols
], dtype=float)
maccs_te = np.array([
    list(MACCSkeys.GenMACCSKeys(m)) if m else [0] * 167
    for m in test_mols
], dtype=float)
maccs_col_names = [f"MACCS_{i}" for i in range(167)]

print("  SHAP-selecting top-20 MACCS bits...")
maccs_idx = shap_select_features(maccs_tr, y_all, maccs_col_names, 20, fold_assignments)
X_maccs_tr = maccs_tr[:, maccs_idx]
X_maccs_te = maccs_te[:, maccs_idx]

# 5. AtomPair
print("Computing AtomPair fingerprints...")
atompair_tr = np.array([
    list(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=2048)) if m else [0] * 2048
    for m in train_mols
], dtype=float)
atompair_te = np.array([
    list(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=2048)) if m else [0] * 2048
    for m in test_mols
], dtype=float)
atompair_col_names = [f"AP_{i}" for i in range(2048)]

print("  SHAP-selecting top-15 AtomPair bits...")
ap_idx = shap_select_features(atompair_tr, y_all, atompair_col_names, 15, fold_assignments)
X_ap_tr = atompair_tr[:, ap_idx]
X_ap_te = atompair_te[:, ap_idx]

# 6. TopoTorsion
print("Computing TopologicalTorsion fingerprints...")
topo_tr = np.array([
    list(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=2048)) if m else [0] * 2048
    for m in train_mols
], dtype=float)
topo_te = np.array([
    list(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=2048)) if m else [0] * 2048
    for m in test_mols
], dtype=float)
topo_col_names = [f"TT_{i}" for i in range(2048)]

print("  SHAP-selecting top-15 TopoTorsion bits...")
tt_idx = shap_select_features(topo_tr, y_all, topo_col_names, 15, fold_assignments)
X_tt_tr = topo_tr[:, tt_idx]
X_tt_te = topo_te[:, tt_idx]

# Fuse all feature groups
X_fused_tr = np.hstack([X_mordred_tr, X_ecfp4_tr, X_ecfp6_tr, X_maccs_tr, X_ap_tr, X_tt_tr])
X_fused_te = np.hstack([X_mordred_te, X_ecfp4_te, X_ecfp6_te, X_maccs_te, X_ap_te, X_tt_te])
print(f"\nFused feature matrix: {X_fused_tr.shape} (train), {X_fused_te.shape} (test)")

# Scale features for SVM
scaler = StandardScaler()
X_scaled_tr = scaler.fit_transform(X_fused_tr)
X_scaled_te = scaler.transform(X_fused_te)
print(f"Features scaled (StandardScaler) for SVM")


# ============================================================
# 5-fold scaffold CV: LGB + XGB + CAT + ET + SVM(calibrated)
# ============================================================
print("\n--- 5-fold scaffold CV: LGB + XGB + CAT + ET + SVM-RBF (calibrated) ---")

n = len(train_df)
oof_lgb = np.zeros(n)
oof_xgb = np.zeros(n)
oof_cat = np.zeros(n)
oof_et = np.zeros(n)
oof_svm = np.zeros(n)

test_preds_lgb = np.zeros(len(test_df))
test_preds_xgb = np.zeros(len(test_df))
test_preds_cat = np.zeros(len(test_df))
test_preds_et = np.zeros(len(test_df))
test_preds_svm = np.zeros(len(test_df))

fold_scores_lgb = []
fold_scores_xgb = []
fold_scores_cat = []
fold_scores_et = []
fold_scores_svm = []

clf_lgb_last = clf_xgb_last = clf_cat_last = None

for k in folds:
    tr_mask = fold_assignments != k
    val_mask = fold_assignments == k
    X_tr, X_val = X_fused_tr[tr_mask], X_fused_tr[val_mask]
    X_sc_tr, X_sc_val = X_scaled_tr[tr_mask], X_scaled_tr[val_mask]
    y_tr, y_val = y_all[tr_mask], y_all[val_mask]
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    print(f"\nFold {k}: train={tr_mask.sum()}, val={val_mask.sum()}, pos={y_val.sum()}")

    # LightGBM (default champion-style)
    clf_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbose=-1,
        n_jobs=8,
    )
    clf_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    p_lgb_val = clf_lgb.predict_proba(X_val)[:, 1]
    p_lgb_test = clf_lgb.predict_proba(X_fused_te)[:, 1]
    oof_lgb[val_mask] = p_lgb_val
    test_preds_lgb += p_lgb_test / n_folds
    score_lgb = average_precision_score(y_val, p_lgb_val)
    fold_scores_lgb.append(score_lgb)
    clf_lgb_last = clf_lgb
    print(f"  LGB PR-AUC: {score_lgb:.4f}")

    # XGBoost
    clf_xgb = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50,
        eval_metric="aucpr",
        n_jobs=8,
    )
    clf_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    p_xgb_val = clf_xgb.predict_proba(X_val)[:, 1]
    p_xgb_test = clf_xgb.predict_proba(X_fused_te)[:, 1]
    oof_xgb[val_mask] = p_xgb_val
    test_preds_xgb += p_xgb_test / n_folds
    score_xgb = average_precision_score(y_val, p_xgb_val)
    fold_scores_xgb.append(score_xgb)
    clf_xgb_last = clf_xgb
    print(f"  XGB PR-AUC: {score_xgb:.4f}")

    # CatBoost
    clf_cat = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
        eval_metric="PRAUC",
    )
    clf_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    p_cat_val = clf_cat.predict_proba(X_val)[:, 1]
    p_cat_test = clf_cat.predict_proba(X_fused_te)[:, 1]
    oof_cat[val_mask] = p_cat_val
    test_preds_cat += p_cat_test / n_folds
    score_cat = average_precision_score(y_val, p_cat_val)
    fold_scores_cat.append(score_cat)
    clf_cat_last = clf_cat
    print(f"  CAT PR-AUC: {score_cat:.4f}")

    # ExtraTrees
    clf_et = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=8,
    )
    clf_et.fit(X_tr, y_tr)
    p_et_val = clf_et.predict_proba(X_val)[:, 1]
    p_et_test = clf_et.predict_proba(X_fused_te)[:, 1]
    oof_et[val_mask] = p_et_val
    test_preds_et += p_et_test / n_folds
    score_et = average_precision_score(y_val, p_et_val)
    fold_scores_et.append(score_et)
    print(f"  ET  PR-AUC: {score_et:.4f}")

    # SVM with RBF kernel + calibration
    # Train on scaled features; use class_weight='balanced' for imbalance
    base_svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=False,  # use CalibratedClassifierCV instead
        random_state=42,
    )
    # CalibratedClassifierCV with internal CV for probability calibration
    cal_svm = CalibratedClassifierCV(base_svm, method="isotonic", cv=3)
    cal_svm.fit(X_sc_tr, y_tr)
    p_svm_val = cal_svm.predict_proba(X_sc_val)[:, 1]
    p_svm_test = cal_svm.predict_proba(X_scaled_te)[:, 1]
    oof_svm[val_mask] = p_svm_val
    test_preds_svm += p_svm_test / n_folds
    score_svm = average_precision_score(y_val, p_svm_val)
    fold_scores_svm.append(score_svm)
    print(f"  SVM PR-AUC: {score_svm:.4f}")


# ============================================================
# LR meta-learner on 5-model OOF (proven better than GBM meta)
# ============================================================
print("\n--- LR meta-learner (5-model OOF: LGB+XGB+CAT+ET+SVM) ---")
oof_stack = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et, oof_svm])
test_stack = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_et, test_preds_svm])

meta_fold_scores = []
oof_meta = np.zeros(len(train_df))
test_meta_preds = np.zeros(len(test_df))

# Also test 4-model without SVM for reference
oof_stack_4 = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_et])
test_stack_4 = np.column_stack([test_preds_lgb, test_preds_xgb, test_preds_cat, test_preds_et])
meta_fold_scores_4 = []

for k in folds:
    tr_mask = fold_assignments != k
    val_mask = fold_assignments == k
    y_tr_meta = y_all[tr_mask]
    y_val_meta = y_all[val_mask]

    # 5-model LR meta (with SVM)
    meta5 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta5.fit(oof_stack[tr_mask], y_tr_meta)
    p_meta5_val = meta5.predict_proba(oof_stack[val_mask])[:, 1]
    oof_meta[val_mask] = p_meta5_val
    score_meta5 = average_precision_score(y_val_meta, p_meta5_val)
    meta_fold_scores.append(score_meta5)
    test_meta_preds += meta5.predict_proba(test_stack)[:, 1] / n_folds
    print(f"  Fold {k} 5-model LR meta PR-AUC: {score_meta5:.4f}")

    # 4-model LR meta (without SVM, reference)
    meta4 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta4.fit(oof_stack_4[tr_mask], y_tr_meta)
    p_meta4_val = meta4.predict_proba(oof_stack_4[val_mask])[:, 1]
    score_meta4 = average_precision_score(y_val_meta, p_meta4_val)
    meta_fold_scores_4.append(score_meta4)

fold_scores_stack = meta_fold_scores

# Summary
print("\n=== CV Results ===")
print(f"LGB         : {np.mean(fold_scores_lgb):.4f} +/- {np.std(fold_scores_lgb):.4f} | {[f'{s:.4f}' for s in fold_scores_lgb]}")
print(f"XGB         : {np.mean(fold_scores_xgb):.4f} +/- {np.std(fold_scores_xgb):.4f} | {[f'{s:.4f}' for s in fold_scores_xgb]}")
print(f"CAT         : {np.mean(fold_scores_cat):.4f} +/- {np.std(fold_scores_cat):.4f} | {[f'{s:.4f}' for s in fold_scores_cat]}")
print(f"ET          : {np.mean(fold_scores_et):.4f} +/- {np.std(fold_scores_et):.4f} | {[f'{s:.4f}' for s in fold_scores_et]}")
print(f"SVM (RBF)   : {np.mean(fold_scores_svm):.4f} +/- {np.std(fold_scores_svm):.4f} | {[f'{s:.4f}' for s in fold_scores_svm]}")
print(f"STACK-5 (LR): {np.mean(meta_fold_scores):.4f} +/- {np.std(meta_fold_scores):.4f} | {[f'{s:.4f}' for s in meta_fold_scores]}")
print(f"STACK-4 (LR): {np.mean(meta_fold_scores_4):.4f} +/- {np.std(meta_fold_scores_4):.4f} | {[f'{s:.4f}' for s in meta_fold_scores_4]}")

# Use the better of 5 vs 4 model stack
stack5_mean = np.mean(meta_fold_scores)
stack4_mean = np.mean(meta_fold_scores_4)

if stack5_mean >= stack4_mean:
    mean_cv_score = stack5_mean
    std_cv_score = np.std(meta_fold_scores)
    fold_scores_stack = meta_fold_scores
    final_test_preds = test_meta_preds
    stack_label = "5-model+SVM"
else:
    mean_cv_score = stack4_mean
    std_cv_score = np.std(meta_fold_scores_4)
    fold_scores_stack = meta_fold_scores_4
    # Re-compute 4-model test preds
    final_meta4 = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    final_meta4.fit(oof_stack_4, y_all)
    final_test_preds = final_meta4.predict_proba(test_stack_4)[:, 1]
    stack_label = "4-model (SVM not beneficial)"

champion_score = 0.7367241148091737
delta = mean_cv_score - champion_score

print(f"\nMean CV PR-AUC ({stack_label}): {mean_cv_score:.4f} +/- {std_cv_score:.4f}")
print(f"Champion (beta_007): {champion_score:.4f}")
print(f"Delta: {delta:+.4f}")

outcome = "KEEP" if mean_cv_score > champion_score else "DISCARD"
print(f"Outcome: {outcome}")

# Hyperparameters JSON
hp = {
    "exp_id": EXP_ID,
    "approach": "beta007-SHAP160-LGB+XGB+CAT+ET+SVM-RBF-calibrated-LR-meta",
    "n_features_total": X_fused_tr.shape[1],
    "features": "Mordred-50+ECFP4-30+ECFP6-30+MACCS-20+AP-15+TT-15",
    "svm_params": {"C": 1.0, "gamma": "scale", "kernel": "rbf", "calibration": "isotonic(cv=3)"},
    "stack_used": stack_label,
    "val_metric": "PR-AUC",
    "mean_cv_prauc": mean_cv_score,
    "std_cv_prauc": std_cv_score,
    "per_fold_prauc": fold_scores_stack,
    "svm_per_fold": fold_scores_svm,
    "champion_baseline": champion_score,
    "delta_from_champion": round(delta, 6),
    "outcome": outcome,
}
print("\n" + "=" * 60)
print(json.dumps(hp, indent=2))
print("=" * 60)


# ============================================================
# Final model: retrain on full data
# ============================================================
print("\n--- Retraining on full data for submission ---")
pos_weight_full = (y_all == 0).sum() / max((y_all == 1).sum(), 1)

final_lgb = lgb.LGBMClassifier(
    n_estimators=clf_lgb_last.best_iteration_ + 50 if hasattr(clf_lgb_last, "best_iteration_") and clf_lgb_last.best_iteration_ else 300,
    learning_rate=0.05, num_leaves=63, min_child_samples=5,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_weight_full, random_state=42, verbose=-1, n_jobs=8,
)
final_lgb.fit(X_fused_tr, y_all)

final_xgb = xgb.XGBClassifier(
    n_estimators=clf_xgb_last.best_iteration + 50 if hasattr(clf_xgb_last, "best_iteration") and clf_xgb_last.best_iteration else 300,
    learning_rate=0.05, max_depth=6, min_child_weight=3,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_weight_full, random_state=42, verbosity=0, n_jobs=8,
)
final_xgb.fit(X_fused_tr, y_all)

final_cat = CatBoostClassifier(
    iterations=clf_cat_last.best_iteration_ + 50 if hasattr(clf_cat_last, "best_iteration_") and clf_cat_last.best_iteration_ else 300,
    learning_rate=0.05, depth=6, l2_leaf_reg=3,
    auto_class_weights="Balanced", random_seed=42, verbose=0,
)
final_cat.fit(X_fused_tr, y_all)

final_et = ExtraTreesClassifier(
    n_estimators=500, max_features="sqrt", min_samples_leaf=2,
    class_weight="balanced", random_state=42, n_jobs=8,
)
final_et.fit(X_fused_tr, y_all)

p_test_lgb = final_lgb.predict_proba(X_fused_te)[:, 1]
p_test_xgb = final_xgb.predict_proba(X_fused_te)[:, 1]
p_test_cat = final_cat.predict_proba(X_fused_te)[:, 1]
p_test_et = final_et.predict_proba(X_fused_te)[:, 1]

if stack_label.startswith("5"):
    # Train SVM on full data too
    final_svm_base = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                         probability=False, random_state=42)
    final_svm = CalibratedClassifierCV(final_svm_base, method="isotonic", cv=3)
    final_svm.fit(X_scaled_tr, y_all)
    p_test_svm = final_svm.predict_proba(X_scaled_te)[:, 1]
    test_full_stack = np.column_stack([p_test_lgb, p_test_xgb, p_test_cat, p_test_et, p_test_svm])

    final_meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    final_meta.fit(oof_stack, y_all)
    final_preds = final_meta.predict_proba(test_full_stack)[:, 1]
else:
    test_full_stack = np.column_stack([p_test_lgb, p_test_xgb, p_test_cat, p_test_et])
    final_meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    final_meta.fit(oof_stack_4, y_all)
    final_preds = final_meta.predict_proba(test_full_stack)[:, 1]

# Save submission
AGENT_WORKSPACE.mkdir(parents=True, exist_ok=True)
submission = pd.DataFrame({"id": test_df["id"], "Y": final_preds})
sub_path = AGENT_WORKSPACE / "submission.csv"
submission.to_csv(sub_path, index=False)
print(f"Saved submission.csv to {sub_path}")
print(f"Submission preview:\n{submission.head()}")

# Stamped copies
stamped_sub = AGENT_WORKSPACE / f"submission_{EXP_ID}.csv"
shutil.copy(sub_path, stamped_sub)
print(f"Stamped submission: {stamped_sub}")

# Stamp train.py (copy this file as train.py)
script_path = Path(__file__).resolve()
train_dest = AGENT_WORKSPACE / "train.py"
stamped_train = AGENT_WORKSPACE / f"train_{EXP_ID}.py"

# Write fresh copies to avoid SameFileError
content = script_path.read_bytes()
if train_dest.resolve() != script_path:
    train_dest.write_bytes(content)
if stamped_train.resolve() != script_path:
    stamped_train.write_bytes(content)
print(f"Saved train.py and {stamped_train.name}")

# Write result_latest.json
result_summary = {
    "val_score": mean_cv_score,
    "direction": "maximize",
    "exp_id": EXP_ID,
        "submission_path": str(stamped_sub),
    "train_path": str(stamped_train),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "delta_from_champion": round(delta, 6),
    "outcome": outcome,
    "stack": stack_label,
    "svm_per_fold": fold_scores_svm,
}
result_json_path = FOCUS_ROOT / "agents" / AGENT_NAME / "workspace" / "result_latest.json"
result_json_path.write_text(json.dumps(result_summary, indent=2))
print(f"\n[result_latest.json] written — val_score={mean_cv_score:.4f}")

print(f"\n=== DONE: {EXP_ID} ===")
print(f"Final PR-AUC: {mean_cv_score:.4f} | Champion: {champion_score:.4f} | Delta: {delta:+.4f} | Outcome: {outcome}")
