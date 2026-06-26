#!/usr/bin/env python
"""
Experiment: exp_gamma_gpu6_c3
Approach: 7-model Optuna stacking with CatBoost + 150 trials/fold + TPESampler(multivariate=True)
Full-data retrain of champion (exp_beta_gpu2_c3, PR-AUC=0.7941)

Champion 7-model stack:
  1. XGBoost (Morgan+MACCS+RDKit descriptors)
  2. LightGBM (same descriptor features)
  3. ExtraTrees (same descriptor features)
  4. SVM (Tanimoto kernel, Morgan FPs)
  5. GP (Tanimoto kernel, Morgan FPs)
  6. CatBoost (ordered boosting, auto class_weights='Balanced')
  7. Chemprop MPNN (GPU-native graph neural network)

Changes vs champion:
  - 150 Optuna trials per fold (was 120) for more thorough weight search
  - TPESampler(multivariate=True) for correlation-aware weight search
  - All base models trained on FULL training data (no fold holdout) for final predictions

Team: team_gamma
Cycle: 3 (CPU experiment - full-data retrain + 150 Optuna trials)
"""

import os
import json
import shutil
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Kernel
from sklearn.ensemble import ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import torch
import torch.nn as nn
import lightning.pytorch as pl
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors, rdMolDescriptors
from rdkit.DataStructs import ConvertToNumpyArray
from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.models import MPNN
from chemprop.nn import (
    BinaryClassificationFFN, MeanAggregation, BondMessagePassing, BinaryAUPRC
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.set_float32_matmul_precision('medium')

try:
    import rdkit.rdBase as rdBase
    rdBase.DisableLog("rdApp.warning")
except Exception:
    pass

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).parent
FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"AGENT_WORKSPACE = SCRIPT_DIR
EXP_ID = "exp_gamma_gpu6_c3"

DESCRIPTOR_NAMES = [name for name, fn in Descriptors.descList]
print(f"[{EXP_ID}] RDKit descriptor count: {len(DESCRIPTOR_NAMES)}")

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"
print(f"[{EXP_ID}] Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")


# ============================================================
# Feature extraction functions
# ============================================================

def smiles_to_descriptor_features(smiles_list):
    """Convert SMILES to Morgan + MACCS + RDKit descriptors for tree-based models."""
    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(2048 + 167 + len(DESCRIPTOR_NAMES)))
            continue
        morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)
        morgan_arr = np.array(morgan, dtype=np.float32)
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.array(maccs, dtype=np.float32)
        desc_vals = []
        for name in DESCRIPTOR_NAMES:
            try:
                v = getattr(Descriptors, name)(mol)
                if v is None or np.isnan(v) or np.isinf(v):
                    v = 0.0
            except Exception:
                v = 0.0
            desc_vals.append(float(v))
        features.append(np.concatenate([morgan_arr, maccs_arr, np.array(desc_vals, dtype=np.float32)]))
    return np.array(features, dtype=np.float32)


def smiles_to_morgan_fp(smiles_list, radius=2, n_bits=2048):
    """Convert SMILES to binary Morgan fingerprints for Tanimoto kernel methods."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float64))
        else:
            fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=np.float64)
            ConvertToNumpyArray(fp, arr)
            fps.append(arr)
    return np.array(fps)


def smiles_to_rdkit_aux_for_chemprop(smiles_list):
    """
    Compute a fixed set of physicochemical descriptors as auxiliary features for Chemprop x_d.
    These global molecular-level features complement graph topology learned by message passing.
    Returns (features_array, list_of_descriptor_names).
    """
    desc_names_subset = [
        'MolWt', 'ExactMolWt', 'HeavyAtomMolWt', 'NumHAcceptors', 'NumHDonors',
        'NumHeteroatoms', 'NumRotatableBonds', 'NumAromaticRings', 'NumSaturatedRings',
        'NumAliphaticRings', 'NumAromaticHeterocycles', 'NumSaturatedHeterocycles',
        'NumAliphaticHeterocycles', 'NumAromaticCarbocycles', 'NumSaturatedCarbocycles',
        'NumAliphaticCarbocycles', 'RingCount', 'FractionCSP3', 'MolLogP', 'TPSA',
        'LabuteASA', 'BalabanJ', 'BertzCT', 'Ipc', 'HallKierAlpha', 'Kappa1',
        'Kappa2', 'Kappa3', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
        'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
        'VSA_EState1', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5',
        'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'VSA_EState10',
        'EState_VSA1', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
        'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'EState_VSA10',
        'SlogP_VSA1', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5',
        'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'SlogP_VSA10',
        'SlogP_VSA11', 'SlogP_VSA12', 'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3',
        'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9',
        'SMR_VSA10', 'PEOE_VSA1', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4',
        'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
        'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
        'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex',
        'qed', 'SPS',
    ]
    # Only keep descriptors that exist in this RDKit version
    valid_descs = [name for name in desc_names_subset if hasattr(Descriptors, name)]

    features = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            features.append(np.zeros(len(valid_descs), dtype=np.float32))
            continue
        vals = []
        for name in valid_descs:
            try:
                v = getattr(Descriptors, name)(mol)
                if v is None or np.isnan(v) or np.isinf(v):
                    v = 0.0
            except Exception:
                v = 0.0
            vals.append(float(v))
        features.append(np.array(vals, dtype=np.float32))

    X = np.array(features, dtype=np.float32)
    return X, valid_descs


def clean_features(X):
    """Remove NaN/inf and clip extreme values."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    return X


# ============================================================
# Tanimoto Kernel (for SVM and GP)
# ============================================================

class TanimotoKernel(Kernel):
    """Tanimoto similarity kernel: k(x,x') = |x AND x'| / |x OR x'|"""

    def __init__(self):
        pass

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X
        X_sum = X.sum(axis=1, keepdims=True)
        Y_sum = Y.sum(axis=1, keepdims=True).T
        intersection = X @ Y.T
        union = X_sum + Y_sum - intersection
        union = np.maximum(union, 1e-10)
        K = intersection / union
        if eval_gradient:
            return K, np.zeros((X.shape[0], Y.shape[0], X.shape[1]))
        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return False

    def clone_with_theta(self, theta):
        return TanimotoKernel()


# ============================================================
# Classical Base Model Training (6 classical + CatBoost)
# ============================================================

def train_classical_base_models(X_tr_desc, y_tr, X_tr_fp, pos_weight):
    """Train XGB, LGB, ExtraTrees, SVM, GP, CatBoost base models (6 classical models)."""
    class_weights = {0: 1.0, 1: pos_weight}

    # XGBoost (champion HP)
    xgb_model = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.035, n_estimators=434,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.42, reg_lambda=2.42,
        scale_pos_weight=pos_weight, eval_metric="aucpr", verbosity=0,
        random_state=42, n_jobs=4
    )
    xgb_model.fit(X_tr_desc, y_tr)

    # LightGBM (champion HP)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=5, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
        scale_pos_weight=pos_weight, random_state=42, n_jobs=4, verbose=-1
    )
    lgb_model.fit(X_tr_desc, y_tr)

    # ExtraTrees (champion HP)
    et_model = ExtraTreesClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        class_weight=class_weights, random_state=42, n_jobs=4
    )
    et_model.fit(X_tr_desc, y_tr)

    # SVM (Tanimoto kernel, champion HP)
    svm_model = SVC(kernel=TanimotoKernel(), C=0.5, probability=True, max_iter=1000)
    svm_model.fit(X_tr_fp, y_tr)

    # GP (Tanimoto kernel, champion HP)
    gp_model = GaussianProcessClassifier(
        kernel=TanimotoKernel(), n_restarts_optimizer=0, random_state=42
    )
    gp_model.fit(X_tr_fp, y_tr)

    # CatBoost (7th model - ordered boosting, auto class weights)
    # CatBoost differs from XGB/LGB: uses ordered boosting to avoid target leakage,
    # symmetric trees for stability on small datasets, native class balancing.
    cat_model = cb.CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        auto_class_weights='Balanced',  # handles 9:1 class imbalance natively
        eval_metric='PRAUC',
        random_seed=42,
        verbose=0,
        thread_count=4,
    )
    cat_model.fit(X_tr_desc, y_tr)

    return xgb_model, lgb_model, et_model, svm_model, gp_model, cat_model


def get_classical_preds(models, X_desc, X_fp):
    """Get probability predictions from all 6 classical base models (incl CatBoost)."""
    xgb_m, lgb_m, et_m, svm_m, gp_m, cat_m = models
    return np.column_stack([
        xgb_m.predict_proba(X_desc)[:, 1],
        lgb_m.predict_proba(X_desc)[:, 1],
        et_m.predict_proba(X_desc)[:, 1],
        svm_m.predict_proba(X_fp)[:, 1],
        gp_m.predict_proba(X_fp)[:, 1],
        cat_m.predict_proba(X_desc)[:, 1],
    ])


# ============================================================
# Chemprop MPNN Training (7th base model - GNN)
# ============================================================

def make_molecule_dataset(smiles_list, y_list, x_d_array=None):
    """Build a MoleculeDataset from SMILES, optional labels, and optional auxiliary features."""
    fallback_mol = Chem.MolFromSmiles("C")  # fallback for invalid SMILES
    mols = [Chem.MolFromSmiles(s) or fallback_mol for s in smiles_list]

    if y_list is not None:
        dps = [
            MoleculeDatapoint(
                mol=m,
                y=[float(yv)],
                x_d=x_d_array[i] if x_d_array is not None else None
            )
            for i, (m, yv) in enumerate(zip(mols, y_list))
        ]
    else:
        dps = [
            MoleculeDatapoint(
                mol=m,
                x_d=x_d_array[i] if x_d_array is not None else None
            )
            for i, m in enumerate(mols)
        ]
    return MoleculeDataset(dps)


def train_chemprop_model(smiles_tr, y_tr, x_d_tr=None, epochs=60, d_h=300, seed=42):
    """
    Train a Chemprop MPNN on molecular graphs.

    Uses class_balance=True in DataLoader to handle class imbalance (10% positive).
    Auxiliary RDKit descriptors (x_d) provided as global molecular-level features.
    GPU acceleration via Lightning Trainer.
    """
    pl.seed_everything(seed, workers=True)

    d_xd = x_d_tr.shape[1] if x_d_tr is not None else 0

    ds_train = make_molecule_dataset(smiles_tr, y_tr, x_d_tr)

    # class_balance=True: use ClassBalanceSampler to oversample positives
    dl_train = build_dataloader(
        ds_train,
        batch_size=32,
        class_balance=True,
        shuffle=True,
        seed=seed,
        num_workers=2
    )

    # Build MPNN: BondMessagePassing -> MeanAggregation -> BinaryClassificationFFN
    mp = BondMessagePassing(d_h=d_h, depth=3, dropout=0.1)
    agg = MeanAggregation()
    # input_dim = d_h (graph) + d_xd (auxiliary molecular features)
    pred = BinaryClassificationFFN(
        n_tasks=1,
        input_dim=d_h + d_xd,
        hidden_dim=300,
        n_layers=2,
        dropout=0.1
    )

    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=pred,
        metrics=[BinaryAUPRC(task="binary")],
        init_lr=1e-4,
        max_lr=3e-4,
        final_lr=1e-5,
        warmup_epochs=2,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if DEVICE == "gpu" else "cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, dl_train)

    return model, trainer


def predict_chemprop_model(model, trainer, smiles_list, x_d=None):
    """Get probability predictions from a trained Chemprop model."""
    ds = make_molecule_dataset(smiles_list, None, x_d)
    dl = build_dataloader(ds, batch_size=64, shuffle=False, num_workers=2)
    preds = trainer.predict(model, dl)
    probs = torch.cat(preds, dim=0).cpu().numpy()[:, 0]
    return probs


# ============================================================
# Optuna Weight Optimization (120 trials, TPESampler multivariate)
# ============================================================

def optimize_weights_optuna(oof_preds, y_labels, n_trials=120):
    """
    Use Optuna to find optimal non-negative weights for all base models
    that maximize PR-AUC on OOF predictions.

    Changes vs champion:
    - 120 trials (was 60) for more thorough weight search
    - TPESampler(multivariate=True) exploits correlations between weight dimensions
      (useful when weights are correlated, e.g., XGB and LGB co-vary)
    """
    n_models = oof_preds.shape[1]

    def objective(trial):
        raw = [trial.suggest_float(f'w{i}', 0.0, 1.0) for i in range(n_models)]
        total = sum(raw) + 1e-10
        weights = np.array(raw) / total
        weighted_preds = (oof_preds * weights).sum(axis=1)
        return average_precision_score(y_labels, weighted_preds)

    # TPESampler with multivariate=True: models joint distribution of parameters
    # This is better than independent TPE when parameters are correlated (weights often are)
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_raw = [study.best_params[f'w{i}'] for i in range(n_models)]
    total = sum(best_raw) + 1e-10
    best_weights = np.array(best_raw) / total
    best_score = study.best_value
    return best_weights, best_score


# ============================================================
# Main
# ============================================================

def main():
    print(f"\n[{EXP_ID}] ============ EXPERIMENT START ============")
    print(f"[{EXP_ID}] Approach: 7-model Optuna stacking (full-data retrain, 150 trials)")
    print(f"[{EXP_ID}] Changes: 150 Optuna trials (vs 120 champion), TPESampler(multivariate=True)")
    print(f"[{EXP_ID}] Agent: {AGENT_NAME}")
    print(f"[{EXP_ID}] Start time: {datetime.now(timezone.utc).isoformat()}\n")

    # Load data
    print(f"[{EXP_ID}] Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test_features.csv")

    n_neg = (train_df["CLASS_EGFR"] == 0).sum()
    n_pos = train_df["CLASS_EGFR"].sum()
    scale_pos_weight = float(n_neg / n_pos)
    print(f"[{EXP_ID}] Data: {len(train_df)} train, {len(test_df)} test")
    print(f"[{EXP_ID}] Class balance: {n_pos} pos, {n_neg} neg (pos_weight={scale_pos_weight:.2f})")

    # Extract features for classical models
    print(f"\n[{EXP_ID}] Featurizing for classical models (Morgan+MACCS+RDKit descriptors)...")
    X_desc = clean_features(smiles_to_descriptor_features(train_df["smiles"].tolist()))
    X_fp = smiles_to_morgan_fp(train_df["smiles"].tolist())
    X_test_desc = clean_features(smiles_to_descriptor_features(test_df["smiles"].tolist()))
    X_test_fp = smiles_to_morgan_fp(test_df["smiles"].tolist())
    print(f"[{EXP_ID}] Descriptor shape: {X_desc.shape}, FP shape: {X_fp.shape}")

    # Extract auxiliary features for Chemprop (physicochemical descriptors as x_d)
    print(f"\n[{EXP_ID}] Computing auxiliary RDKit features for Chemprop x_d...")
    X_xd_train_raw, desc_names_used = smiles_to_rdkit_aux_for_chemprop(train_df["smiles"].tolist())
    X_xd_test_raw, _ = smiles_to_rdkit_aux_for_chemprop(test_df["smiles"].tolist())

    # Normalize using training set statistics (z-score)
    xd_mean = X_xd_train_raw.mean(axis=0)
    xd_std = X_xd_train_raw.std(axis=0) + 1e-8
    X_xd_train = np.nan_to_num(((X_xd_train_raw - xd_mean) / xd_std).astype(np.float32))
    X_xd_test = np.nan_to_num(((X_xd_test_raw - xd_mean) / xd_std).astype(np.float32))
    print(f"[{EXP_ID}] Chemprop x_d shape: {X_xd_train.shape} ({len(desc_names_used)} descriptors)")

    y_all = train_df["CLASS_EGFR"].values.astype(int)
    cv_folds = train_df["cv_fold"].values
    smiles_all = train_df["smiles"].tolist()
    smiles_test = test_df["smiles"].tolist()

    # ============================================================
    # 5-fold CV: train 7 base models per fold, collect OOF predictions
    # ============================================================
    print(f"\n[{EXP_ID}] Running 5-fold CV with 7 base models + Optuna weight optimization...")
    print(f"[{EXP_ID}] Models: XGB, LGB, ET, SVM-Tanimoto, GP-Tanimoto, CatBoost, Chemprop-MPNN")
    print(f"[{EXP_ID}] Optuna: 150 trials/fold, TPESampler(multivariate=True)")

    cv_fold_scores = []
    meta_preds_all = np.zeros(len(y_all))
    all_fold_weights = []
    all_fold_classical_models = []
    all_fold_chemprop = []

    for k in range(5):
        print(f"\n[{EXP_ID}] ===== Fold {k} =====")
        tr_mask = cv_folds != k
        val_mask = cv_folds == k

        X_tr_desc_k = X_desc[tr_mask]
        X_val_desc_k = X_desc[val_mask]
        X_tr_fp_k = X_fp[tr_mask]
        X_val_fp_k = X_fp[val_mask]
        y_tr_k = y_all[tr_mask]
        y_val_k = y_all[val_mask]

        smiles_tr_k = [s for s, m in zip(smiles_all, tr_mask) if m]
        smiles_val_k = [s for s, m in zip(smiles_all, val_mask) if m]
        X_xd_tr_k = X_xd_train[tr_mask]
        X_xd_val_k = X_xd_train[val_mask]

        pos_weight_k = float((y_tr_k == 0).sum()) / max(float((y_tr_k == 1).sum()), 1)

        # --- Train 6 classical models (incl CatBoost) ---
        print(f"[{EXP_ID}]   Training 6 classical base models (XGB, LGB, ET, SVM, GP, CatBoost)...")
        classical_models_k = train_classical_base_models(X_tr_desc_k, y_tr_k, X_tr_fp_k, pos_weight_k)
        all_fold_classical_models.append(classical_models_k)

        classical_val_preds = get_classical_preds(classical_models_k, X_val_desc_k, X_val_fp_k)
        print(f"[{EXP_ID}]   Classical val PR-AUCs: "
              f"XGB={average_precision_score(y_val_k, classical_val_preds[:,0]):.4f}, "
              f"LGB={average_precision_score(y_val_k, classical_val_preds[:,1]):.4f}, "
              f"ET={average_precision_score(y_val_k, classical_val_preds[:,2]):.4f}, "
              f"SVM={average_precision_score(y_val_k, classical_val_preds[:,3]):.4f}, "
              f"GP={average_precision_score(y_val_k, classical_val_preds[:,4]):.4f}, "
              f"CatBoost={average_precision_score(y_val_k, classical_val_preds[:,5]):.4f}")

        # --- Train Chemprop MPNN (7th model, GPU/CPU) ---
        print(f"[{EXP_ID}]   Training Chemprop MPNN (7th model, {DEVICE.upper()}, class_balance=True)...")
        chemprop_model_k, chemprop_trainer_k = train_chemprop_model(
            smiles_tr_k, y_tr_k,
            x_d_tr=X_xd_tr_k,
            epochs=60,
            d_h=300,
            seed=42
        )
        all_fold_chemprop.append((chemprop_model_k, chemprop_trainer_k))

        chemprop_val_preds = predict_chemprop_model(
            chemprop_model_k, chemprop_trainer_k, smiles_val_k, x_d=X_xd_val_k
        )
        chemprop_prauc = average_precision_score(y_val_k, chemprop_val_preds)
        print(f"[{EXP_ID}]   Chemprop val PR-AUC: {chemprop_prauc:.4f}")

        # Combine all 7 model OOF predictions
        oof_preds_k = np.column_stack([classical_val_preds, chemprop_val_preds.reshape(-1, 1)])

        # --- Optuna 7-way weight optimization (150 trials, TPE multivariate) ---
        print(f"[{EXP_ID}]   Optimizing 7-way weights with Optuna (150 trials, TPE multivariate)...")
        best_weights_k, optuna_score_k = optimize_weights_optuna(oof_preds_k, y_val_k, n_trials=150)
        all_fold_weights.append(best_weights_k)
        print(f"[{EXP_ID}]   Optimal weights: XGB={best_weights_k[0]:.3f}, LGB={best_weights_k[1]:.3f}, "
              f"ET={best_weights_k[2]:.3f}, SVM={best_weights_k[3]:.3f}, "
              f"GP={best_weights_k[4]:.3f}, CatBoost={best_weights_k[5]:.3f}, "
              f"Chemprop={best_weights_k[6]:.3f}")
        print(f"[{EXP_ID}]   Optuna best PR-AUC: {optuna_score_k:.4f}")

        # Weighted ensemble score
        weighted_val_preds = (oof_preds_k * best_weights_k).sum(axis=1)
        fold_score = average_precision_score(y_val_k, weighted_val_preds)
        cv_fold_scores.append(fold_score)
        meta_preds_all[val_mask] = weighted_val_preds
        print(f"[{EXP_ID}]   Fold {k} weighted PR-AUC: {fold_score:.4f}")

    # CV Summary
    overall_pr_auc = average_precision_score(y_all, meta_preds_all)
    mean_cv_pr_auc = np.mean(cv_fold_scores)
    std_cv_pr_auc = np.std(cv_fold_scores)
    avg_weights = np.mean(all_fold_weights, axis=0)

    print(f"\n[{EXP_ID}] ============ CV RESULTS ============")
    print(f"[{EXP_ID}] Fold scores: {[f'{s:.4f}' for s in cv_fold_scores]}")
    print(f"[{EXP_ID}] Mean CV PR-AUC: {mean_cv_pr_auc:.4f}")
    print(f"[{EXP_ID}] Std  CV PR-AUC: {std_cv_pr_auc:.4f}")
    print(f"[{EXP_ID}] Overall CV PR-AUC: {overall_pr_auc:.4f}")
    print(f"[{EXP_ID}] Avg weights: XGB={avg_weights[0]:.3f}, LGB={avg_weights[1]:.3f}, "
          f"ET={avg_weights[2]:.3f}, SVM={avg_weights[3]:.3f}, "
          f"GP={avg_weights[4]:.3f}, CatBoost={avg_weights[5]:.3f}, "
          f"Chemprop={avg_weights[6]:.3f}")

    # ============================================================
    # Final Test Predictions: train on full data
    # ============================================================
    print(f"\n[{EXP_ID}] Training final models on full data for test predictions...")

    print(f"[{EXP_ID}]   Training 6 classical models on full data (incl CatBoost)...")
    classical_final = train_classical_base_models(X_desc, y_all, X_fp, scale_pos_weight)
    test_classical_preds = get_classical_preds(classical_final, X_test_desc, X_test_fp)

    print(f"[{EXP_ID}]   Training Chemprop MPNN on full data ({DEVICE.upper()})...")
    chemprop_final, chemprop_final_trainer = train_chemprop_model(
        smiles_all, y_all,
        x_d_tr=X_xd_train,
        epochs=60,
        d_h=300,
        seed=42
    )
    test_chemprop_preds = predict_chemprop_model(
        chemprop_final, chemprop_final_trainer, smiles_test, x_d=X_xd_test
    )

    # Apply average CV weights to test predictions
    test_preds_matrix = np.column_stack([test_classical_preds, test_chemprop_preds.reshape(-1, 1)])
    final_test_preds = (test_preds_matrix * avg_weights).sum(axis=1)

    # ============================================================
    # Save Results
    # ============================================================
    print(f"\n[{EXP_ID}] Saving results...")

    submission = test_df[["id"]].copy()
    submission["CLASS_EGFR"] = final_test_preds
    submission_path = AGENT_WORKSPACE / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"[{EXP_ID}] Submission saved: {submission_path}")

    submission_stamped = AGENT_WORKSPACE / f"submission_{EXP_ID}.csv"
    shutil.copy(submission_path, submission_stamped)

    train_stamped = AGENT_WORKSPACE / f"train_{EXP_ID}.py"
    if Path(__file__).resolve() != train_stamped.resolve():
        shutil.copy(__file__, train_stamped)

    champion_pr_auc = 0.7941  # Current champion score (exp_beta_gpu2_c3)
    delta = mean_cv_pr_auc - champion_pr_auc
    outcome = "KEEP" if mean_cv_pr_auc > champion_pr_auc else "DISCARD"

    result_json = {
        "val_score": float(mean_cv_pr_auc),
        "direction": "maximize",
        "exp_id": EXP_ID,
                "submission_path": str(submission_stamped),
        "train_path": str(train_stamped),
        "model": "Optuna-weighted stacking: XGB+LGB+ET+SVM-Tanimoto+GP-Tanimoto+CatBoost+Chemprop-MPNN",
        "cv_fold_scores": [float(s) for s in cv_fold_scores],
        "cv_mean": float(mean_cv_pr_auc),
        "cv_std": float(std_cv_pr_auc),
        "cv_overall": float(overall_pr_auc),
        "avg_weights": {
            "xgb": float(avg_weights[0]),
            "lgb": float(avg_weights[1]),
            "extratrees": float(avg_weights[2]),
            "svm": float(avg_weights[3]),
            "gp": float(avg_weights[4]),
            "catboost": float(avg_weights[5]),
            "chemprop": float(avg_weights[6]),
        },
        "delta_vs_champion": float(delta),
        "outcome": outcome,
        "hyperparameters": {
            "catboost_iterations": 500,
            "catboost_lr": 0.05,
            "catboost_depth": 6,
            "catboost_auto_class_weights": "Balanced",
            "chemprop_d_h": 300,
            "chemprop_depth": 3,
            "chemprop_dropout": 0.1,
            "chemprop_epochs": 60,
            "chemprop_class_balance": True,
            "xgb_max_depth": 5,
            "xgb_lr": 0.035,
            "xgb_n_estimators": 434,
            "optuna_trials": 150,
            "optuna_sampler": "TPESampler(multivariate=True)",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # result_latest.json lives in workspace/ (one level above workspace/repo/)
    result_path = AGENT_WORKSPACE.parent / "result_latest.json"
    with open(result_path, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"[{EXP_ID}] result_latest.json saved: {result_path}")

    print(f"\n[{EXP_ID}] ============ EXPERIMENT COMPLETE ============")
    print(f"[{EXP_ID}] Mean CV PR-AUC: {mean_cv_pr_auc:.4f} +/- {std_cv_pr_auc:.4f}")
    print(f"[{EXP_ID}] Delta vs champion (0.7941): {delta:+.4f}")
    print(f"[{EXP_ID}] Outcome: {outcome}")
    print(f"[{EXP_ID}] Fold scores: {[f'{s:.4f}' for s in cv_fold_scores]}")
    print(f"[{EXP_ID}] End time: {datetime.now(timezone.utc).isoformat()}")

    return mean_cv_pr_auc


if __name__ == "__main__":
    main()
