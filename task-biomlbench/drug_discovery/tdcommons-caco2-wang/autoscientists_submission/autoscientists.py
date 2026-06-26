"""
Experiment: LightGBM-Optuna-ECFP-Mordred (exp_alpha_004)
Agent: biomlbdc_caco2_3_gpu2
Team: alpha
Approach: Optuna-tuned LightGBM on ECFP4+ECFP6+Mordred 2D descriptors.
- Pre-compute features for all molecules once
- Optuna per-fold tuning for 50 trials
- LGB with MAE objective, early stopping
- 5-fold scaffold CV
- Metric: MAE (lower is better)
- Retrain on ALL training data for final test predictions
"""

import os
import sys
import json
import shutil
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# Paths
FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"WORKSPACE = Path(__file__).parent / "outputs"
SUBMISSION_PATH = WORKSPACE / "submission.csv"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test_features.csv"

HP = {
    "ecfp_radius2_bits": 2048,
    "ecfp_radius3_bits": 2048,
    "cv_folds": 5,
    "random_seed": 42,
    "optuna_trials": 50,
}

print("=" * 60)
print(json.dumps({k: str(v) for k, v in HP.items()}, indent=2))
print("=" * 60)


def compute_ecfp(smiles_list, radius=2, nbits=2048):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(nbits))
        else:
            gen = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fps.append(np.array(gen))
    return np.array(fps, dtype=np.float32)


def precompute_all_features(all_smiles):
    from rdkit import Chem
    from mordred import Calculator, descriptors as mordred_descriptors
    from sklearn.impute import SimpleImputer

    print(f"\n[PRECOMPUTE] Computing features for {len(all_smiles)} molecules...")
    print("  Computing ECFP4...")
    ecfp4 = compute_ecfp(all_smiles, radius=2, nbits=HP["ecfp_radius2_bits"])
    print("  Computing ECFP6...")
    ecfp6 = compute_ecfp(all_smiles, radius=3, nbits=HP["ecfp_radius3_bits"])

    calc = Calculator(mordred_descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(s) for s in all_smiles]
    print(f"  Computing Mordred descriptors for {len(mols)} mols...")
    mordred_df = calc.pandas(mols)
    mordred_df = mordred_df.apply(pd.to_numeric, errors='coerce')

    # Filter
    nan_frac = mordred_df.isnull().mean()
    mordred_df = mordred_df.loc[:, nan_frac <= 0.2]
    std = mordred_df.std()
    mordred_df = mordred_df.loc[:, std > 1e-8]
    mordred_cols = list(mordred_df.columns)
    print(f"  Mordred cols: {len(mordred_cols)}")

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    mordred_arr = imputer.fit_transform(mordred_df.values).astype(np.float32)

    return ecfp4, ecfp6, mordred_arr, mordred_cols, imputer


def main():
    print(f"\n[{datetime.now()}] Loading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    all_smiles = list(train_df["Drug"]) + list(test_df["Drug"])
    n_train = len(train_df)

    ecfp4, ecfp6, mordred_arr, mordred_cols, imputer = precompute_all_features(all_smiles)

    train_ecfp4 = ecfp4[:n_train]
    train_ecfp6 = ecfp6[:n_train]
    train_mordred = mordred_arr[:n_train]
    test_ecfp4 = ecfp4[n_train:]
    test_ecfp6 = ecfp6[n_train:]
    test_mordred = mordred_arr[n_train:]

    y_all = train_df["Y"].values

    # 5-fold CV with Optuna per fold
    import optuna
    import lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    fold_maes = []
    best_params_per_fold = []

    for k in range(HP["cv_folds"]):
        print(f"\n{'='*50}")
        print(f"Fold {k}")

        train_mask = train_df["cv_fold"] != k
        val_mask = train_df["cv_fold"] == k
        tr_idx = np.where(train_mask.values)[0]
        val_idx = np.where(val_mask.values)[0]

        print(f"  Train: {len(tr_idx)}, Val: {len(val_idx)}")

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        mordred_tr = scaler.fit_transform(train_mordred[tr_idx]).astype(np.float32)
        mordred_val = scaler.transform(train_mordred[val_idx]).astype(np.float32)

        X_train = np.concatenate([train_ecfp4[tr_idx], train_ecfp6[tr_idx], mordred_tr], axis=1)
        X_val = np.concatenate([train_ecfp4[val_idx], train_ecfp6[val_idx], mordred_val], axis=1)
        y_tr = y_all[tr_idx]
        y_val = y_all[val_idx]

        print(f"  Feature shape: {X_train.shape}")

        def objective(trial):
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
                "objective": "mae",
                "random_state": HP["random_seed"],
                "n_jobs": 8,
                "verbose": -1,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            preds = model.predict(X_val)
            return np.mean(np.abs(preds - y_val))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=HP["random_seed"] + k)
        )
        study.optimize(objective, n_trials=HP["optuna_trials"], show_progress_bar=False)

        best_p = study.best_params
        best_p.update({"objective": "mae", "random_state": HP["random_seed"],
                       "n_jobs": 8, "verbose": -1, "subsample_freq": 1})

        # Retrain with best params
        final_model = lgb.LGBMRegressor(**best_p)
        final_model.fit(X_train, y_tr)
        val_preds = final_model.predict(X_val)
        mae = np.mean(np.abs(val_preds - y_val))
        fold_maes.append(mae)
        best_params_per_fold.append(best_p)

        print(f"  Fold {k} MAE: {mae:.4f} (Optuna best: {study.best_value:.4f})")

    mean_mae = np.mean(fold_maes)
    std_mae = np.std(fold_maes)
    print(f"\n{'='*60}")
    print(f"Mean CV MAE: {mean_mae:.4f} ± {std_mae:.4f} (per fold: {[f'{s:.4f}' for s in fold_maes]})")
    print(f"{'='*60}")

    # Final training on all data
    print(f"\n[{datetime.now()}] Retraining on all training data...")
    from sklearn.preprocessing import StandardScaler
    scaler_final = StandardScaler()
    mordred_all = scaler_final.fit_transform(train_mordred).astype(np.float32)
    X_all = np.concatenate([train_ecfp4, train_ecfp6, mordred_all], axis=1)

    # Use average best params (or first fold params as representative)
    avg_params = {}
    int_keys = {'num_leaves', 'n_estimators', 'min_child_samples', 'subsample_freq', 'n_jobs', 'random_state', 'verbose'}
    for key in best_params_per_fold[0].keys():
        vals = [p.get(key) for p in best_params_per_fold if p.get(key) is not None]
        if key in int_keys:
            avg_params[key] = int(round(np.mean([v for v in vals if isinstance(v, (int, float))])))
        elif isinstance(vals[0], float):
            avg_params[key] = float(np.mean(vals))
        else:
            avg_params[key] = vals[0]

    print(f"  Final LGB params (averaged across folds): {avg_params}")
    final_lgb = lgb.LGBMRegressor(**avg_params)
    final_lgb.fit(X_all, y_all)

    # Test predictions
    print(f"\n[{datetime.now()}] Generating test predictions...")
    mordred_test = scaler_final.transform(test_mordred).astype(np.float32)
    X_test = np.concatenate([test_ecfp4, test_ecfp6, mordred_test], axis=1)
    test_preds = final_lgb.predict(X_test)

    print(f"  Test predictions: {len(test_preds)} samples")
    print(f"  Pred range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")

    # Save submission
    submission = pd.DataFrame({"id": test_df["id"], "Y": test_preds})
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n  Submission saved to: {SUBMISSION_PATH}")

    task_submission = FOCUS_ROOT / "submission.csv"
    submission.to_csv(task_submission, index=False)
    print(f"  Submission also saved to: {task_submission}")

    # Save result summary
    result_summary = {
        "val_score": float(mean_mae),
        "direction": "minimize",
        "exp_id": "exp_alpha_004",
        "submission_path": str(SUBMISSION_PATH),
        "fold_maes": [float(x) for x in fold_maes],
        "std_mae": float(std_mae),
    }
    result_path = WORKSPACE / "result_latest.json"
    result_path.write_text(json.dumps(result_summary, indent=2))
    print(f"  Result saved to: {result_path}")

    # Copy train script
    script_src = Path(__file__).resolve()
    dst_train = WORKSPACE / "train.py"
    if str(script_src) != str(dst_train):
        shutil.copy(str(script_src), str(dst_train))
    dst_task = FOCUS_ROOT / "train.py"
    shutil.copy(str(script_src), str(dst_task))

    print(f"\n[{datetime.now()}] Complete! CV MAE: {mean_mae:.4f} ± {std_mae:.4f}")
    return mean_mae, fold_maes


if __name__ == "__main__":
    main()
