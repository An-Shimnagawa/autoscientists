"""
Experiment: exp_gamma_003 — Chemprop MPNN (Message Passing Neural Network)
Team: gamma

Approach: Chemprop v2 MPNN directly on molecular graphs
- Message passing over bond-level atom/bond features
- BondMessagePassing (depth=4, hidden=300, dropout=0.1) + MeanAggregation
- RegressionFFN head (2 layers, hidden=512, dropout=0.1)
- 5-fold scaffold CV using cv_fold column (per task spec)
- Final model trained on full dataset, predicts test_features.csv

Rationale:
- Chemprop MPNN learns directly from molecular graph topology
- Fundamentally different from feature-based approaches (XGBoost + descriptors)
- Prior Chemprop benchmarks show strong performance on ADME tasks
- No hand-crafted features — end-to-end molecular representation learning
- Expected CV Pearson r: 0.60-0.70 based on prior ADME benchmarks
"""

import os, sys, shutil, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import pearsonr

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

from rdkit import Chem

import chemprop
from chemprop import data as cpdata, models as cpmodels, nn as cpnn
from chemprop.data import build_dataloader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# ── paths ────────────────────────────────────────────────────────────────────
FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"
AGENT_DIR  = Path(f"{FOCUS_ROOT}/agents/{AGENT_NAME}/workspace/repo")
AGENT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV  = DATA_DIR / "test_features.csv"
EXP_ID    = "exp_gamma_003"

# ── hyperparameters ──────────────────────────────────────────────────────────
HP = {
    "seed": 42,
    "cv_folds": 5,
    # Message passing
    "mp_depth": 4,
    "mp_hidden_dim": 300,
    "mp_dropout": 0.1,
    # FFN head
    "ffn_hidden_dim": 512,
    "ffn_n_layers": 2,
    "ffn_dropout": 0.1,
    # Training
    "max_epochs": 100,
    "batch_size": 64,
    "init_lr": 1e-4,
    "max_lr": 1e-3,
    "final_lr": 1e-4,
    "warmup_epochs": 2,
    "patience": 20,
    "num_workers": 0,
}
print("=" * 60)
print(json.dumps(HP, indent=2))
print("=" * 60)

torch.manual_seed(HP["seed"])
np.random.seed(HP["seed"])


def smiles_to_mol(smi):
    """Convert SMILES to RDKit mol, with fallback."""
    mol = Chem.MolFromSmiles(smi)
    return mol if mol is not None else Chem.MolFromSmiles("C")


def make_datapoints(smiles_list, targets=None):
    """Create Chemprop MoleculeDatapoints from SMILES + optional targets."""
    datapoints = []
    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        y = np.array([targets[i]], dtype=np.float32) if targets is not None else None
        dp = cpdata.MoleculeDatapoint(mol=mol, y=y)
        datapoints.append(dp)
    return datapoints


def build_model():
    """Build a fresh Chemprop MPNN model."""
    mp = cpnn.BondMessagePassing(
        d_h=HP["mp_hidden_dim"],
        depth=HP["mp_depth"],
        dropout=HP["mp_dropout"],
    )
    agg = cpnn.MeanAggregation()
    ffn = cpnn.RegressionFFN(
        n_tasks=1,
        input_dim=HP["mp_hidden_dim"],
        hidden_dim=HP["ffn_hidden_dim"],
        n_layers=HP["ffn_n_layers"],
        dropout=HP["ffn_dropout"],
    )
    model = cpmodels.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        warmup_epochs=HP["warmup_epochs"],
        init_lr=HP["init_lr"],
        max_lr=HP["max_lr"],
        final_lr=HP["final_lr"],
    )
    return model


def train_and_predict(train_dps, val_dps, test_dps, ckpt_dir, fold_k):
    """Train MPNN on train_dps, evaluate on val_dps, predict test_dps."""
    train_dataset = cpdata.MoleculeDataset(train_dps)
    val_dataset   = cpdata.MoleculeDataset(val_dps)
    test_dataset  = cpdata.MoleculeDataset(test_dps)

    train_loader = build_dataloader(train_dataset, batch_size=HP["batch_size"],
                                    num_workers=HP["num_workers"], shuffle=True,
                                    seed=HP["seed"])
    val_loader   = build_dataloader(val_dataset, batch_size=HP["batch_size"],
                                    num_workers=HP["num_workers"], shuffle=False)
    test_loader  = build_dataloader(test_dataset, batch_size=HP["batch_size"],
                                    num_workers=HP["num_workers"], shuffle=False)

    model = build_model()

    ckpt_path = str(ckpt_dir / f"fold_{fold_k}_best.ckpt")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=HP["patience"], mode="min", verbose=False),
        ModelCheckpoint(dirpath=str(ckpt_dir), filename=f"fold_{fold_k}_best",
                        monitor="val_loss", mode="min", save_top_k=1),
    ]

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=HP["max_epochs"],
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint for inference
    best_model = cpmodels.MPNN.load_from_checkpoint(ckpt_path)
    best_model.eval()

    # Val predictions
    val_preds_raw = trainer.predict(best_model, val_loader)
    val_preds = np.concatenate([p.numpy() for p in val_preds_raw]).flatten()

    # Test predictions
    test_preds_raw = trainer.predict(best_model, test_loader)
    test_preds = np.concatenate([p.numpy() for p in test_preds_raw]).flatten()

    return val_preds, test_preds


def main():
    # Load data
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    train_smiles = train_df["smiles"].tolist()
    test_smiles  = test_df["smiles"].tolist()
    y_train      = train_df["LOG_SOLUBILITY"].values.astype(np.float32)
    cv_folds     = train_df["cv_fold"].values

    print(f"Train: {len(train_smiles)}, Test: {len(test_smiles)}")

    # Checkpoint directory
    ckpt_dir = AGENT_DIR / "ckpts" / EXP_ID
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build all test datapoints once (no targets)
    test_dps = make_datapoints(test_smiles, targets=None)

    # ── 5-fold scaffold CV ───────────────────────────────────────────────────
    fold_scores   = []
    test_preds_cv = np.zeros(len(test_smiles))

    for k in range(HP["cv_folds"]):
        tr_mask  = cv_folds != k
        val_mask = cv_folds == k

        tr_smiles_k  = [s for s, m in zip(train_smiles, tr_mask)  if m]
        val_smiles_k = [s for s, m in zip(train_smiles, val_mask) if m]
        y_tr_k       = y_train[tr_mask]
        y_val_k      = y_train[val_mask]

        train_dps_k = make_datapoints(tr_smiles_k,  targets=y_tr_k)
        val_dps_k   = make_datapoints(val_smiles_k, targets=y_val_k)

        print(f"\nFold {k}: train={len(train_dps_k)}, val={len(val_dps_k)}")
        val_preds_k, test_preds_k = train_and_predict(
            train_dps_k, val_dps_k, test_dps, ckpt_dir, k
        )

        r, _ = pearsonr(y_val_k, val_preds_k)
        fold_scores.append(float(r))
        test_preds_cv += test_preds_k / HP["cv_folds"]

        print(f"  Fold {k} Pearson r: {r:.4f}")

    mean_score = float(np.mean(fold_scores))
    std_score  = float(np.std(fold_scores))
    print(f"\nCV Pearson r: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Per fold: {[f'{s:.4f}' for s in fold_scores]}")

    # ── Final model on full dataset ──────────────────────────────────────────
    print("\nTraining final model on full training set...")
    all_train_dps = make_datapoints(train_smiles, targets=y_train)
    test_dps_final = make_datapoints(test_smiles, targets=None)

    # For final model, use val=fold 4 for early stopping
    final_val_mask  = cv_folds == 4
    final_tr_mask   = ~final_val_mask
    final_tr_dps    = [dp for dp, m in zip(all_train_dps, final_tr_mask) if m]
    final_val_dps   = [dp for dp, m in zip(all_train_dps, final_val_mask) if m]

    _, final_test_preds = train_and_predict(
        final_tr_dps, final_val_dps, test_dps_final, ckpt_dir, "final"
    )

    # Blend CV-averaged and final predictions (equal weight)
    test_preds = (test_preds_cv + final_test_preds) / 2

    # ── Save submission ──────────────────────────────────────────────────────
    submission = pd.DataFrame({"id": test_df["id"], "LOG_SOLUBILITY": test_preds})
    submission_path = AGENT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Saved submission to {submission_path}")

    # Stamped copies (isolation rule)
    stamped_sub   = AGENT_DIR / f"submission_{EXP_ID}.csv"
    stamped_train = AGENT_DIR / f"train_{EXP_ID}.py"
    shutil.copy(str(submission_path), str(stamped_sub))

    src = Path(sys.argv[0]).resolve()
    if src != stamped_train.resolve():
        shutil.copy(str(src), str(stamped_train))

    print(f"Saved stamped: {stamped_sub}")
    print(f"Saved stamped: {stamped_train}")

    # ── Write result_latest.json ─────────────────────────────────────────────
    result_summary = {
        "val_score":       mean_score,
        "val_std":         std_score,
        "direction":       "maximize",
        "exp_id":          EXP_ID,
                "team":            "gamma",
        "fold_scores":     fold_scores,
        "submission_path": str(stamped_sub),
        "train_path":      str(stamped_train),
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "hyperparameters": HP,
    }
    result_path = Path(f"{FOCUS_ROOT}/agents/{AGENT_NAME}/workspace/result_latest.json")
    result_path.write_text(json.dumps(result_summary, indent=2))
    print(f"\n[result_latest.json] written — val_score={mean_score:.4f}")

    return mean_score, std_score, fold_scores


if __name__ == "__main__":
    score, std, folds = main()
    print(f"\n{'='*60}")
    print(f"FINAL: Pearson r = {score:.4f} ± {std:.4f}")
    print(f"{'='*60}")
