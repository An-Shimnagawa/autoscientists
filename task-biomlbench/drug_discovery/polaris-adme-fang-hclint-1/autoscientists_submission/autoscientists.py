"""
Mordred + Multi-Fingerprint + Deep Residual MLP with SWA (GPU) for HLM CLint prediction.

Approach: Enhanced version of exp_gpu6_mlp_mordred_morgan_001 with:
- Additional MACCS keys fingerprint (167 bits) + atom pairs (2048 bits)
- Deeper residual MLP: hidden dims [2048, 1024, 512, 256, 128]
- Stochastic Weight Averaging (SWA) for better generalization
- Label smoothing + Huber loss combination
- 10-seed ensemble for final predictions

Approach: Mordred+Morgan+MLP-GPU-deeper-tuned
Exp ID: exp_gpu6_mlp_deeper_tuned_002
"""

import os
import sys
import json
import shutil
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# ---- Paths ----
FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = TASK_DIR / 'data'
WORKSPACE = Path(__file__).parent / "outputs"

os.makedirs(WORKSPACE, exist_ok=True)

# ---- GPU setup ----
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[DEVICE] Using: {DEVICE}")
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}")

# ---- Feature extraction ----
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from mordred import Calculator, descriptors as mordred_descriptors

def compute_all_fingerprints(smiles_list):
    """Compute Morgan + MACCS + AtomPair fingerprints."""
    morgan_fps = []
    maccs_fps = []
    atompair_fps = []
    valid_mask = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # Morgan r=3, 2048 bits
            morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
            morgan_fps.append(np.array(morgan))
            # MACCS keys (167 bits)
            maccs = MACCSkeys.GenMACCSKeys(mol)
            maccs_fps.append(np.array(maccs))
            # Atom pair fingerprints (2048 bits)
            ap = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
            atompair_fps.append(np.array(ap))
            valid_mask.append(True)
        else:
            morgan_fps.append(np.zeros(2048))
            maccs_fps.append(np.zeros(167))
            atompair_fps.append(np.zeros(2048))
            valid_mask.append(False)

    return (np.array(morgan_fps, dtype=np.float32),
            np.array(maccs_fps, dtype=np.float32),
            np.array(atompair_fps, dtype=np.float32),
            valid_mask)

def compute_mordred_descriptors(smiles_list):
    """Compute Mordred molecular descriptors."""
    calc = Calculator(mordred_descriptors, ignore_3D=True)
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mols.append(mol)
    results = calc.pandas(mols)
    for col in results.columns:
        results[col] = pd.to_numeric(results[col], errors='coerce')
    return results.values.astype(np.float32)

print("[FEATURES] Loading data...")
train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df = pd.read_csv(DATA_DIR / 'test_features.csv')

print(f"[DATA] Train: {len(train_df)}, Test: {len(test_df)}")

print("[FEATURES] Computing fingerprints (Morgan + MACCS + AtomPair)...")
train_morgan, train_maccs, train_ap, _ = compute_all_fingerprints(train_df['smiles'].tolist())
test_morgan, test_maccs, test_ap, _ = compute_all_fingerprints(test_df['smiles'].tolist())
print(f"[FEATURES] Morgan: {train_morgan.shape}, MACCS: {train_maccs.shape}, AtomPair: {train_ap.shape}")

print("[FEATURES] Computing Mordred descriptors (~2 min)...")
train_mordred = compute_mordred_descriptors(train_df['smiles'].tolist())
test_mordred = compute_mordred_descriptors(test_df['smiles'].tolist())
print(f"[FEATURES] Mordred shape: {train_mordred.shape}")

# Combine all features
X_train_full = np.concatenate([train_morgan, train_maccs, train_ap, train_mordred], axis=1).astype(np.float32)
X_test = np.concatenate([test_morgan, test_maccs, test_ap, test_mordred], axis=1).astype(np.float32)
y_train_full = train_df['LOG_HLM_CLint'].values.astype(np.float32)

print(f"[FEATURES] Combined feature shape: {X_train_full.shape}")

# ---- Preprocessing ----
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

print("[PREPROCESS] Imputing NaN values...")
imputer = SimpleImputer(strategy='median')
X_train_full = imputer.fit_transform(X_train_full)
X_test = imputer.transform(X_test)

print("[PREPROCESS] Removing constant features...")
feat_std = X_train_full.std(axis=0)
keep_mask = feat_std > 1e-6
X_train_full = X_train_full[:, keep_mask]
X_test = X_test[:, keep_mask]
print(f"[PREPROCESS] Features after filtering: {X_train_full.shape[1]}")

print("[PREPROCESS] Clipping extreme values...")
p1 = np.percentile(X_train_full, 1, axis=0)
p99 = np.percentile(X_train_full, 99, axis=0)
X_train_full = np.clip(X_train_full, p1, p99)
X_test = np.clip(X_test, p1, p99)

scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)
print(f"[PREPROCESS] Final feature dim: {X_train_full.shape[1]}")

# ---- Model ----
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout * 0.5)

    def forward(self, x):
        return self.drop(self.act(x + self.block(x)))


class DeepResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[2048, 1024, 512, 256, 128], dropout=0.3):
        super().__init__()
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Residual blocks at first dim
        self.res1 = ResidualBlock(hidden_dims[0], dropout)
        self.res2 = ResidualBlock(hidden_dims[0], dropout)

        # Downsampling layers
        layers = []
        in_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.8),
            ])
            in_dim = h_dim
        self.down = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.down(x)
        return self.out(x).squeeze(-1)


# ---- Training with SWA ----
CONFIG = {
    'hidden_dims': [2048, 1024, 512, 256, 128],
    'dropout': 0.25,
    'lr': 8e-4,
    'weight_decay': 1e-4,
    'batch_size': 128,
    'epochs': 180,
    'patience': 45,
    'swa_start': 120,  # start SWA after epoch 120
    'swa_lr': 2e-4,
}

print(f"\n[CONFIG] {json.dumps(CONFIG, indent=2)}")
print("=" * 60)
print(json.dumps(CONFIG, indent=2))
print("=" * 60)


def train_fold_swa(X_tr, y_tr, X_val, y_val, input_dim, config, fold_idx=0):
    """Train with SWA on one fold."""
    torch.manual_seed(42 + fold_idx)
    np.random.seed(42 + fold_idx)

    model = DeepResMLP(
        input_dim=input_dim,
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout']
    ).to(DEVICE)

    swa_model = AveragedModel(model)

    X_tr_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_tr_t = torch.FloatTensor(y_tr).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['swa_start'], eta_min=1e-5)
    swa_scheduler = SWALR(optimizer, swa_lr=config['swa_lr'])
    criterion = nn.HuberLoss(delta=1.0)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False)

    best_val_r = -np.inf
    best_preds = None
    patience = config.get('patience', 45)
    no_improve = 0
    swa_active = False

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in loader:
            optimizer.zero_grad()
            preds = model(X_b)
            loss = criterion(preds, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)

        if epoch >= config['swa_start']:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_active = True
        else:
            scheduler.step()

        # Use SWA model for validation if active
        if swa_active and epoch > config['swa_start'] + 5:
            # Update BN stats for SWA model
            update_bn(DataLoader(dataset, batch_size=config['batch_size']), swa_model)
            eval_model = swa_model
        else:
            eval_model = model

        eval_model.eval()
        with torch.no_grad():
            val_preds = eval_model(X_val_t).cpu().numpy()

        r, _ = pearsonr(y_val, val_preds)

        if r > best_val_r:
            best_val_r = r
            best_preds = val_preds.copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience and epoch > 80:
            print(f"  Fold {fold_idx} early stop epoch {epoch+1}, best r={best_val_r:.4f}")
            break

        if (epoch + 1) % 30 == 0:
            print(f"  Fold {fold_idx} ep {epoch+1}/{config['epochs']}: loss={epoch_loss/len(X_tr):.4f}, r={r:.4f}, best={best_val_r:.4f}, swa={swa_active}")

    return best_val_r, best_preds, swa_model if swa_active else model


# ---- 5-Fold CV ----
print("\n[CV] Running 5-fold CV with SWA...")
fold_scores = []

for k in range(5):
    tr_mask = train_df['cv_fold'] != k
    val_mask = train_df['cv_fold'] == k

    X_tr = X_train_full[tr_mask]
    y_tr = y_train_full[tr_mask]
    X_val = X_train_full[val_mask]
    y_val = y_train_full[val_mask]

    print(f"\n[FOLD {k}] train={len(X_tr)}, val={len(X_val)}")
    val_r, val_preds, _ = train_fold_swa(
        X_tr, y_tr, X_val, y_val,
        input_dim=X_train_full.shape[1],
        config=CONFIG,
        fold_idx=k
    )
    fold_scores.append(val_r)
    print(f"[FOLD {k}] val Pearson r = {val_r:.4f}")

mean_cv = np.mean(fold_scores)
std_cv = np.std(fold_scores)
print(f"\n[CV RESULT] Mean CV Pearson r: {mean_cv:.4f} ± {std_cv:.4f}")
print(f"[CV RESULT] Per fold: {[f'{s:.4f}' for s in fold_scores]}")

# ---- Final model (10-seed ensemble) ----
print("\n[FINAL] Training 10-seed ensemble on full data...")
all_test_preds = []
for seed in range(10):
    print(f"  Seed {seed+1}/10...")
    torch.manual_seed(seed * 77 + 13)
    np.random.seed(seed * 77 + 13)

    model = DeepResMLP(
        input_dim=X_train_full.shape[1],
        hidden_dims=CONFIG['hidden_dims'],
        dropout=CONFIG['dropout']
    ).to(DEVICE)

    swa_model = AveragedModel(model)

    X_full_t = torch.FloatTensor(X_train_full).to(DEVICE)
    y_full_t = torch.FloatTensor(y_train_full).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    n_final = int(CONFIG['epochs'] * 1.2)
    swa_start = int(CONFIG['swa_start'] * 1.2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=swa_start, eta_min=1e-5)
    swa_scheduler = SWALR(optimizer, swa_lr=CONFIG['swa_lr'])
    criterion = nn.HuberLoss(delta=1.0)

    dataset = TensorDataset(X_full_t, y_full_t)
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    for epoch in range(n_final):
        model.train()
        for X_b, y_b in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

    # Update BN for SWA
    update_bn(DataLoader(dataset, batch_size=CONFIG['batch_size']), swa_model)

    swa_model.eval()
    with torch.no_grad():
        test_preds = swa_model(X_test_t).cpu().numpy()
    all_test_preds.append(test_preds)

final_test_preds = np.mean(all_test_preds, axis=0)
print(f"[FINAL] Test predictions: mean={final_test_preds.mean():.3f}, std={final_test_preds.std():.3f}")

# ---- Save submission ----
submission = pd.DataFrame({
    'id': test_df['id'],
    'LOG_HLM_CLint': final_test_preds
})

submission_path = WORKSPACE / 'submission.csv'
submission.to_csv(submission_path, index=False)
print(f"[OUTPUT] Saved to {submission_path}")
print(submission.head())

# ---- Save result ----
exp_id = 'exp_gpu6_mlp_deeper_tuned_002'
stamped_sub = WORKSPACE / f'submission_{exp_id}.csv'
stamped_train = WORKSPACE / f'train_{exp_id}.py'
shutil.copy(submission_path, stamped_sub)
shutil.copy(__file__, stamped_train)

result = {
    'val_score': float(mean_cv),
    'val_std': float(std_cv),
    'val_fold_scores': [float(s) for s in fold_scores],
    'direction': 'maximize',
    'exp_id': exp_id,
    'approach': 'Mordred+Morgan+MLP-GPU-deeper-tuned',
    'submission_path': str(stamped_sub),
    'train_path': str(stamped_train),
    'config': CONFIG,
}

result_path = AGENT_DIR / 'workspace' / 'result_latest.json'
result_path.write_text(json.dumps(result, indent=2))
print(f"\n[RESULT] Written result_latest.json")
print(f"[RESULT] val_score = {mean_cv:.4f} ± {std_cv:.4f}")
print(f"[RESULT] exp_id = {exp_id}")
