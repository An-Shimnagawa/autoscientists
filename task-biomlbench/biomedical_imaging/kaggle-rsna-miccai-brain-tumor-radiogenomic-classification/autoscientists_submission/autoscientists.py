"""
exp_gamma_011: EfficientNet-B0 + CosineWarmRestart(T_0=10) + LabelSmoothing

Hypothesis: The champion (gamma_008) used T_0=5 (short cosine cycles 5,10,20).
            Shorter cycles may restart LR too aggressively before converging.
            T_0=10 gives the model longer to descend into minima (cycles 10,20)
            before the LR resets, potentially reaching flatter/better minima.

Architecture: Identical to champion (exp_gamma_008/exp_beta_003):
  EfficientNet-B0 + SliceAttentionPool (per-sequence) + 2-layer cross-sequence
  Transformer (4 heads, CLS token)

Key changes from gamma_008 (champion):
  - CosineWarmRestart T_0=10 instead of T_0=5 (longer initial cycle)
  - N_EPOCHS=25 to cover 1.5 cycles (10 + 15 of 20-epoch cycle 2)
  - Everything else identical (AdamW, LabelSmoothing=0.05, patience=8)
"""

import os
import sys
import json
import shutil
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import pydicom
from sklearn.metrics import roc_auc_score

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms

warnings.filterwarnings("ignore")

FOCUS_ROOT = Path(__file__).parent.parent
DATA_ROOT = FOCUS_ROOT / "data"EXP_ID = "exp_gamma_011"
WORKSPACE = Path(__file__).parent / "outputs"

SEED = 42
N_SLICES = 8
IMG_SIZE = 224
BATCH_SIZE = 4
N_EPOCHS = 25
LR = 2e-4
WEIGHT_DECAY = 1e-3
EARLY_STOP_PATIENCE = 8
LABEL_SMOOTHING = 0.05
COSINE_T0 = 10        # longer initial cycle (vs 5 in champion)
COSINE_T_MULT = 2
N_SEQ_HEADS = 4
N_TRANSFORMER_LAYERS = 2
SEQUENCES = ["FLAIR", "T1w", "T1wCE", "T2w"]
EXCLUDE_IDS = {"00109", "00123", "00709"}
TEST_DIR = f"{DATA_ROOT}/test_biomlbench"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_dicom_slices(seq_dir: str, n_slices: int = N_SLICES) -> np.ndarray:
    seq_path = Path(seq_dir)
    if not seq_path.exists():
        return np.zeros((n_slices, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    files = sorted(
        seq_path.glob("*.dcm"),
        key=lambda f: int(''.join(filter(str.isdigit, f.stem)) or 0)
    )
    if len(files) == 0:
        return np.zeros((n_slices, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    n_total = len(files)
    start = int(n_total * 0.30)
    end = int(n_total * 0.70)
    if end - start < n_slices:
        start = max(0, n_total // 2 - n_slices // 2)
        end = min(n_total, start + n_slices)
    indices = np.linspace(start, end - 1, n_slices, dtype=int)
    slices = []
    for idx in indices:
        try:
            dcm = pydicom.dcmread(str(files[idx]))
            arr = dcm.pixel_array.astype(np.float32)
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = (arr - arr_min) / (arr_max - arr_min)
            from PIL import Image as PILImage
            img = PILImage.fromarray((arr * 255).astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE), PILImage.BILINEAR
            )
            arr = np.array(img, dtype=np.float32) / 255.0
            slices.append(arr)
        except Exception:
            slices.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))
    while len(slices) < n_slices:
        slices.append(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32))
    return np.stack(slices[:n_slices], axis=0)


class BrainMRIDataset(Dataset):
    def __init__(self, patient_ids, labels, data_dir, is_test=False, augment=False):
        self.patient_ids = patient_ids
        self.labels = labels
        self.data_dir = data_dir
        self.is_test = is_test
        self.augment = augment
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        pid_str = str(pid).zfill(5)
        seq_tensors = []
        for seq in SEQUENCES:
            seq_dir = f"{self.data_dir}/{pid_str}/{seq}"
            slices = load_dicom_slices(seq_dir)
            t = torch.from_numpy(slices).float().unsqueeze(1).repeat(1, 3, 1, 1)
            t = torch.stack([self.normalize(s) for s in t])
            if self.augment and not self.is_test:
                if random.random() > 0.5:
                    t = torch.flip(t, dims=[-1])
            seq_tensors.append(t)
        x = torch.stack(seq_tensors, dim=0)
        if self.is_test:
            return x, pid_str
        else:
            label = self.labels.get(pid_str, 0)
            return x, torch.tensor(label, dtype=torch.float32)


class SliceAttentionPool(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.Tanh(), nn.Linear(64, 1)
        )

    def forward(self, x):
        weights = self.attn(x)
        weights = F.softmax(weights, dim=1)
        return (x * weights).sum(dim=1)


class CrossSequenceTransformerModel(nn.Module):
    def __init__(self, n_sequences=4, n_slices=N_SLICES,
                 n_transformer_layers=N_TRANSFORMER_LAYERS,
                 n_heads=N_SEQ_HEADS, dropout=0.3):
        super().__init__()
        self.n_sequences = n_sequences
        self.n_slices = n_slices
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, global_pool="")
        self.feat_dim = self.backbone.num_features
        print(f"EfficientNet-B0 feature dim: {self.feat_dim}")
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.slice_attn = SliceAttentionPool(self.feat_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.feat_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_sequences + 1, self.feat_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feat_dim, nhead=n_heads,
            dim_feedforward=self.feat_dim * 2, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 256), nn.LayerNorm(256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1)
        )

    def forward(self, x):
        B, S, T, C, H, W = x.shape
        seq_embeds = []
        for s_idx in range(S):
            x_seq = x[:, s_idx, :, :, :, :]
            x_flat = x_seq.reshape(B * T, C, H, W)
            feats = self.backbone(x_flat)
            feats = self.gap(feats).squeeze(-1).squeeze(-1)
            feats = feats.view(B, T, self.feat_dim)
            pooled = self.slice_attn(feats)
            seq_embeds.append(pooled)
        seq_stack = torch.stack(seq_embeds, dim=1)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, seq_stack], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.transformer(tokens)
        cls_out = tokens[:, 0, :]
        return self.classifier(cls_out).squeeze(-1)


class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        smooth_targets = targets * (1 - self.smoothing) + self.smoothing * 0.5
        return F.binary_cross_entropy_with_logits(logits, smooth_targets)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(batch_y)
        preds = torch.sigmoid(logits).detach().cpu().float().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch_y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.5
    return avg_loss, auc


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            with torch.cuda.amp.autocast():
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_y)
            preds = torch.sigmoid(logits).cpu().float().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.5
    return avg_loss, auc, all_preds, all_labels


def predict(model, loader, device):
    model.eval()
    all_preds, all_ids = [], []
    with torch.no_grad():
        for batch_x, batch_ids in loader:
            batch_x = batch_x.to(device)
            with torch.cuda.amp.autocast():
                logits = model(batch_x)
            preds = torch.sigmoid(logits).cpu().float().numpy()
            all_preds.extend(preds.tolist())
            all_ids.extend(batch_ids)
    return all_ids, all_preds


def main():
    print("=" * 70)
    print(f"exp_gamma_011: EfficientNet-B0 + CosineWarmRestart(T_0=10) + LabelSmoothing")
    print(f"N_SLICES={N_SLICES}, BATCH_SIZE={BATCH_SIZE}, N_EPOCHS={N_EPOCHS}")
    print(f"AdamW wd={WEIGHT_DECAY}, label_smooth={LABEL_SMOOTHING}")
    print(f"CosineWarmRestart: T_0={COSINE_T0}, T_mult={COSINE_T_MULT} (longer cycles vs champion T_0=5)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    sample_sub = pd.read_csv(f"{DATA_ROOT}/sample_submission.csv", dtype={"BraTS21ID": str})
    sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].str.zfill(5)
    test_ids = sample_sub["BraTS21ID"].tolist()
    print(f"Test patients: {len(test_ids)}")

    test_check = os.path.join(TEST_DIR, test_ids[0], "FLAIR")
    if os.path.exists(test_check):
        print(f"Test data verified: {test_ids[0]}/FLAIR has {len(os.listdir(test_check))} slices")

    labels_df = pd.read_csv(f"{DATA_ROOT}/train_labels.csv", dtype={"BraTS21ID": str})
    labels_df["BraTS21ID"] = labels_df["BraTS21ID"].str.zfill(5)
    labels_df = labels_df[~labels_df["BraTS21ID"].isin(EXCLUDE_IDS)]
    print(f"Training patients: {len(labels_df)}, MGMT+ rate: {labels_df['MGMT_value'].mean():.3f}")
    label_dict = dict(zip(labels_df["BraTS21ID"], labels_df["MGMT_value"]))

    criterion = LabelSmoothingBCELoss(smoothing=LABEL_SMOOTHING)
    fold_scores = []

    for k in range(5):
        print(f"\n{'='*50}\nFold {k+1}/5\n{'='*50}")
        tr = labels_df[labels_df["cv_fold"] != k]
        val = labels_df[labels_df["cv_fold"] == k]
        print(f"  Train: {len(tr)}, Val: {len(val)}")

        tr_ids = tr["BraTS21ID"].tolist()
        val_ids = val["BraTS21ID"].tolist()

        tr_ds = BrainMRIDataset(tr_ids, label_dict, f"{DATA_ROOT}/train", augment=True)
        val_ds = BrainMRIDataset(val_ids, label_dict, f"{DATA_ROOT}/train", augment=False)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        model = CrossSequenceTransformerModel().to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=COSINE_T0, T_mult=COSINE_T_MULT, eta_min=1e-6
        )

        best_val_auc = 0.0
        patience_counter = 0

        for epoch in range(N_EPOCHS):
            tr_loss, tr_auc = train_epoch(model, tr_loader, optimizer, criterion, DEVICE)
            val_loss, val_auc, _, _ = eval_epoch(model, val_loader, criterion, DEVICE)
            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:2d}/{N_EPOCHS}: tr_loss={tr_loss:.4f}, tr_auc={tr_auc:.4f} | "
                  f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f} | lr={current_lr:.2e}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(model.state_dict(), WORKSPACE / f"best_model_g011_fold{k}.pt")
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        print(f"  Fold {k+1} best val AUC: {best_val_auc:.4f}")
        fold_scores.append(best_val_auc)
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    mean_auc = float(np.mean(fold_scores))
    std_auc = float(np.std(fold_scores))
    print(f"\n{'='*70}")
    print(f"5-fold CV Results:")
    print(f"  Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Per-fold: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"{'='*70}")

    print("\nRetraining on all data for final submission...")
    all_train_ids = labels_df["BraTS21ID"].tolist()
    final_ds = BrainMRIDataset(all_train_ids, label_dict, f"{DATA_ROOT}/train", augment=True)
    final_loader = DataLoader(final_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    final_model = CrossSequenceTransformerModel().to(DEVICE)
    final_optimizer = torch.optim.AdamW(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    final_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        final_optimizer, T_0=COSINE_T0, T_mult=COSINE_T_MULT, eta_min=1e-6
    )
    final_criterion = LabelSmoothingBCELoss(smoothing=LABEL_SMOOTHING)

    for epoch in range(N_EPOCHS):
        tr_loss, tr_auc = train_epoch(final_model, final_loader, final_optimizer, final_criterion, DEVICE)
        final_scheduler.step(epoch)
        print(f"  Final epoch {epoch+1:2d}/{N_EPOCHS}: loss={tr_loss:.4f}, train_auc={tr_auc:.4f}")

    print("\nGenerating test predictions...")
    test_ds = BrainMRIDataset(test_ids, {}, TEST_DIR, is_test=True, augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    _, test_preds = predict(final_model, test_loader, DEVICE)
    test_preds = [float(p) for p in test_preds]

    print(f"Test predictions: mean={np.mean(test_preds):.4f}, std={np.std(test_preds):.4f}")

    submission = pd.DataFrame({
        "BraTS21ID": [int(pid) for pid in test_ids],
        "MGMT_value": test_preds
    })
    sub_path = WORKSPACE / f"submission_{EXP_ID}.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\nSaved submission to: {sub_path}")

    train_path = WORKSPACE / f"train_{EXP_ID}.py"
    src = os.path.abspath(__file__)
    dst = os.path.abspath(train_path)
    if src != dst:
        shutil.copy(src, dst)
    print(f"Saved train script to: {train_path}")

    result_summary = {
        "val_score": mean_auc,
        "val_std": std_auc,
        "fold_scores": fold_scores,
        "direction": "maximize",
        "exp_id": EXP_ID,
                "submission_path": str(sub_path),
        "train_path": str(train_path),
        "status": "complete",
        "posted_to_workshop": False,
        "result_post_id": None,
        "pid": None, "monitor_id": None,
        "stdout_path": None, "stderr_path": None,
        "item": {
            "id": EXP_ID,
            "axis": "schedule",
            "direction": "change",
            "value": "EfficientNet-B0-CosineWarmRestart-LongerCycles-LabelSmoothing",
        },
        "queue_claimed": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Champion architecture + AdamW + CosineWarmRestart(T_0=10,T_mult=2) + "
            "LabelSmoothing=0.05 + N_EPOCHS=25. T_0=10 vs champion T_0=5 — longer "
            "cosine cycles allow deeper descent before LR restart."
        ),
    }
    result_path = Path(__file__).parent / "outputs" / "result_latest.json"
    result_path.write_text(json.dumps(result_summary, indent=2, default=str))
    print(f"\nResult saved: val_score={mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"Completed: {datetime.now().isoformat()}")

    print("\n" + "=" * 60)
    print(json.dumps({
        "exp_id": EXP_ID,
        "model": "efficientnet_b0",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingWarmRestarts",
        "cosine_T0": COSINE_T0,
        "cosine_T_mult": COSINE_T_MULT,
        "label_smoothing": LABEL_SMOOTHING,
        "mean_cv_auc": mean_auc,
        "fold_scores": fold_scores,
    }, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
