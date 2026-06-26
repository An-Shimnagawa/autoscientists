"""
Experiment: exp_beta_004 — Dual-stream EfficientNet-B3 with center-48 crop + deep fusion head
Approach: Combines alpha_003 (B3 encoder) with zeta_002 (center-48 crop, deep fusion).
  - Stream A: full 96x96 image resized to 224 (context stream, EfficientNet-B3)
  - Stream B: center 48x48 region upsampled to 224 (label-region stream, EfficientNet-B3)
    Note: center crop offset=24 captures slightly more spatial context than 32x32 (offset=32)
  Deep fusion head: Linear(3072->512)->BN->ReLU->Dropout(0.3)->Linear(512->128)->BN->ReLU->Dropout(0.2)->Linear(128->1)
  MixUp alpha=0.2 (same as alpha_003)
  25 epochs, LR=1e-4, batch_size=64
Rationale:
  alpha_003 (B3, center-32, simple head): val_auc=0.9968
  zeta_002 (B0, center-48, deep head): val_auc=0.9959
  Combining B3 encoder + center-48 + deep head may push past 0.9968.
  Delta: center-48 showed +0.0034 vs champion in zeta_002; B3 adds more capacity.
Team: beta
Agent: biomlbgle_hist_2_gpu2
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import timm
from timm.scheduler import CosineLRScheduler

# ========== CONFIGURATION ==========
FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"AGENT_WORKSPACE = Path(__file__).parent / "outputs"
CACHE_DIR = FOCUS_ROOT / ".cache"
HF_HOME = str(CACHE_DIR / "huggingface")
TORCH_HOME = str(CACHE_DIR / "torch")

os.environ["HF_HOME"] = HF_HOME
os.environ["TORCH_HOME"] = TORCH_HOME
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[CONFIG] Device: {DEVICE}")
print(f"[CONFIG] CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[CONFIG] GPU: {torch.cuda.get_device_name(0)}")

MODEL_NAME = "efficientnet_b3"
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-4
WARMUP_EPOCHS = 3
WEIGHT_DECAY = 0.01
DROPOUT_1 = 0.3
DROPOUT_2 = 0.2
VAL_FRACTION = 0.2
RANDOM_SEED = 77
EXP_ID = "exp_beta_004"

# EfficientNet-B3 output features: 1536
EFFNET_FEATURES = 1536
# Center crop: 48x48 (offset=24 from edge) — captures more context than 32x32
CENTER_CROP = 48
MIXUP_ALPHA = 0.2

CONFIG = {
    "exp_id": EXP_ID,
    "model": MODEL_NAME,
    "architecture": "dual_stream_B3_center48_deep_fusion",
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "lr": LR,
    "warmup_epochs": WARMUP_EPOCHS,
    "weight_decay": WEIGHT_DECAY,
    "dropout_1": DROPOUT_1,
    "dropout_2": DROPOUT_2,
    "val_fraction": VAL_FRACTION,
    "random_seed": RANDOM_SEED,
    "center_crop": CENTER_CROP,
    "stream_a": "full_96x96_resize_224",
    "stream_b": f"center_{CENTER_CROP}x{CENTER_CROP}_upsample_224",
    "fusion": "deep_3layer: 3072->512->128->1 with BN+ReLU+Dropout",
    "mixup_alpha": MIXUP_ALPHA,
    "note": "B3 + center-48 + deep fusion (combines alpha_003 and zeta_002 innovations)",
}
print("=" * 60)
print(json.dumps(CONFIG, indent=2))
print("=" * 60)

# ========== DATA PATHS ==========
TRAIN_LABELS_CSV = DATA_DIR / "train_labels.csv"
SAMPLE_SUB_CSV = DATA_DIR / "sample_submission.csv"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "train"   # PCam: test images live in data/train/

AGENT_WORKSPACE.mkdir(parents=True, exist_ok=True)


# ========== SEED ==========
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)


# ========== TRANSFORMS ==========
train_transform_context = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation(90)], p=0.5),
    T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_transform_center = T.Compose([
    T.CenterCrop(CENTER_CROP),
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation(90)], p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform_context = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform_center = T.Compose([
    T.CenterCrop(CENTER_CROP),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ========== DATASET ==========
class PCamDataset(Dataset):
    def __init__(self, df, img_dir, transform_context, transform_center, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform_context = transform_context
        self.transform_center = transform_center
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        img_path = self.img_dir / f"{img_id}.tif"
        img = Image.open(img_path).convert("RGB")
        img_context = self.transform_context(img)
        img_center = self.transform_center(img)
        if self.is_test:
            return img_context, img_center, img_id
        else:
            label = float(row["label"])
            return img_context, img_center, label


# ========== MODEL ==========
class DualStreamB3DeepFusion(nn.Module):
    def __init__(self, model_name, features, dropout_1, dropout_2):
        super().__init__()
        self.context_stream = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.center_stream = timm.create_model(model_name, pretrained=True, num_classes=0)

        # Deep fusion: 3072 -> 512 -> 128 -> 1
        self.fusion = nn.Sequential(
            nn.Linear(features * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_1),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_2),
            nn.Linear(128, 1),
        )

    def forward(self, x_context, x_center):
        feat_context = self.context_stream(x_context)
        feat_center = self.center_stream(x_center)
        combined = torch.cat([feat_context, feat_center], dim=1)
        return self.fusion(combined).squeeze(1)


# ========== MIXUP ==========
def mixup_data(x_ctx, x_ctr, y, alpha=0.2):
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size = x_ctx.size(0)
    index = torch.randperm(batch_size).to(x_ctx.device)
    mixed_ctx = lam * x_ctx + (1 - lam) * x_ctx[index, :]
    mixed_ctr = lam * x_ctr + (1 - lam) * x_ctr[index, :]
    y_a, y_b = y, y[index]
    return mixed_ctx, mixed_ctr, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ========== VALIDATION ==========
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            x_ctx, x_ctr, labels = batch
            x_ctx = x_ctx.to(device)
            x_ctr = x_ctr.to(device)
            labels = labels.to(device).float()
            with torch.amp.autocast('cuda'):
                logits = model(x_ctx, x_ctr)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = torch.sigmoid(logits).cpu().float().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(all_labels)
    auc = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc


# ========== TEST PREDICTION WITH 4-ROTATION TTA ==========
def predict_test_tta(model, sample_sub, test_dir, device, batch_size):
    """4-rotation TTA: average over 0, 90, 180, 270 degree rotations."""
    all_preds_tta = []
    rotation_angles = [0, 90, 180, 270]

    for angle in rotation_angles:
        if angle == 0:
            tta_transform_context = val_transform_context
            tta_transform_center = val_transform_center
        else:
            tta_transform_context = T.Compose([
                T.Resize((224, 224)),
                T.RandomRotation((angle, angle)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tta_transform_center = T.Compose([
                T.CenterCrop(CENTER_CROP),
                T.Resize((224, 224)),
                T.RandomRotation((angle, angle)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        dataset = PCamDataset(
            sample_sub, test_dir,
            tta_transform_context, tta_transform_center,
            is_test=True
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)

        model.eval()
        preds_this_angle = []
        ids_this_angle = []
        with torch.no_grad():
            for batch in loader:
                x_ctx, x_ctr, ids = batch
                x_ctx = x_ctx.to(device)
                x_ctr = x_ctr.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(x_ctx, x_ctr)
                preds = torch.sigmoid(logits).cpu().float().numpy()
                preds_this_angle.extend(preds.tolist())
                ids_this_angle.extend(list(ids))

        all_preds_tta.append(preds_this_angle)
        print(f"  TTA angle={angle}°: done ({len(preds_this_angle)} predictions)")

    # Average TTA predictions
    avg_preds = np.mean(all_preds_tta, axis=0).tolist()
    return ids_this_angle, avg_preds


# ========== MAIN ==========
def main():
    print(f"\n[{datetime.now(timezone.utc).isoformat()}] Starting training...")

    labels_df = pd.read_csv(TRAIN_LABELS_CSV)
    print(f"[DATA] Total training samples: {len(labels_df)}")
    print(f"[DATA] Positive rate: {labels_df['label'].mean():.3f}")

    train_df, val_df = train_test_split(
        labels_df, test_size=VAL_FRACTION, random_state=RANDOM_SEED, stratify=labels_df["label"]
    )
    print(f"[DATA] Train: {len(train_df)}, Val: {len(val_df)}")

    train_dataset = PCamDataset(train_df, TRAIN_DIR, train_transform_context, train_transform_center)
    val_dataset = PCamDataset(val_df, TRAIN_DIR, val_transform_context, val_transform_center)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, prefetch_factor=2)

    model = DualStreamB3DeepFusion(MODEL_NAME, EFFNET_FEATURES, DROPOUT_1, DROPOUT_2)
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Dual-stream EfficientNet-B3 (deep fusion, center-{CENTER_CROP}) on {DEVICE}, {total_params/1e6:.1f}M params")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')

    num_batches = len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=EPOCHS * num_batches,
        warmup_t=WARMUP_EPOCHS * num_batches,
        warmup_lr_init=LR / 100,
        lr_min=LR / 100,
    )

    best_auc = 0.0
    best_epoch = 0
    model_path = AGENT_WORKSPACE / "best_model.pth"
    train_losses = []
    val_aucs = []

    print(f"\n[TRAIN] Starting {EPOCHS} epochs (warmup={WARMUP_EPOCHS})...")
    step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        n = 0
        for batch in train_loader:
            x_ctx, x_ctr, labels = batch
            x_ctx = x_ctx.to(DEVICE)
            x_ctr = x_ctr.to(DEVICE)
            labels = labels.to(DEVICE).float()

            optimizer.zero_grad()
            x_ctx_m, x_ctr_m, y_a, y_b, lam = mixup_data(x_ctx, x_ctr, labels, MIXUP_ALPHA)

            with torch.amp.autocast('cuda'):
                logits = model(x_ctx_m, x_ctr_m)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(step)
            step += 1

            total_loss += loss.item() * len(labels)
            n += len(labels)

        train_loss = total_loss / n
        train_losses.append(train_loss)

        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)
        val_aucs.append(val_auc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"[EPOCH {epoch:02d}/{EPOCHS}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  lr={current_lr:.2e}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print(f"  -> New best! Saved checkpoint (epoch {epoch}, auc={best_auc:.4f})")

    print(f"\n[BEST] Epoch {best_epoch}, Val AUC = {best_auc:.4f}")

    still_decreasing = False
    if len(train_losses) >= 2:
        idx_80 = int(0.8 * len(train_losses))
        loss_at_80pct = train_losses[idx_80]
        loss_final = train_losses[-1]
        still_decreasing = bool(loss_final < loss_at_80pct * 0.99)

    print(f"\n[PREDICT] 4-rotation TTA inference on test set...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))

    sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
    test_ids, test_preds = predict_test_tta(model, sample_sub, TEST_DIR, DEVICE, BATCH_SIZE)

    pred_map = dict(zip(test_ids, test_preds))
    submission = pd.DataFrame({
        "id": sample_sub["id"],
        "label": [pred_map[i] for i in sample_sub["id"]]
    })

    submission_path = AGENT_WORKSPACE / "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"[SUBMISSION] Saved to {submission_path} ({len(submission)} rows)")
    print(f"[SUBMISSION] label stats: min={submission['label'].min():.4f} max={submission['label'].max():.4f} mean={submission['label'].mean():.4f}")

    stamped_sub = AGENT_WORKSPACE / f"submission_{EXP_ID}.csv"
    stamped_train = AGENT_WORKSPACE / f"train_{EXP_ID}.py"
    shutil.copy(submission_path, stamped_sub)
    if Path(__file__).resolve() != stamped_train.resolve():
        shutil.copy(__file__, stamped_train)
    print(f"[ISOLATION] submission -> {stamped_sub}")
    print(f"[ISOLATION] train      -> {stamped_train}")

    hyperparameters = {
        "model": MODEL_NAME,
        "architecture": "dual_stream_B3_center48_deep_fusion",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "warmup_epochs": WARMUP_EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "dropout_1": DROPOUT_1,
        "dropout_2": DROPOUT_2,
        "val_fraction": VAL_FRACTION,
        "random_seed": RANDOM_SEED,
        "center_crop": CENTER_CROP,
        "mixup_alpha": MIXUP_ALPHA,
        "stream_a": "full_96x96_resize_224",
        "stream_b": f"center_{CENTER_CROP}x{CENTER_CROP}_upsample_224",
        "fusion": "3layer_deep: 3072->512->128->1 BN+ReLU+Dropout",
        "tta": "4_rotation_0_90_180_270",
        "augmentation": "HFlip+VFlip+Rotation90+Affine+ColorJitter+MixUp",
        "scheduler": "CosineLRScheduler_with_warmup",
        "optimizer": "AdamW",
    }

    result = {
        "val_score": best_auc,
        "direction": "maximize",
        "exp_id": EXP_ID,
                "submission_path": str(stamped_sub),
        "train_path": str(stamped_train),
        "status": "complete",
        "posted_to_workshop": False,
        "result_post_id": None,
        "pid": None,
        "monitor_id": None,
        "queue_claimed": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "item": {
            "id": EXP_ID,
            "axis": "efficientnet_b3_dual_stream_center48",
            "direction": "maximize",
            "value": best_auc,
        },
        "description": f"B3 dual-stream center-{CENTER_CROP} deep-fusion, 25 epochs, MixUp, 4-TTA. Best val_auc={best_auc:.4f} at epoch {best_epoch}",
        "train_dynamics": {
            "train_losses": train_losses,
            "val_aucs": val_aucs,
            "best_epoch": best_epoch,
            "best_auc": best_auc,
            "still_decreasing_at_end": still_decreasing,
        },
        "hyperparameters": hyperparameters,
    }

    result_path = Path(__file__).parent / "outputs" / "result_latest.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"[RESULT] Written to {result_path}")

    print("\n" + "=" * 60)
    print(json.dumps(hyperparameters, indent=2))
    print("=" * 60)
    print(f"\n[FINAL] EXP_ID={EXP_ID}  val_auc={best_auc:.4f}  best_epoch={best_epoch}")

    return best_auc, best_epoch


if __name__ == "__main__":
    main()
