"""
Experiment: exp_alpha_002_2p5d_effb4_k7_40ep
Team: team_alpha
Approach: 2.5D U-Net with EfficientNet-B4 backbone, k=7 slices, 40 epochs

Changes from champion (exp_beta_002_2p5d_efficientnetb4_k5_40ep, val=0.5495):
1. k=7 adjacent slices instead of k=5 (7-channel input: [-3,-2,-1,0,+1,+2,+3])
   -- More volumetric context should help Hausdorff consistency (team_alpha hypothesis)
2. All other hyperparameters unchanged: EfficientNet-B4, 40ep, lr=1e-4, CosineAnnealingLR T_max=40

Architecture:
- segmentation_models_pytorch Unet(encoder_name='efficientnet-b4', encoder_weights='imagenet',
  in_channels=7 for 7-slice stack, classes=3)
- Input: stack 7 adjacent slices as 7-channel tensor
- Normalize 16-bit PNG to [0,1] then scale/normalize with ImageNet mean/std (cycled across 7 channels)
- Target: 3-class binary masks (large_bowel, small_bowel, stomach)
- Loss: 0.4*BCEWithLogitsLoss + 0.6*DiceLoss (metric-aligned)
- AdamW lr=1e-4, weight_decay=1e-5, CosineAnnealingLR T_max=40 epochs
- Augmentation: HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate
- Batch size: 32, epochs: 40, mixed precision
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from scipy.spatial.distance import directed_hausdorff

# -- Paths
FOCUS_ROOT = Path(__file__).parent.parent
EXP_ID = "exp_alpha_002_2p5d_effb4_k7_40ep"
DATA_DIR = FOCUS_ROOT / "data"WORKSPACE = Path(__file__).parent / "outputs"
WORKSPACE.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = DATA_DIR / "train_for_modeling.csv"
VAL_CSV   = DATA_DIR / "val.csv"
TEST_CSV  = DATA_DIR / "test_split.csv"
TRAIN_IMG_DIR = DATA_DIR / "train"
TEST_IMG_DIR  = DATA_DIR / "test"

# -- Config
CFG = {
    "exp_id": EXP_ID,
    "encoder": "efficientnet-b4",
    "encoder_weights": "imagenet",
    "in_channels": 7,
    "num_classes": 3,
    "img_size": 256,
    "batch_size": 32,
    "epochs": 40,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "bce_weight": 0.4,
    "dice_weight": 0.6,
    "num_workers": 4,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "k_slices": 7,
}

print("=" * 60)
print(json.dumps(CFG, indent=2))
print("=" * 60)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

CLASSES = ["large_bowel", "small_bowel", "stomach"]

# ImageNet normalization cycled across 7 channels
_IMAGENET_MEAN_3 = [0.485, 0.456, 0.406]
_IMAGENET_STD_3  = [0.229, 0.224, 0.225]


def rle_decode(mask_rle, shape):
    if pd.isna(mask_rle) or str(mask_rle).strip() == "":
        return np.zeros(shape, dtype=np.uint8)
    s = str(mask_rle).split()
    starts = np.array(s[0::2], dtype=int) - 1
    lengths = np.array(s[1::2], dtype=int)
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order="F")


def rle_encode(mask):
    flat = mask.flatten(order="F")
    if flat.sum() == 0:
        return ""
    changes = np.diff(np.concatenate([[0], flat, [0]]))
    starts = np.where(changes == 1)[0] + 1
    ends   = np.where(changes == -1)[0] + 1
    lengths = ends - starts
    rle = []
    for s, l in zip(starts, lengths):
        rle.extend([str(s), str(l)])
    return " ".join(rle)


def load_16bit(path):
    img = np.array(Image.open(path), dtype=np.float32)
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn) * 255.0
    else:
        img = np.zeros_like(img)
    return img


def parse_slice_path(id_str):
    parts = id_str.split("_")
    case = parts[0]
    day  = parts[1]
    slice_num = int(parts[-1])
    return case, day, slice_num


def get_image_path(case, day, slice_num, img_dir):
    scan_dir = img_dir / case / f"{case}_{day}" / "scans"
    if not scan_dir.exists():
        return None
    pattern = f"slice_{slice_num:04d}_"
    for f in scan_dir.iterdir():
        if f.name.startswith(pattern):
            return f
    return None


def get_image_hw(case, day, img_dir):
    scan_dir = img_dir / case / f"{case}_{day}" / "scans"
    if not scan_dir.exists():
        return 266, 266
    files = list(scan_dir.iterdir())
    if not files:
        return 266, 266
    name = files[0].stem
    parts = name.split("_")
    try:
        w, h = int(parts[2]), int(parts[3])
        return h, w
    except:
        return 266, 266


class GITractDataset(Dataset):
    def __init__(self, df, img_dir, img_size=256, transform=None, is_test=False, k=7):
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        self.is_test = is_test
        self.k = k

        mask_col = "predicted" if "predicted" in df.columns else "segmentation"
        self.slice_ids = df["id"].unique()
        self.df = df.set_index(["id", "class"])
        self.mask_col = mask_col

        self.records = []
        for sid in self.slice_ids:
            case, day, slice_num = parse_slice_path(sid)
            self.records.append((sid, case, day, slice_num))

        from collections import defaultdict
        case_day_slices = defaultdict(list)
        for sid, case, day, snum in self.records:
            case_day_slices[(case, day)].append(snum)
        self.case_day_slices = {ky: sorted(v) for ky, v in case_day_slices.items()}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        sid, case, day, slice_num = self.records[idx]

        h, w = get_image_hw(case, day, self.img_dir)

        slices_in_vol = self.case_day_slices[(case, day)]
        curr_idx = slices_in_vol.index(slice_num) if slice_num in slices_in_vol else 0

        # k=7: offsets [-3, -2, -1, 0, +1, +2, +3]
        half = self.k // 2
        slice_channels = []
        for offset in range(-half, half + 1):
            target_idx = max(0, min(len(slices_in_vol) - 1, curr_idx + offset))
            target_snum = slices_in_vol[target_idx]
            path = get_image_path(case, day, target_snum, self.img_dir)
            if path is None or not path.exists():
                slice_channels.append(np.zeros((h, w), dtype=np.float32))
            else:
                slice_channels.append(load_16bit(path))

        image = np.stack(slice_channels, axis=-1)  # (H, W, k)

        if self.is_test:
            mask = np.zeros((h, w, 3), dtype=np.float32)
        else:
            masks = []
            for cls in CLASSES:
                try:
                    rle = self.df.loc[(sid, cls), self.mask_col]
                except KeyError:
                    rle = None
                masks.append(rle_decode(rle, (h, w)).astype(np.float32))
            mask = np.stack(masks, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask)
            mask = mask.permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask  = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        return {"image": image, "mask": mask, "id": sid}


def get_train_transforms(img_size, k=7):
    mean = [_IMAGENET_MEAN_3[i % 3] for i in range(k)]
    std  = [_IMAGENET_STD_3[i % 3]  for i in range(k)]
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_val_transforms(img_size, k=7):
    mean = [_IMAGENET_MEAN_3[i % 3] for i in range(k)]
    std  = [_IMAGENET_STD_3[i % 3]  for i in range(k)]
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ])


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        B, C = probs.shape[:2]
        probs_flat = probs.view(B, C, -1)
        tgt_flat   = targets.view(B, C, -1)
        inter = (probs_flat * tgt_flat).sum(-1)
        denom = probs_flat.sum(-1) + tgt_flat.sum(-1)
        dice  = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_w=0.4, dice_w=0.6):
        super().__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets) + self.dice_w * self.dice(logits, targets)


UNIT_CUBE_DIAGONAL = np.sqrt(3)

def dice_2d(pred, true):
    inter = int((pred & true).sum())
    denom = int(pred.sum()) + int(true.sum())
    return float(2 * inter / denom) if denom else 0.0


def hausdorff_3d(pred_vol, true_vol):
    if pred_vol.sum() == 0 and true_vol.sum() == 0:
        return np.nan
    if pred_vol.sum() == 0 or true_vol.sum() == 0:
        return 1.0
    if pred_vol.sum() > 10 * true_vol.sum():
        return 1.0
    tc = np.argwhere(true_vol) / np.array(true_vol.shape, dtype=float)
    pc = np.argwhere(pred_vol) / np.array(pred_vol.shape, dtype=float)
    h = max(directed_hausdorff(tc, pc)[0], directed_hausdorff(pc, tc)[0])
    return h / UNIT_CUBE_DIAGONAL


def compute_metric(pred_df, gt_df, img_dir):
    gt_df = gt_df.copy()
    pred_df = pred_df.copy()

    merged = gt_df.rename(columns={"predicted": "gt_rle"}).merge(
        pred_df.rename(columns={"predicted": "pred_rle"}),
        on=["id", "class"], how="left"
    )
    merged["pred_rle"] = merged["pred_rle"].fillna("")

    dice_scores = []
    for _, row in merged.iterrows():
        sid = row["id"]
        case, day, snum = parse_slice_path(sid)
        h, w = get_image_hw(case, day, img_dir)
        gt_mask   = rle_decode(row["gt_rle"], (h, w)).astype(bool)
        pred_mask = rle_decode(row["pred_rle"], (h, w)).astype(bool)
        dice_scores.append(dice_2d(pred_mask, gt_mask))

    def get_case_day(sid):
        c, d, _ = parse_slice_path(sid)
        return f"{c}_{d}"

    merged["case_day"] = merged["id"].apply(get_case_day)
    merged["slice_num"] = merged["id"].apply(lambda x: parse_slice_path(x)[2])

    hd_scores = []
    for cd, grp in merged.groupby("case_day"):
        slices_sorted = sorted(grp["slice_num"].unique())
        sample_id = grp["id"].iloc[0]
        case, day, _ = parse_slice_path(sample_id)
        h, w = get_image_hw(case, day, img_dir)

        pred_vol = []
        true_vol = []
        for sn in slices_sorted:
            sg = grp[grp["slice_num"] == sn]
            pred_slice = np.zeros((h, w), dtype=bool)
            true_slice = np.zeros((h, w), dtype=bool)
            for _, row in sg.iterrows():
                pred_slice |= rle_decode(row["pred_rle"], (h, w)).astype(bool)
                true_slice |= rle_decode(row["gt_rle"],   (h, w)).astype(bool)
            pred_vol.append(pred_slice)
            true_vol.append(true_slice)

        pred_vol = np.stack(pred_vol, axis=0)
        true_vol = np.stack(true_vol, axis=0)
        hd_scores.append(hausdorff_3d(pred_vol, true_vol))

    dice_mean = float(np.mean(dice_scores))
    hd_mean   = float(np.nanmean(hd_scores)) if any(not np.isnan(x) for x in hd_scores) else 1.0
    score = 0.4 * dice_mean + 0.6 * (1 - hd_mean)
    print(f"Dice (mean): {dice_mean:.4f}, Hausdorff (mean): {hd_mean:.4f}, Score: {score:.4f}")
    return score


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        imgs  = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs)
            loss   = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def predict_loader(model, loader, device, img_size, data_img_dir=None):
    if data_img_dir is None:
        data_img_dir = TRAIN_IMG_DIR
    model.eval()
    results = []
    for batch in loader:
        imgs    = batch["image"].to(device)
        ids     = batch["id"]
        with torch.cuda.amp.autocast():
            logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().float()
        pred_masks = (probs > 0.5).numpy()

        for b_idx in range(len(ids)):
            sid = ids[b_idx]
            case, day, snum = parse_slice_path(sid)
            orig_h, orig_w = get_image_hw(case, day, data_img_dir)
            for cls_idx, cls_name in enumerate(CLASSES):
                pred = pred_masks[b_idx, cls_idx]
                pred_full = pred.astype(np.uint8)
                if pred.shape != (orig_h, orig_w):
                    from PIL import Image as PILImage
                    pred_img = PILImage.fromarray(pred_full)
                    pred_img = pred_img.resize((orig_w, orig_h), PILImage.NEAREST)
                    pred_full = np.array(pred_img)
                rle = rle_encode(pred_full)
                results.append({"id": sid, "class": cls_name, "predicted": rle})
    return results


def main():
    device = CFG["device"]
    print(f"Device: {device}")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")

    k = CFG["k_slices"]

    train_transforms = get_train_transforms(CFG["img_size"], k=k)
    val_transforms   = get_val_transforms(CFG["img_size"], k=k)

    train_ds = GITractDataset(train_df, TRAIN_IMG_DIR, CFG["img_size"], train_transforms, is_test=False, k=k)
    val_ds   = GITractDataset(val_df,   TRAIN_IMG_DIR, CFG["img_size"], val_transforms,   is_test=False, k=k)
    test_ds  = GITractDataset(test_df,  TEST_IMG_DIR,  CFG["img_size"], val_transforms,   is_test=True,  k=k)

    print(f"Train slices: {len(train_ds)}, Val slices: {len(val_ds)}, Test slices: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                              num_workers=CFG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False,
                              num_workers=CFG["num_workers"], pin_memory=True)

    # EfficientNet-B4 with 7-channel input
    model = smp.Unet(
        encoder_name=CFG["encoder"],
        encoder_weights=CFG["encoder_weights"],
        in_channels=CFG["in_channels"],
        classes=CFG["num_classes"],
        activation=None,
    ).to(device)

    criterion = CombinedLoss(bce_w=CFG["bce_weight"], dice_w=CFG["dice_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    scaler    = torch.cuda.amp.GradScaler()

    best_score  = -1.0
    best_epoch  = -1
    best_state  = None

    train_losses = []
    val_scores   = []

    print("\nStarting training...")
    for epoch in range(1, CFG["epochs"] + 1):
        t0 = datetime.utcnow()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        scheduler.step()
        train_losses.append(train_loss)

        model.eval()
        dice_sum = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs  = batch["image"].to(device)
                masks = batch["mask"].to(device)
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                inter = (preds * masks).sum(dim=(2, 3))
                denom = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                d = (2 * inter / (denom + 1e-6))
                dice_sum += d.mean().item() * imgs.size(0)
                n += imgs.size(0)
        val_dice = dice_sum / n
        val_scores.append(val_dice)

        elapsed = (datetime.utcnow() - t0).total_seconds()
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d}/{CFG['epochs']} | loss={train_loss:.4f} | "
              f"val_dice={val_dice:.4f} | lr={lr_now:.2e} | {elapsed:.0f}s")

        if val_dice > best_score:
            best_score = val_dice
            best_epoch = epoch
            best_state = {ky: v.cpu().clone() for ky, v in model.state_dict().items()}
            print(f"  *** New best val_dice={best_score:.4f} at epoch {epoch} ***")

    print("\n[TRAINING DYNAMICS]")
    print(f"  Total epochs: {CFG['epochs']}")
    print(f"  Best epoch: {best_epoch} | Best val_dice: {best_score:.4f}")
    if len(val_scores) >= 2:
        early = np.mean(val_scores[:4]) if len(val_scores) >= 4 else val_scores[0]
        late  = np.mean(val_scores[-4:]) if len(val_scores) >= 4 else val_scores[-1]
        print(f"  Val dice (first 4 epochs avg): {early:.4f}")
        print(f"  Val dice (last 4 epochs avg):  {late:.4f}")
        if late > early * 1.01:
            print("  Dynamics: still improving at end -- consider more epochs")
        elif late > early:
            print("  Dynamics: still improving but converging")
        else:
            print("  Dynamics: plateau reached before final epoch")
    print(f"  Train loss trajectory: first={train_losses[0]:.4f}, last={train_losses[-1]:.4f}")

    model.load_state_dict(best_state)
    model.to(device)

    print("\nRunning full val prediction for metric computation...")
    val_preds = predict_loader(model, val_loader, device, CFG["img_size"], TRAIN_IMG_DIR)
    val_pred_df = pd.DataFrame(val_preds)
    print("Computing val metric (Dice + Hausdorff)...")
    val_score = compute_metric(val_pred_df, val_df, TRAIN_IMG_DIR)
    print(f"\nFinal Val Score: {val_score:.4f}")

    print("\nGenerating test predictions...")
    test_preds = predict_loader(model, test_loader, device, CFG["img_size"], TEST_IMG_DIR)
    submission = pd.DataFrame(test_preds)

    test_ids = set(zip(test_df["id"], test_df["class"]))
    sub_ids  = set(zip(submission["id"], submission["class"]))
    missing  = test_ids - sub_ids
    if missing:
        print(f"WARNING: {len(missing)} missing rows, filling with empty predictions")
        extra_rows = [{"id": i, "class": c, "predicted": ""} for i, c in missing]
        submission = pd.concat([submission, pd.DataFrame(extra_rows)], ignore_index=True)

    submission = submission[["id", "class", "predicted"]]
    print(f"Submission shape: {submission.shape}")

    sub_path      = WORKSPACE / f"submission_{EXP_ID}.csv"
    train_py_path = WORKSPACE / f"train_{EXP_ID}.py"

    submission.to_csv(sub_path, index=False)
    print(f"Saved submission -> {sub_path}")

    shutil.copy(__file__, train_py_path)
    print(f"Saved train.py   -> {train_py_path}")

    submission.to_csv(WORKSPACE / "submission.csv", index=False)
    shutil.copy(__file__, WORKSPACE / "train.py")

    result = {
        "val_score": val_score,
        "direction": "maximize",
        "exp_id": EXP_ID,
                "submission_path": str(sub_path),
        "train_path": str(train_py_path),
        "status": "complete",
        "posted_to_workshop": False,
        "result_post_id": None,
        "pid": None,
        "monitor_id": None,
        "stdout_path": None,
        "stderr_path": None,
        "item": None,
        "queue_claimed": True,
        "timestamp": datetime.utcnow().isoformat(),
        "best_val_dice": best_score,
        "best_epoch": best_epoch,
        "train_losses": train_losses,
        "val_scores": val_scores,
    }
    result_path = WORKSPACE.parent / "result_latest.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved result_latest.json -> {result_path}")
    print(f"\n{'='*60}")
    print(f"FINAL VAL SCORE: {val_score:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
