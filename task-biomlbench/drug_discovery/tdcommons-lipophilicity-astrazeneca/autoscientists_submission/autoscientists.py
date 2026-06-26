"""
exp_beta_001: ChemBERTa-2 fine-tuning for lipophilicity prediction (v3).

Approach: Fine-tune ChemBERTa (seyonec/ChemBERTa-zinc-base-v1) via RobertaForMaskedLM
backbone to ensure proper embedding loading. CLS token + two-layer regression head.

Key design choices:
- Load via RobertaForMaskedLM to fix the word_embeddings.weight tie issue
- CLS token pooling (better for this pretrained model)
- Batch size 64 for speed
- LR 2e-5 with cosine decay
- Dropout 0.1 on head
- Early stopping patience=5, max epochs=20
- Data: Drug column (SMILES) from train.csv with cv_fold for 5-fold scaffold CV

Team: beta (SMILES-Transformer-ChemBERTa-MolFormer paradigm)
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, RobertaForMaskedLM, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import pathlib

# --- Paths ---
FOCUS_ROOT = Path(__file__).parent.parent
DATA_DIR = FOCUS_ROOT / "data"OUTPUT_DIR = FOCUS_ROOT

# --- Hyperparameters ---
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
MAX_LENGTH = 128
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.06
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
hp = {
    "model": MODEL_NAME,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "warmup_frac": WARMUP_FRAC,
    "num_epochs": NUM_EPOCHS,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "dropout": DROPOUT,
    "device": DEVICE,
}
print("=" * 60)
print(json.dumps(hp, indent=2))
print("=" * 60)

# ============================================================
class SmilesDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length=128):
        self.smiles = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.smiles[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


class ChemBERTaRegressor(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        # Use RobertaForMaskedLM to properly load ALL weights (including tied embeddings)
        mlm_model = RobertaForMaskedLM.from_pretrained(model_name, use_safetensors=True)
        self.encoder = mlm_model.roberta  # Extract the base encoder
        del mlm_model  # Free LM head memory
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Two-layer regression head
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        return self.head(cls_output).squeeze(-1)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        preds = model(input_ids, attention_mask)
        loss = nn.functional.mse_loss(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            preds = model(input_ids, attention_mask)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
    return mae, all_preds


def predict(model, smiles_list, tokenizer, device, max_length=128, batch_size=128):
    dataset = SmilesDataset(smiles_list, None, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids, attention_mask)
            all_preds.extend(preds.cpu().numpy().tolist())
    return np.array(all_preds)


def train_model(train_smiles, train_labels, val_smiles, val_labels, tokenizer, device):
    """Train a ChemBERTa regressor and return the best model and val MAE."""
    train_dataset = SmilesDataset(train_smiles, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = SmilesDataset(val_smiles, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                             num_workers=2, pin_memory=True)

    model = ChemBERTaRegressor(MODEL_NAME, dropout=DROPOUT).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_mae, _ = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch+1}/{NUM_EPOCHS}: train_loss={train_loss:.4f}, val_MAE={val_mae:.4f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, best_val_mae


# ============================================================
# Main
# ============================================================

print("Loading data...")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test_features.csv")

print(f"Train: {len(train_df)}, Test: {len(test_df)}")
print(f"Folds: {sorted(train_df['cv_fold'].unique())}")

print(f"Loading tokenizer: {MODEL_NAME}")
os.environ["HF_HOME"] = str(FOCUS_ROOT / ".cache" / "huggingface")
os.makedirs(FOCUS_ROOT / ".cache" / "huggingface", exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- 5-fold scaffold CV ---
fold_scores = []

for k in range(5):
    tr = train_df[train_df["cv_fold"] != k].reset_index(drop=True)
    val = train_df[train_df["cv_fold"] == k].reset_index(drop=True)

    print(f"\nFold {k}: train={len(tr)}, val={len(val)}")

    model, best_mae = train_model(
        tr["Drug"].tolist(), tr["Y"].values.tolist(),
        val["Drug"].tolist(), val["Y"].values.tolist(),
        tokenizer, DEVICE
    )

    fold_scores.append(best_mae)
    print(f"Fold {k} best val MAE: {best_mae:.4f}")
    model = model.cpu()
    del model
    torch.cuda.empty_cache()

mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)
print(f"\n{'='*60}")
print(f"Mean CV MAE: {mean_score:.4f} ± {std_score:.4f}")
print(f"Per fold: {[f'{s:.4f}' for s in fold_scores]}")
print(f"{'='*60}")

# --- Final model on full training data ---
print("\nTraining final model on full train data...")
full_smiles = train_df["Drug"].tolist()
full_labels = train_df["Y"].values.tolist()

# 90/10 split for early stopping during final training
np.random.seed(42)
n = len(full_smiles)
perm = np.random.permutation(n)
split_idx = int(0.9 * n)
train_idx = perm[:split_idx]
early_idx = perm[split_idx:]

final_model, _ = train_model(
    [full_smiles[i] for i in train_idx], [full_labels[i] for i in train_idx],
    [full_smiles[i] for i in early_idx], [full_labels[i] for i in early_idx],
    tokenizer, DEVICE
)

# Generate submission
print("\nGenerating test predictions...")
final_model = final_model.to(DEVICE)
test_preds = predict(final_model, test_df["Drug"].tolist(), tokenizer, DEVICE)

submission = pd.DataFrame({"id": test_df["id"], "Y": test_preds})
submission_path = f"{OUTPUT_DIR}/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")
print(f"Submission shape: {submission.shape}")
print(submission.head())

# Save train.py to output directory
shutil.copy(__file__, f"{OUTPUT_DIR}/train.py")
print(f"Saved train.py to {OUTPUT_DIR}/train.py")

# Save result summary
result_summary = {
    "val_score": float(mean_score),
    "direction": "minimize",
    "exp_id": "exp_beta_001",
    "submission_path": submission_path,
    "fold_scores": [float(s) for s in fold_scores],
    "std": float(std_score),
}
workspace = str(Path(__file__).parent / "outputs")
pathlib.Path(workspace).mkdir(parents=True, exist_ok=True)
(pathlib.Path(workspace) / "result_latest.json").write_text(
    json.dumps(result_summary, indent=2)
)
print(f"\nResult summary: {json.dumps(result_summary, indent=2)}")
print(f"\n[BIOMLBENCH] FINAL: Mean CV MAE = {mean_score:.4f}")
