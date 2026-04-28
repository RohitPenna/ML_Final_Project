"""
BERT-based Fake News Classifier for the LIAR Dataset
=====================================================
Based on: "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection
          Wang (2017), ACL

Label mapping (6-class → 5-class):
  pants-fire → false  (merged as per user request)
  false      → false
  barely-true → barely-true
  half-true   → half-true
  mostly-true → mostly-true
  true        → true

LIAR dataset TSV columns:
  0:  ID
  1:  label
  2:  statement
  3:  subject(s)
  4:  speaker
  5:  speaker's job title
  6:  state info
  7:  party affiliation
  8:  barely_true_counts
  9:  false_counts
  10: half_true_counts
  11: mostly_true_counts
  12: pants_on_fire_counts
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "Data"          # Directory containing train.tsv, valid.tsv, test.tsv
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
SEED = 42
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------

# Original 6 labels → 5 labels after merging pants-fire into false
LABEL_MAP = {
    "pants-fire": "false",   # <-- merge step
    "false":      "false",
    "barely-true": "false",
    "half-true":   "true",
    "mostly-true": "true",
    "true":        "true",
}

CLASSES = ["false", "true"]
LABEL2ID = {label: idx for idx, label in enumerate(CLASSES)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(CLASSES)

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

COLUMNS = [
    "id", "label", "statement", "subject", "speaker",
    "job", "state", "party",
    "barely_true_counts", "false_counts", "half_true_counts",
    "mostly_true_counts", "pants_fire_counts", "context"
]


def load_liar_tsv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t", header=None, names=COLUMNS, on_bad_lines='skip')

    # Normalise + merge pants-fire → false
    df["label"] = df["label"].str.strip().str.lower().map(LABEL_MAP)

    unknown = df["label"].isna()
    if unknown.any():
        print(f"  Warning: {unknown.sum()} rows with unrecognised labels dropped.")
        df = df[~unknown]

    df["label_id"] = df["label"].map(LABEL2ID)
    return df


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class LIARDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer, max_len: int):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        statement = str(self.data.loc[idx, "statement"])
        label = int(self.data.loc[idx, "label_id"])

        encoding = self.tokenizer(
            statement,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training & evaluation helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        loss = outputs.loss
        preds = outputs.logits.argmax(dim=-1)

        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, save_path: str = "training_curves.png"):
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Validation")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Validation")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved → {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path: str = "confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASSES, yticklabels=CLASSES,
    )
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Load data -------------------------------------------------------
    print("\n[1/5] Loading data...")
    train_df = load_liar_tsv(os.path.join(DATA_DIR, "train.tsv"))
    val_df   = load_liar_tsv(os.path.join(DATA_DIR, "valid.tsv"))
    test_df  = load_liar_tsv(os.path.join(DATA_DIR, "test.tsv"))

    print(f"  Train : {len(train_df):,} samples")
    print(f"  Valid : {len(val_df):,} samples")
    print(f"  Test  : {len(test_df):,} samples")
    print(f"\n  Label distribution (train):\n{train_df['label'].value_counts().to_string()}")

    # ---- Tokenizer -------------------------------------------------------
    print(f"\n[2/5] Loading tokenizer: {MODEL_NAME}")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # ---- Datasets & Loaders ----------------------------------------------
    train_ds = LIARDataset(train_df, tokenizer, MAX_LEN)
    val_ds   = LIARDataset(val_df,   tokenizer, MAX_LEN)
    test_ds  = LIARDataset(test_df,  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ---- Model -----------------------------------------------------------
    print(f"\n[3/5] Loading model: {MODEL_NAME} ({NUM_LABELS} labels)")
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(DEVICE)

    # ---- Optimizer & Scheduler -------------------------------------------
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ---- Training loop ---------------------------------------------------
    print(f"\n[4/5] Training for {EPOCHS} epochs...")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained("best_bert_liar")
            tokenizer.save_pretrained("best_bert_liar")
            print(f"    ✓ New best model saved (val_acc={val_acc:.4f})")

    plot_training_curves(history)

    # ---- Test evaluation -------------------------------------------------
    print("\n[5/5] Evaluating best model on test set...")
    model = BertForSequenceClassification.from_pretrained("best_bert_liar")
    model.to(DEVICE)

    _, test_acc, test_preds, test_labels = evaluate(model, test_loader)

    print(f"\n  Test Accuracy: {test_acc:.4f}")
    print("\n  Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=CLASSES))

    plot_confusion_matrix(test_labels, test_preds)
    print("\nDone!")


if __name__ == "__main__":
    main()