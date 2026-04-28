"""
LightGBM classifier using frozen BERT mean-pooled embeddings + speaker metadata features.

Pipeline:
  1. Load LIAR train / val / test TSV files
  2. Binarize 6-class labels  (true/mostly-true/half-true -> 0, barely-true/false/pants-fire -> 1)
  3. Extract 768-dim mean-pooled embeddings from bert-base-uncased (frozen, batch inference)
  4. Build speaker metadata features:
       - Numeric : false_count, barely_true_count, pants_fire_count,
                   mostly_true_count, half_true_count  (raw credibility history)
       - Encoded : party, state, speaker_job  (label-encoded, fit on train only)
  5. Concatenate BERT embeddings + metadata → (N, 776) feature matrix
  6. Train LightGBM with early stopping on the validation set
  7. Evaluate on the test set
  8. Save results to models/light_gbm/results.json
  9. Save model to models/light_gbm/lgbm_model.pkl
"""

import json
import os
import time
import tracemalloc
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertTokenizer

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR   = os.path.join(BASE_DIR, "Data")
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(OUT_DIR, "lgbm_model.pkl")
RESULTS_PATH = os.path.join(OUT_DIR, "results.json")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BERT_MODEL   = "bert-base-uncased"
BATCH_SIZE   = 32
MAX_LENGTH   = 128
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

LGBM_PARAMS = dict(
    num_leaves        = 31,        # reduced from 63 — less overfitting on small data
    learning_rate     = 0.02,      # slower convergence, more room to improve
    n_estimators      = 2000,      # high ceiling; early stopping will find the sweet spot
    min_child_samples = 30,        # stronger regularization
    subsample         = 0.8,       # row subsampling per tree
    colsample_bytree  = 0.8,       # feature subsampling per tree
    reg_alpha         = 0.1,       # L1 regularization
    reg_lambda        = 0.1,       # L2 regularization
    class_weight      = "balanced",# compensate for class imbalance
    objective         = "binary",
    metric            = "binary_logloss",
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

TSV_COLS = [
    "id", "label", "statement", "subjects", "speaker", "speaker_job",
    "state", "party", "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count", "context",
]

# Map 6-class -> binary
FAKE_LABELS = {"barely-true", "false", "pants-fire"}

def binarize(label: str) -> int:
    return 1 if label.strip() in FAKE_LABELS else 0


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_split(filename: str) -> pd.DataFrame:
    df = pd.read_csv(
        os.path.join(DATA_DIR, filename),
        sep="\t", header=None, names=TSV_COLS,
    )
    df["binary_label"] = df["label"].apply(binarize)
    df["statement"]    = df["statement"].fillna("").astype(str)
    return df


print("Loading data...")
train_df = load_split("train.tsv")
val_df   = load_split("valid.tsv")
test_df  = load_split("test.tsv")
print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")


# ---------------------------------------------------------------------------
# 2. Load frozen BERT
# ---------------------------------------------------------------------------
print(f"\nLoading {BERT_MODEL} on {DEVICE}...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
bert      = BertModel.from_pretrained(BERT_MODEL).to(DEVICE)
bert.eval()
for param in bert.parameters():
    param.requires_grad = False
print("  BERT loaded and frozen.")


# ---------------------------------------------------------------------------
# 3. Embedding extraction
# ---------------------------------------------------------------------------
def extract_mean_embeddings(texts: list[str], desc: str = "") -> np.ndarray:
    """Return (N, 768) array of mean-pooled embeddings for a list of texts.

    Mean pooling averages all non-padding token embeddings, which captures
    more of the sentence than the single [CLS] token on short text.
    """
    all_embeddings = []
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(n_batches):
        batch = texts[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding        = True,
            truncation     = True,
            max_length     = MAX_LENGTH,
            return_tensors = "pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():
            output = bert(**encoded)

        # Mean pool over non-padding tokens using attention mask
        token_embeddings = output.last_hidden_state          # (B, T, 768)
        mask = encoded["attention_mask"].unsqueeze(-1).float()  # (B, T, 1)
        summed = (token_embeddings * mask).sum(dim=1)         # (B, 768)
        counts = mask.sum(dim=1).clamp(min=1e-9)             # (B, 1)
        mean_vecs = (summed / counts).cpu().numpy()           # (B, 768)

        all_embeddings.append(mean_vecs)

        if (i + 1) % 20 == 0 or (i + 1) == n_batches:
            print(f"    {desc}: batch {i+1}/{n_batches}")

    return np.vstack(all_embeddings)


print("\nExtracting BERT mean-pooled embeddings...")

t0 = time.time()
X_train = extract_mean_embeddings(train_df["statement"].tolist(), "train")
embed_time_train = time.time() - t0
print(f"  Train embeddings: {X_train.shape}  ({embed_time_train:.1f}s)")

t0 = time.time()
X_val = extract_mean_embeddings(val_df["statement"].tolist(), "val")
embed_time_val = time.time() - t0
print(f"  Val embeddings:   {X_val.shape}  ({embed_time_val:.1f}s)")

t0 = time.time()
X_test = extract_mean_embeddings(test_df["statement"].tolist(), "test")
embed_time_test = time.time() - t0
print(f"  Test embeddings:  {X_test.shape}  ({embed_time_test:.1f}s)")

y_train = train_df["binary_label"].values
y_val   = val_df["binary_label"].values
y_test  = test_df["binary_label"].values


# ---------------------------------------------------------------------------
# 4. Build speaker metadata features
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    "false_count", "barely_true_count", "pants_fire_count",
    "mostly_true_count", "half_true_count",
]
CATEGORICAL_COLS = ["party", "state", "speaker_job"]

print("\nBuilding metadata features...")

# Fill missing values
for df in [train_df, val_df, test_df]:
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()

# Label-encode categoricals — fit only on train, apply to val/test
encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    le.fit(train_df[col])
    known = set(le.classes_)
    # Unseen values in val/test get encoded as -1
    for df in [train_df, val_df, test_df]:
        df[f"{col}_enc"] = df[col].apply(
            lambda v: le.transform([v])[0] if v in known else -1
        )
    encoders[col] = le

ENC_COLS = [f"{c}_enc" for c in CATEGORICAL_COLS]
META_COLS = NUMERIC_COLS + ENC_COLS

def get_meta(df: pd.DataFrame) -> np.ndarray:
    return df[META_COLS].values.astype(float)

M_train = get_meta(train_df)
M_val   = get_meta(val_df)
M_test  = get_meta(test_df)
print(f"  Metadata shape: {M_train.shape}  (cols: {META_COLS})")

# Concatenate BERT embeddings + metadata
X_train = np.hstack([X_train, M_train])
X_val   = np.hstack([X_val,   M_val])
X_test  = np.hstack([X_test,  M_test])
print(f"  Final feature shape: {X_train.shape}  (768 BERT + {len(META_COLS)} metadata)")


# ---------------------------------------------------------------------------
# 5. Train LightGBM with early stopping
# ---------------------------------------------------------------------------
print("\nTraining LightGBM...")
tracemalloc.start()

clf = lgb.LGBMClassifier(**LGBM_PARAMS)

t0 = time.time()
clf.fit(
    X_train, y_train,
    eval_set            = [(X_val, y_val)],
    callbacks           = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                           lgb.log_evaluation(period=50)],
)
train_time = time.time() - t0

_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_mem_mb = peak_mem / (1024 ** 2)

print(f"  Training done in {train_time:.2f}s  |  Best iteration: {clf.best_iteration_}")


# ---------------------------------------------------------------------------
# 6. Evaluate on test set
# ---------------------------------------------------------------------------
print("\nEvaluating on test set...")

t0 = time.time()
y_pred = clf.predict(X_test)
inference_time = time.time() - t0
inference_per_sample_ms = (inference_time / len(y_test)) * 1000

acc       = accuracy_score(y_test, y_pred)
prec_mac  = precision_score(y_test, y_pred, average="macro",    zero_division=0)
rec_mac   = recall_score(y_test, y_pred, average="macro",       zero_division=0)
f1_mac    = f1_score(y_test, y_pred, average="macro",           zero_division=0)
prec_wt   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec_wt    = recall_score(y_test, y_pred, average="weighted",    zero_division=0)
f1_wt     = f1_score(y_test, y_pred, average="weighted",        zero_division=0)
cm        = confusion_matrix(y_test, y_pred).tolist()

print(f"  Accuracy : {acc:.4f}")
print(f"  F1 macro : {f1_mac:.4f}  |  F1 weighted : {f1_wt:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Real','Fake'])}")


# ---------------------------------------------------------------------------
# 7. Save model
# ---------------------------------------------------------------------------
joblib.dump(clf, MODEL_PATH)
model_size_kb = os.path.getsize(MODEL_PATH) / 1024
print(f"Model saved to {MODEL_PATH}  ({model_size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# 8. Save results.json
# ---------------------------------------------------------------------------
results = {
    # Classification metrics
    "accuracy":           round(float(acc),      4),
    "precision_macro":    round(float(prec_mac),  4),
    "recall_macro":       round(float(rec_mac),   4),
    "f1_macro":           round(float(f1_mac),    4),
    "precision_weighted": round(float(prec_wt),   4),
    "recall_weighted":    round(float(rec_wt),    4),
    "f1_weighted":        round(float(f1_wt),     4),
    "confusion_matrix":   cm,

    # Efficiency metrics
    "train_time_sec":             round(train_time,              2),
    "inference_time_sec":         round(inference_time,          4),
    "inference_per_sample_ms":    round(inference_per_sample_ms, 4),
    "peak_memory_mb":             round(peak_mem_mb,             2),
    "model_size_kb":              round(model_size_kb,           1),
    "embed_extraction_train_sec": round(embed_time_train,        2),
    "embed_extraction_val_sec":   round(embed_time_val,          2),
    "embed_extraction_test_sec":  round(embed_time_test,         2),

    # Config metadata
    "bert_model":        BERT_MODEL,
    "embedding_dim":     768,
    "metadata_features": META_COLS,
    "total_feature_dim": X_train.shape[1],
    "batch_size":        BATCH_SIZE,
    "max_length":        MAX_LENGTH,
    "device":            DEVICE,
    "lgbm_params":       LGBM_PARAMS,
    "best_iteration":    int(clf.best_iteration_),
    "train_samples":     int(len(y_train)),
    "val_samples":       int(len(y_val)),
    "test_samples":      int(len(y_test)),
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {RESULTS_PATH}")
print("\nDone.")
