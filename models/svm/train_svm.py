"""
Support Vector Machine classifier using TF-IDF features + metadata + C tuning.

Pipeline:
  1. Load LIAR train / val / test TSV files
  2. Binarize 6-class labels  (true/mostly-true/half-true -> 0, barely-true/false/pants-fire -> 1)
  3. Build feature matrix (3 components hstacked):
       a) Word TF-IDF  (unigrams + bigrams, max 100k features, sublinear_tf=True)
       b) Char TF-IDF  (char_wb 3-5 grams, max 50k features — captures casing/punct signals)
       c) Speaker metadata  (credibility counts + label-encoded party/state/speaker_job)
  4. Grid-search C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0] using validation F1 macro
  5. Retrain final LinearSVC with best C on full training set
  6. Evaluate on the test set and save results to models/svm/results.json
  7. Save model to models/svm/svm_model.pkl
     Save vectorizers to models/svm/tfidf_word_vectorizer.pkl
                         models/svm/tfidf_char_vectorizer.pkl
"""

import json
import os
import time
import tracemalloc
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR          = os.path.join(BASE_DIR, "Data")
OUT_DIR           = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH        = os.path.join(OUT_DIR, "svm_model.pkl")
WORD_VEC_PATH     = os.path.join(OUT_DIR, "tfidf_word_vectorizer.pkl")
CHAR_VEC_PATH     = os.path.join(OUT_DIR, "tfidf_char_vectorizer.pkl")
RESULTS_PATH      = os.path.join(OUT_DIR, "results.json")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORD_TFIDF_PARAMS = dict(
    max_features  = 100_000,
    sublinear_tf  = True,
    ngram_range   = (1, 2),
    min_df        = 2,
    strip_accents = "unicode",
    analyzer      = "word",
    token_pattern = r"\b[a-zA-Z][a-zA-Z0-9']+\b",
)

CHAR_TFIDF_PARAMS = dict(
    max_features  = 50_000,
    sublinear_tf  = True,
    ngram_range   = (3, 5),     # char 3–5 grams: captures casing, punctuation, style
    min_df        = 3,
    strip_accents = "unicode",
    analyzer      = "char_wb",  # pads word boundaries — cleaner than raw char
)

C_GRID   = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
MAX_ITER = 2000

TSV_COLS = [
    "id", "label", "statement", "subjects", "speaker", "speaker_job",
    "state", "party", "barely_true_count", "false_count", "half_true_count",
    "mostly_true_count", "pants_fire_count", "context",
]

NUMERIC_COLS     = ["false_count", "barely_true_count", "pants_fire_count",
                    "mostly_true_count", "half_true_count"]
CATEGORICAL_COLS = ["party", "state", "speaker_job"]

FAKE_LABELS = {"barely-true", "false", "pants-fire"}

def binarize(label: str) -> int:
    return 1 if str(label).strip() in FAKE_LABELS else 0


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

y_train = train_df["binary_label"].values
y_val   = val_df["binary_label"].values
y_test  = test_df["binary_label"].values


# ---------------------------------------------------------------------------
# 2a. Word TF-IDF
# ---------------------------------------------------------------------------
print("\nFitting word TF-IDF vectorizer...")
word_vec = TfidfVectorizer(**WORD_TFIDF_PARAMS)

t0 = time.time()
Xw_train = word_vec.fit_transform(train_df["statement"])
vec_time_train = time.time() - t0
Xw_val  = word_vec.transform(val_df["statement"])
Xw_test = word_vec.transform(test_df["statement"])
print(f"  Shape: {Xw_train.shape}  ({vec_time_train:.3f}s)  vocab={len(word_vec.vocabulary_):,}")


# ---------------------------------------------------------------------------
# 2b. Char TF-IDF
# ---------------------------------------------------------------------------
print("Fitting char TF-IDF vectorizer...")
char_vec = TfidfVectorizer(**CHAR_TFIDF_PARAMS)

t0 = time.time()
Xc_train = char_vec.fit_transform(train_df["statement"])
vec_time_char = time.time() - t0
Xc_val  = char_vec.transform(val_df["statement"])
Xc_test = char_vec.transform(test_df["statement"])
print(f"  Shape: {Xc_train.shape}  ({vec_time_char:.3f}s)  vocab={len(char_vec.vocabulary_):,}")

vec_time_total = vec_time_train + vec_time_char


# ---------------------------------------------------------------------------
# 2c. Speaker metadata
# ---------------------------------------------------------------------------
print("Building speaker metadata features...")

for df in [train_df, val_df, test_df]:
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()

encoders = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    le.fit(train_df[col])
    known = set(le.classes_)
    for df in [train_df, val_df, test_df]:
        df[f"{col}_enc"] = df[col].apply(
            lambda v: int(le.transform([v])[0]) if v in known else -1
        )
    encoders[col] = le

META_COLS = NUMERIC_COLS + [f"{c}_enc" for c in CATEGORICAL_COLS]

scaler = StandardScaler()
Xm_train = sp.csr_matrix(scaler.fit_transform(train_df[META_COLS].values.astype(float)))
Xm_val   = sp.csr_matrix(scaler.transform(val_df[META_COLS].values.astype(float)))
Xm_test  = sp.csr_matrix(scaler.transform(test_df[META_COLS].values.astype(float)))
print(f"  Metadata shape: {Xm_train.shape}  (cols: {META_COLS}, scaled)")


# ---------------------------------------------------------------------------
# 2d. Stack all features
# ---------------------------------------------------------------------------
X_train = sp.hstack([Xw_train, Xc_train, Xm_train], format="csr")
X_val   = sp.hstack([Xw_val,   Xc_val,   Xm_val],   format="csr")
X_test  = sp.hstack([Xw_test,  Xc_test,  Xm_test],  format="csr")
print(f"\nFinal feature matrix: {X_train.shape}  "
      f"(word={Xw_train.shape[1]} + char={Xc_train.shape[1]} + meta={Xm_train.shape[1]})")


# ---------------------------------------------------------------------------
# 3. Grid-search C on validation set
# ---------------------------------------------------------------------------
print("\nGrid-searching C on validation F1 macro...")
print(f"  C values: {C_GRID}")

best_C, best_val_f1 = None, -1.0
c_search_results = {}

for C in C_GRID:
    clf_tmp = LinearSVC(C=C, max_iter=MAX_ITER, class_weight="balanced", random_state=42)
    clf_tmp.fit(X_train, y_train)
    val_pred  = clf_tmp.predict(X_val)
    val_f1_c  = f1_score(y_val, val_pred, average="macro", zero_division=0)
    val_acc_c = accuracy_score(y_val, val_pred)
    c_search_results[C] = {"val_f1_macro": round(val_f1_c, 4),
                           "val_accuracy": round(val_acc_c, 4)}
    marker = " ◄ best" if val_f1_c > best_val_f1 else ""
    print(f"    C={C:<6}  val F1 macro={val_f1_c:.4f}  val acc={val_acc_c:.4f}{marker}")
    if val_f1_c > best_val_f1:
        best_val_f1 = val_f1_c
        best_C      = C

print(f"\n  Best C = {best_C}  (val F1 macro = {best_val_f1:.4f})")


# ---------------------------------------------------------------------------
# 4. Retrain with best C and measure training time / memory
# ---------------------------------------------------------------------------
print(f"\nRetraining final SVM with C={best_C}...")
tracemalloc.start()

clf = LinearSVC(C=best_C, max_iter=MAX_ITER, class_weight="balanced", random_state=42)

t0 = time.time()
clf.fit(X_train, y_train)
train_time = time.time() - t0

_, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()
peak_mem_mb = peak_mem / (1024 ** 2)
print(f"  Training done in {train_time:.3f}s")


# ---------------------------------------------------------------------------
# 5. Test evaluation
# ---------------------------------------------------------------------------
print("\nEvaluating on test set...")

t0 = time.time()
y_pred = clf.predict(X_test)
inference_time = time.time() - t0
inference_per_sample_ms = (inference_time / len(y_test)) * 1000

acc      = accuracy_score(y_test, y_pred)
prec_mac = precision_score(y_test, y_pred, average="macro",    zero_division=0)
rec_mac  = recall_score(y_test, y_pred, average="macro",       zero_division=0)
f1_mac   = f1_score(y_test, y_pred, average="macro",           zero_division=0)
prec_wt  = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec_wt   = recall_score(y_test, y_pred, average="weighted",    zero_division=0)
f1_wt    = f1_score(y_test, y_pred, average="weighted",        zero_division=0)
cm       = confusion_matrix(y_test, y_pred).tolist()

print(f"  Accuracy : {acc:.4f}")
print(f"  F1 macro : {f1_mac:.4f}  |  F1 weighted : {f1_wt:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Real', 'Fake'])}")


# ---------------------------------------------------------------------------
# 6. Save model and vectorizers
# ---------------------------------------------------------------------------
joblib.dump(clf,      MODEL_PATH)
joblib.dump(word_vec, WORD_VEC_PATH)
joblib.dump(char_vec, CHAR_VEC_PATH)

model_size_kb     = os.path.getsize(MODEL_PATH) / 1024
word_vec_size_kb  = os.path.getsize(WORD_VEC_PATH) / 1024
char_vec_size_kb  = os.path.getsize(CHAR_VEC_PATH) / 1024

print(f"SVM model saved         : {MODEL_PATH}  ({model_size_kb:.1f} KB)")
print(f"Word vectorizer saved   : {WORD_VEC_PATH}  ({word_vec_size_kb:.1f} KB)")
print(f"Char vectorizer saved   : {CHAR_VEC_PATH}  ({char_vec_size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# 7. Save results.json
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

    # Validation / C-search
    "best_C":             best_C,
    "val_f1_macro":       round(best_val_f1, 4),
    "c_search_results":   c_search_results,

    # Efficiency metrics
    "train_time_sec":           round(train_time,              4),
    "inference_time_sec":       round(inference_time,          4),
    "inference_per_sample_ms":  round(inference_per_sample_ms, 4),
    "peak_memory_mb":           round(peak_mem_mb,             2),
    "model_size_kb":            round(model_size_kb,           1),
    "word_vectorizer_size_kb":  round(word_vec_size_kb,        1),
    "char_vectorizer_size_kb":  round(char_vec_size_kb,        1),
    "vec_time_word_sec":        round(vec_time_train,          4),
    "vec_time_char_sec":        round(vec_time_char,           4),
    "vec_time_total_sec":       round(vec_time_total,          4),

    # Config
    "word_tfidf_params":  WORD_TFIDF_PARAMS,
    "char_tfidf_params":  CHAR_TFIDF_PARAMS,
    "metadata_features":  META_COLS,
    "word_vocab_size":    len(word_vec.vocabulary_),
    "char_vocab_size":    len(char_vec.vocabulary_),
    "total_feature_dim":  X_train.shape[1],
    "train_samples":      int(len(y_train)),
    "val_samples":        int(len(y_val)),
    "test_samples":       int(len(y_test)),
}

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {RESULTS_PATH}")
print("\nDone.")
