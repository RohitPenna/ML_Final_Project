import os
import time
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, ConfusionMatrixDisplay)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "..", "..", "Data")

COLS = ["id","label","statement","subject","speaker","speaker_job",
        "state","party","barely_true_count","false_count","half_true_count",
        "mostly_true_count","pants_fire_count","context"]

train = pd.read_csv(os.path.join(DATA, "train.tsv"), sep="\t", header=None, names=COLS)
val   = pd.read_csv(os.path.join(DATA, "valid.tsv"), sep="\t", header=None, names=COLS)
test  = pd.read_csv(os.path.join(DATA, "test.tsv"),  sep="\t", header=None, names=COLS)

FAKE = {"barely-true", "false", "pants-fire"}
for df in [train, val, test]:
    df["label"] = df["label"].apply(lambda x: 1 if x in FAKE else 0)

train = train.dropna(subset=["statement"])
val   = val.dropna(subset=["statement"])
test  = test.dropna(subset=["statement"])

word_vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, analyzer="word")
char_vectorizer = TfidfVectorizer(max_features=50000, analyzer="char", ngram_range=(2,4))

X_train = hstack([word_vectorizer.fit_transform(train["statement"]),
                  char_vectorizer.fit_transform(train["statement"])])
X_val   = hstack([word_vectorizer.transform(val["statement"]),
                  char_vectorizer.transform(val["statement"])])
X_test  = hstack([word_vectorizer.transform(test["statement"]),
                  char_vectorizer.transform(test["statement"])])

y_train, y_val, y_test = train["label"], val["label"], test["label"]

model = ComplementNB()
start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start
print(f"Training time: {train_time:.4f}s")
model_size = len(pickle.dumps(model)) / 1024
word_size  = len(pickle.dumps(word_vectorizer)) / 1024
char_size  = len(pickle.dumps(char_vectorizer)) / 1024
print(f"Model size:       {model_size:.1f} KB")
print(f"Word TF-IDF size: {word_size:.1f} KB")
print(f"Char TF-IDF size: {char_size:.1f} KB")

def evaluate(name, X, y_true):
    y_pred = model.predict(X)
    print(f"\n--- Naive Bayes | {name} ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["false", "true"])
    disp.plot(colorbar=True, cmap="Blues")
    plt.title(f"Naive Bayes Confusion Matrix ({name} Set)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, f"confusion_matrix_{name.lower()}.png"))
    plt.show()

evaluate("Validation", X_val, y_val)
evaluate("Test",       X_test, y_test)
