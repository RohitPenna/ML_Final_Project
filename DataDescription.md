# Data Description

> Part of the [Fake News Detection — ML Final Project](README.md)

---

## Raw Source Files

The raw data lives in `data/` and consists of two CSV files from the [ISOT Fake News Dataset](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php), covering political and world news articles from **2016–2018**:

| File | Articles | Label |
|---|---|---|
| `data/True.csv` | 21,417 | Real (0) |
| `data/Fake.csv` | 23,481 | Fake (1) |
| **Total** | **44,898** | |

Each raw article has four fields: `title`, `text`, `subject`, and `date`.

**Subject breakdown — Fake.csv:**
| Subject | Count |
|---|---|
| News | 9,050 |
| politics | 6,841 |
| left-news | 4,459 |
| Government News | 1,570 |
| US_News | 783 |
| Middle-east | 778 |

**Subject breakdown — True.csv:**
| Subject | Count |
|---|---|
| politicsNews | 11,272 |
| worldnews | 10,145 |

---

## Why the Raw Data Is Too Easy

The raw dataset contains several **data-leakage shortcuts** that allow models to achieve 99%+ accuracy without learning anything meaningful about language or content. We deliberately remove these before training so that all six models — from Logistic Regression to BERT — are actually challenged:

| Shortcut | Example | Problem |
|---|---|---|
| Source attributions in titles | `"Trump fires Comey: Reuters"` | Reveals wire source, not content |
| Article format prefixes | `"Factbox: ..."`, `"WATCH: ..."` | Reveals article type / outlet style |
| Parenthetical media tags | `"(VIDEO)"`, `"(IMAGES)"`, `"(TWEET)"` | Strong stylistic signal tied to fake news |
| `subject` column | `"politicsNews"` vs `"News"` | Near-perfect label by itself |
| `date` column | — | Not informative; encourages overfitting to time period |
| Full `text` body | `"WASHINGTON (Reuters) - ..."` | Dateline alone identifies real Reuters articles |

---

## Cleaning Steps Applied

Run `prepare_data.py` to produce clean, debiased splits:

```bash
python3 prepare_data.py
```

The script applies the following transformations before splitting:

1. **Drop `subject` and `date`** — forces models to work on text content only
2. **Drop full `text`** — use only the `title`; the article body is too easy due to datelines and embedded source mentions
3. **Strip trailing source attributions** from titles — removes patterns like `: Reuters`, `: CNN`, `: NYT`, `: Pentagon`, `: report`, `: sources`
4. **Strip leading format prefixes** — removes `WATCH:`, `Factbox:`, `UPDATE:`, `BREAKING:`, `EXCLUSIVE:`
5. **Strip parenthetical media tags** — removes `(VIDEO)`, `(IMAGES)`, `(TWEET)`, `(GRAPHIC IMAGES)`, `(PHOTOS)`

**Final columns:** `title` (str), `label` (int: 0 = real, 1 = fake)

---

## Data Splits

Splits are saved to `data/splits/` using **stratified sampling** to preserve the ~52/48 class ratio across all three sets. The full dataset is shuffled with a fixed random seed (`42`) for reproducibility.

| Split | File | Rows | Share | Fake (1) | Real (0) |
|---|---|---|---|---|---|
| Train | `data/splits/train.csv` | 31,428 | 70% | 16,436 (52.3%) | 14,992 (47.7%) |
| Validation | `data/splits/val.csv` | 6,735 | 15% | 3,522 (52.3%) | 3,213 (47.7%) |
| Test | `data/splits/test.csv` | 6,735 | 15% | 3,523 (52.3%) | 3,212 (47.7%) |
| **Total** | | **44,898** | **100%** | **23,481 (52.3%)** | **21,417 (47.7%)** |

- **Train** — used to fit all models
- **Validation** — used during development for hyperparameter tuning and early stopping
- **Test** — held out until final evaluation; used once per model for reported metrics

---

## Models Using This Data

All six models in this project train and evaluate on the same cleaned splits:

| Model | Folder |
|---|---|
| Logistic Regression (TF-IDF) | `models/logistic_regression/` |
| Support Vector Machine (TF-IDF) | `models/svm/` |
| Naive Bayes | `models/naive_bayes/` |
| DistilBERT (fine-tuned) | `models/distilbert/` |
| BERT (fine-tuned) | `models/bert/` |
| BERT embeddings + LR/SVM (hybrid) | `models/hybrid_bert_classical/` |
