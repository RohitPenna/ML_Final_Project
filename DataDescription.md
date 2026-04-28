# Data Description

> Part of the [Fake News Detection — ML Final Project](README.md)

---

## Dataset: LIAR

The dataset used in this project is the **[LIAR dataset](https://huggingface.co/datasets/liar)** (Wang, 2017), a benchmark for fake news detection sourced from **PolitiFact**. It contains short political statements made by public figures, each manually fact-checked and assigned one of six veracity labels.

- **Source:** PolitiFact.com
- **Total statements:** 12,791
- **Label type:** 6-class → binarized to 2-class for this project
- **Statement length:** mean ~18 words, median 17 words (range: 2–467)

---

## Columns

| Column | Description |
|---|---|
| `id` | JSON filename identifier |
| `label` | Veracity label (6-class, see below) |
| `statement` | The political claim being fact-checked |
| `subjects` | Comma-separated topic tags |
| `speaker` | Name of the person who made the claim |
| `speaker_job` | Job title of the speaker |
| `state` | State the speaker is from |
| `party` | Political party affiliation |
| `barely_true_count` | Historical count of barely-true rulings for this speaker |
| `false_count` | Historical count of false rulings for this speaker |
| `half_true_count` | Historical count of half-true rulings for this speaker |
| `mostly_true_count` | Historical count of mostly-true rulings for this speaker |
| `pants_fire_count` | Historical count of pants-on-fire rulings for this speaker |
| `context` | Venue/context of the statement |

---

## Label Distribution

The raw dataset has **6 fine-grained labels**:

| Label | Count | Share |
|---|---|---|
| half-true | 2,627 | 20.5% |
| false | 2,507 | 19.6% |
| mostly-true | 2,454 | 19.2% |
| barely-true | 2,103 | 16.4% |
| true | 2,053 | 16.1% |
| pants-fire | 1,047 | 8.2% |
| **Total** | **12,791** | |

### Binary Label Mapping

For binary classification (real vs. fake), labels are mapped as follows:

| Original Label | Binary Label |
|---|---|
| true | 0 — Real |
| mostly-true | 0 — Real |
| half-true | 0 — Real |
| barely-true | 1 — Fake |
| false | 1 — Fake |
| pants-fire | 1 — Fake |

---

## Speaker & Party Breakdown

**Top parties:**
| Party | Statements |
|---|---|
| Republican | 5,665 |
| Democrat | 4,137 |
| None / Unaffiliated | 2,181 |
| Organization | 264 |
| Independent | 180 |

**Top subjects:**
| Subject | Count |
|---|---|
| economy | 1,432 |
| health-care | 1,426 |
| taxes | 1,218 |
| federal-budget | 937 |
| education | 926 |
| jobs | 899 |
| state-budget | 879 |
| candidates-biography | 805 |
| elections | 757 |
| immigration | 642 |

---

## Pre-split Files

The dataset comes pre-split into three TSV files in `Data/`:

| Split | File | Rows |
|---|---|---|
| Train | `Data/train.tsv` | 10,240 (80.1%) |
| Validation | `Data/valid.tsv` | 1,284 (10.0%) |
| Test | `Data/test.tsv` | 1,267 (9.9%) |
| **Total** | | **12,791** |

- **Train** — used to fit all models
- **Validation** — used during development for hyperparameter tuning and early stopping
- **Test** — held out until final evaluation; used once per model for reported metrics

---

## Key Characteristics vs. ISOT

| Property | LIAR | ISOT (previous) |
|---|---|---|
| Size | 12,791 | 44,898 |
| Input | Short political statements (~18 words) | News headlines / full articles |
| Labels | 6-class (binarized) | Binary |
| Source | PolitiFact | Reuters / unreliable outlets |
| Speaker metadata | Yes | No |
| Domain leakage risk | Low | High (datelines, source names) |
| Generalization difficulty | High | Low |

---

## Models Using This Data

All six models in this project train and evaluate on the same LIAR splits:

| Model | Folder |
|---|---|
| Logistic Regression (TF-IDF) | `models/logistic_regression/` |
| Support Vector Machine (TF-IDF) | `models/svm/` |
| Naive Bayes | `models/naive_bayes/` |
| DistilBERT (fine-tuned) | `models/distilbert/` |
| BERT (fine-tuned) | `models/bert/` |
| BERT embeddings + LR/SVM (hybrid) | `models/hybrid_bert_classical/` |
