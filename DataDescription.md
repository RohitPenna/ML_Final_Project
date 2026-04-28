# ML Final Project — Fake News Detection

A binary text classification project to distinguish real news articles from fake ones using the ISOT Fake News Dataset.

---

## Dataset

The raw data lives in `data/` and consists of two CSV files:

| File | Articles | Label |
|---|---|---|
| `data/True.csv` | 21,417 | Real (0) |
| `data/Fake.csv` | 23,481 | Fake (1) |
| **Total** | **44,898** | |

Each article has four fields: `title`, `text`, `subject`, and `date`.

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

## Data Splits

Run `prepare_data.py` to combine, shuffle, and split the data:

```bash
python3 prepare_data.py
```

This produces three stratified splits saved to `data/splits/`:

| Split | File | Rows | Fake | Real |
|---|---|---|---|---|
| Train | `data/splits/train.csv` | 31,428 (70%) | 16,436 (52.3%) | 14,992 (47.7%) |
| Validation | `data/splits/val.csv` | 6,735 (15%) | 3,522 (52.3%) | 3,213 (47.7%) |
| Test | `data/splits/test.csv` | 6,735 (15%) | 3,523 (52.3%) | 3,212 (47.7%) |
| **Total** | | **44,898** | **23,481 (52.3%)** | **21,417 (47.7%)** |

Splits are stratified by label to preserve the ~52/48 fake-to-real class ratio across all three sets. The dataset is shuffled with a fixed random seed (`42`) for reproducibility.

---

## Project Structure

```
ML_Final_Project/
├── data/
│   ├── Fake.csv          # Raw fake news articles
│   ├── True.csv          # Raw real news articles
│   └── splits/
│       ├── train.csv     # 70% — model training
│       ├── val.csv       # 15% — hyperparameter tuning
│       └── test.csv      # 15% — final evaluation
├── prepare_data.py       # Data combination & splitting script
└── README.md
```
