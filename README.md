# Fake News Detection — ML Final Project

A machine learning project that classifies news articles as **real or fake** using the ISOT Fake News Dataset. The project covers the full ML pipeline: data preparation, model training, evaluation, and analysis.

---

## Overview

Misinformation spreads faster than ever online. This project explores how well machine learning models can automatically detect fake news articles based solely on their text content. We frame this as a **binary classification** task (real = 0, fake = 1).

---

## Dataset

~44,900 labeled news articles sourced from the [ISOT Fake News Dataset](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php), covering political and world news from 2016–2018.

- **Fake articles:** 23,481
- **Real articles:** 21,417

See [`DataDescription.md`](DataDescription.md) for full details on the data distribution and splits.

---

## Project Structure

```
ML_Final_Project/
├── data/
│   ├── Fake.csv                      # Raw fake news articles
│   ├── True.csv                      # Raw real news articles
│   └── splits/
│       ├── train.csv                 # 70% — model training
│       ├── val.csv                   # 15% — hyperparameter tuning
│       └── test.csv                  # 15% — final evaluation
├── models/
│   ├── logistic_regression/          # TF-IDF + Logistic Regression
│   ├── svm/                          # TF-IDF + Support Vector Machine
│   ├── naive_bayes/                  # Naive Bayes classifier
│   ├── distilbert/                   # DistilBERT fine-tuned model
│   ├── bert/                         # BERT fine-tuned model
│   └── hybrid_bert_classical/        # BERT embeddings + LR/SVM
├── prepare_data.py                   # Data combination & splitting script
├── DataDescription.md                # Detailed data documentation
└── README.md                         # This file
```

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/RohitPenna/ML_Final_Project.git
cd ML_Final_Project
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Prepare the data splits**
```bash
python3 prepare_data.py
```

This will generate the train, validation, and test CSVs in `data/splits/`.

---

## Results

_Coming soon — model training and evaluation in progress._

---

## License

This project is for academic purposes. The underlying dataset is provided by the University of Victoria ISOT Research Lab.
