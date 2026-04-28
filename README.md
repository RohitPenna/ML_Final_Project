# Fake News Detection — ML Final Project

A machine learning project that classifies political statements as **real or fake** using the LIAR dataset. The project covers the full ML pipeline: data preparation, model training, evaluation, and comparison across six model architectures.

---

## Overview

Misinformation spreads faster than ever online. This project explores how well machine learning models can automatically detect false political claims based on statement text and speaker metadata. We frame this as a **binary classification** task — mapping PolitiFact's 6-class veracity scale down to real (0) vs. fake (1) — and compare classical NLP models against fine-tuned transformers.

---

## Dataset

**12,791 political statements** sourced from the [LIAR dataset](https://huggingface.co/datasets/liar) (Wang, 2017), fact-checked by PolitiFact. Each statement is made by a public figure and assigned one of six veracity labels, which we binarize into real (0) vs. fake (1).

- **Statements:** 12,791 across train / validation / test
- **Labels:** 6-class (`true`, `mostly-true`, `half-true`, `barely-true`, `false`, `pants-fire`) → binarized
- **Avg. statement length:** ~18 words
- **Features:** statement text + optional speaker metadata (party, job, historical credibility counts)

See [`DataDescription.md`](DataDescription.md) for full details on the data distribution, label mapping, and splits.

---

## Project Structure

```
ML_Final_Project/
├── Data/
│   ├── train.tsv                     # 10,240 statements — model training
│   ├── valid.tsv                     # 1,284 statements  — hyperparameter tuning
│   └── test.tsv                      # 1,267 statements  — final evaluation
├── models/
│   ├── logistic_regression/          # TF-IDF + Logistic Regression
│   ├── svm/                          # TF-IDF + Support Vector Machine
│   ├── naive_bayes/                  # Naive Bayes classifier
│   ├── distilbert/                   # DistilBERT fine-tuned model
│   ├── bert/                         # BERT fine-tuned model
│   └── hybrid_bert_classical/        # BERT embeddings + LR/SVM
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

The LIAR dataset splits (`Data/train.tsv`, `Data/valid.tsv`, `Data/test.tsv`) are pre-split and ready to use — no additional preparation step needed.

---

## Results

_Coming soon — model training and evaluation in progress._

---

## License

This project is for academic purposes. The LIAR dataset was introduced by William Yang Wang in the paper ["'Liar, Liar Pants on Fire': A New Benchmark Dataset for Fake News Detection"](https://aclanthology.org/P17-2067/) (ACL 2017) and is sourced from PolitiFact.com.
