# Solvent Screening Framework — Source Code and Data

This repository contains the source code and curated datasets for the manuscript:

> Machine Learning-Based Solubility Prediction and Flammability Risk Assessment: A Safe and Efficient Solvent Screening Framework

The code reproduces the feature engineering, model training, hyperparameter optimization, and evaluation reported in the manuscript.
## Repository structure

solvent-screening-data-2027/
├── README.md
├── data/
│   ├── solubility/
│   │   ├── total.csv
│   │   ├── train.csv
│   │   ├── test_type1.csv
│   │   ├── test_type2.csv
│   │   ├── test_type3.csv
│   │   └── test_type4.csv
│   │   ├── features_total.csv
│   │   ├── features_train.csv
│   │   ├── features_test_type1.csv
│   │   ├── features_test_type2.csv
│   │   ├── features_test_type3.csv
│   │   └── features_test_type4.csv
│   └── flammability/
│       └── dataset.csv
        └── features_dataset.csv
├── src/
│   ├── train_solubility1.py
│   ├── train_solubility2.py
│   ├── train_flammability.py
└── models/
    ├── lightgbm_solubility.pkl
    └── svm_flammability.pkl

## Environment

- Python 3.10
- rdkit
- mordred
- lightgbm
- scikit-learn
- optuna
- shap
- pandas
- numpy
- matplotlib
- seaborn

## Contents

### Data

| File | Description |
|---|---|
| `data/solubility/total.csv` | Full curated solubility dataset (14,950 samples) |
| `data/solubility/train.csv` | Training set (11,451 samples) |
| `data/solubility/test_type1.csv` | Unseen temperatures (1,545 samples) |
| `data/solubility/test_type2.csv` | Unseen solutes (823 samples) |
| `data/solubility/test_type3.csv` | Unseen solvents (978 samples) |
| `models/svm_flammability.pkl` | Trained SVM flammability classifier |
