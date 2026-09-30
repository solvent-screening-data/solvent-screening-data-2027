# Solvent Screening Framework — Source Code and Data

This repository contains the source code and curated datasets for the manuscript:

> Machine Learning-Based Solubility Prediction and Flammability Risk Assessment: A Safe and Efficient Solvent Screening Framework

The code reproduces the feature engineering, model training, hyperparameter optimization, and evaluation reported in the manuscript.

## Repository structure

```
solvent-screening-data-2027/
├── README.md
├── data/
│   ├── solubility/
│   │   ├── total.csv
│   │   ├── train.csv
│   │   ├── test_type1.csv
│   │   ├── test_type2.csv
│   │   ├── test_type3.csv
│   │   ├── test_type4.csv
│   │   ├── features_total.csv
│   │   ├── features_train.csv
│   │   ├── features_test_type1.csv
│   │   ├── features_test_type2.csv
│   │   ├── features_test_type3.csv
│   │   └── features_test_type4.csv
│   └── flammability/
│       ├── dataset.csv
│       └── features_dataset.csv
├── src/
   ├── train_solubility1.py
   ├── train_solubility2.py
   └── train_flammability.py

```

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
| `data/solubility/test_type4.csv` | Fully unseen solute–solvent pairs (153 samples) |
| `data/solubility/features_total.csv` | Generated feature file for the full dataset |
| `data/solubility/features_train.csv` | Generated feature file for the training set |
| `data/solubility/features_test_type1.csv` | Generated feature file for Type I test set |
| `data/solubility/features_test_type2.csv` | Generated feature file for Type II test set |
| `data/solubility/features_test_type3.csv` | Generated feature file for Type III test set |
| `data/solubility/features_test_type4.csv` | Generated feature file for Type IV test set |
| `data/flammability/dataset.csv` | Flammability risk dataset (1,854 compounds, HFL/FL/NFL) |
| `data/flammability/features_dataset.csv` | Generated feature file for the flammability dataset |

### Code

| Script | Description |
|---|---|
| `src/train_solubility1.py` | Preliminary evaluation of solubility prediction models: trains the LightGBM model with TPE-based hyperparameter optimization and evaluates it on the random train/test split |
| `src/train_solubility2.py` | Generalization evaluation of solubility prediction models: retrains the model on the re-partitioned dataset and evaluates it on the four external test sets |
| `src/train_flammability.py` | Trains the SVM classifier, performs feature selection and hyperparameter optimization, and generates the SHAP analysis |
