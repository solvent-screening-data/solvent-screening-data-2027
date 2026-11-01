import pandas as pd
import numpy as np
import time
import lightgbm as lgb
import shap
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set random seed
np.random.seed(42)

# ---------------------- Data Preparation ----------------------
start_time = time.time()

# Load training data
df = pd.read_csv('data/solubility/train.csv')

# Separate features and target variable
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
feature_names = df.columns[1:].tolist()

# ====================== Feature Selection ======================
feature_start = time.time()

# 1. Calculate variance
feature_variances = np.var(X, axis=0)

# 2. Remove zero-variance features
non_zero_indices = np.where(feature_variances > 0)[0]
print(f"Number of zero-variance features: {X.shape[1] - len(non_zero_indices)}")

# 3. Select top 90% features among non-zero variance ones
if len(non_zero_indices) > 0:
    non_zero_variances = feature_variances[non_zero_indices]
    var_threshold = np.percentile(non_zero_variances, 10)

    epsilon = 1e-12
    selected_mask = non_zero_variances >= var_threshold - epsilon
    selected_indices = non_zero_indices[selected_mask]

    if len(selected_indices) == 0:
        print("Warning: No features selected, using all non-zero variance features.")
        selected_indices = non_zero_indices
else:
    print("Warning: All feature variances are zero, using all features.")
    selected_indices = np.arange(X.shape[1])

if len(selected_indices) == 0:
    print("Error: Feature selection failed, using first 100 features.")
    selected_indices = np.arange(min(100, X.shape[1]))

X_filtered = X[:, selected_indices]
selected_feature_names = [feature_names[i] for i in selected_indices]

print(f"\nFeature engineering time: {time.time() - feature_start:.2f} seconds")
print(f"Variance threshold: {var_threshold:.6f}")
print(f"Feature dimension change: {X.shape[1]} -> {X_filtered.shape[1]}")

# ---------------------- Hyperparameter Optimization ----------------------
model_start = time.time()

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'random_state': 42,
        'n_jobs': -1
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_rmse_scores = []

    for train_idx, val_idx in kf.split(X_filtered):
        X_train_fold, X_val_fold = X_filtered[train_idx], X_filtered[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        y_val_pred = model.predict(X_val_fold)
        val_rmse_scores.append(np.sqrt(mean_squared_error(y_val_fold, y_val_pred)))

    return np.mean(val_rmse_scores)

# Hyperparameter search
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=3600)
best_params = study.best_params

print(f"\n[Optimization Result] Best hyperparameters: {best_params}")

# ---------------------- Train Final Model ----------------------
final_model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1)
final_model.fit(X_filtered, y)
print(f"\n[Training Info] Total time: {time.time() - model_start:.2f} seconds")

# ---------------------- Model Evaluation ----------------------
def evaluate_model(model, X, y, set_name):
    start = time.time()
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"\n[{set_name}]")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    return [mse, rmse, mae, r2]

train_results = evaluate_model(final_model, X_filtered, y, "Train Set")

# ---------------------- External Test Sets Evaluation ----------------------
def evaluate_external_data(file_path, model, set_name):
    external_df = pd.read_csv(file_path)
    X_external = external_df.iloc[:, 1:].values
    y_external = external_df.iloc[:, 0].values
    X_external_filtered = X_external[:, selected_indices]
    return evaluate_model(model, X_external_filtered, y_external, set_name)

external_results1 = evaluate_external_data('data/solubility/test_type1.csv', final_model, "External Test Set 1")
external_results2 = evaluate_external_data('data/solubility/test_type2.csv', final_model, "External Test Set 2")
external_results3 = evaluate_external_data('data/solubility/test_type3.csv', final_model, "External Test Set 3")
external_results4 = evaluate_external_data('data/solubility/test_type4.csv', final_model, "External Test Set 4")

# ---------------------- Results Summary ----------------------
metrics = ["MSE", "RMSE", "MAE", "R2"]
print("\n[Final Results]")
print("Train Set:", {metric: value for metric, value in zip(metrics, train_results)})
print("External Test Set 1:", {metric: value for metric, value in zip(metrics, external_results1)})
print("External Test Set 2:", {metric: value for metric, value in zip(metrics, external_results2)})
print("External Test Set 3:", {metric: value for metric, value in zip(metrics, external_results3)})
print("External Test Set 4:", {metric: value for metric, value in zip(metrics, external_results4)})

# ---------------------- SHAP Interpretability Analysis ----------------------
explainer = shap.Explainer(final_model, X_filtered)
shap_values = explainer(X_filtered)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_filtered, feature_names=selected_feature_names, show=False)

# ---------------------- Visualization: True vs Predicted ----------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.regplot(x=y, y=final_model.predict(X_filtered), scatter_kws={'alpha': 0.5})
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
plt.xlabel("True Value")
plt.ylabel("Predicted Value")

print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")
plt.show()
