import pandas as pd
import numpy as np
import time
import lightgbm as lgb
import shap
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.ticker import MultipleLocator

# Global font settings for plots
plt.rcParams['font.family'] = 'Times New Roman'

# ---------------------- Data Loading & Processing ----------------------
start_time = time.time()
df = pd.read_csv('data/solubility/features_total.csv')
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
feature_names = df.columns[1:].tolist()

print("Data loaded successfully.")
print(f"Original data shape: {df.shape}")
print(f"Total features: {X.shape[1]}")

# Train-test split
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split completed.")
print(f"Train set shape: {X_train_full.shape}")
print(f"Test set shape: {X_test.shape}")

# ====================== Feature Selection ======================
print("\nStarting feature selection...")

# 1. Calculate variance
feature_variances = np.var(X_train_full, axis=0)

# 2. Remove zero-variance features
non_zero_indices = np.where(feature_variances > 0)[0]
zero_var_count = X_train_full.shape[1] - len(non_zero_indices)
print(f"Number of zero-variance features: {zero_var_count}")
print(f"Number of non-zero variance features: {len(non_zero_indices)}")

if zero_var_count > 0 and zero_var_count <= 10:
    zero_var_names = [feature_names[i] for i in np.where(feature_variances == 0)[0][:10]]
    print(f"Examples of zero-variance features: {zero_var_names}")

# 3. Select top 90% features among non-zero variance ones
if len(non_zero_indices) > 0:
    non_zero_variances = feature_variances[non_zero_indices]
    var_threshold = np.percentile(non_zero_variances, 10)
    print(f"10th percentile of non-zero variances: {var_threshold:.10f}")

    epsilon = 1e-12
    selected_mask = non_zero_variances >= var_threshold - epsilon
    selected_indices = non_zero_indices[selected_mask]

    if len(selected_indices) == 0:
        print("Warning: No features selected, using all non-zero variance features.")
        selected_indices = non_zero_indices
else:
    print("Warning: All feature variances are zero, using all features.")
    selected_indices = np.arange(X_train_full.shape[1])

if len(selected_indices) == 0:
    print("Error: Feature selection failed, using first 100 features.")
    selected_indices = np.arange(min(100, X_train_full.shape[1]))

print("Feature selection completed:")
print(f"   Original feature count: {X_train_full.shape[1]}")
print(f"   Selected feature count: {len(selected_indices)}")
print(f"   Reduction ratio: {(X_train_full.shape[1] - len(selected_indices)) / X_train_full.shape[1] * 100:.1f}%")
print(f"   Removed {zero_var_count} zero-variance features.")

# Apply feature selection
X_train_filtered = X_train_full[:, selected_indices]
X_test_filtered = X_test[:, selected_indices]
selected_feature_names = [feature_names[i] for i in selected_indices]

# ====================== Feature Quality Check ======================
print("\nFeature quality check:")
selected_variances = np.var(X_train_filtered, axis=0)
print(f"   Selected variance range: [{selected_variances.min():.10f}, {selected_variances.max():.10f}]")
print(f"   Mean selected variance: {selected_variances.mean():.10f}")

remaining_zero_var = np.sum(selected_variances == 0)
if remaining_zero_var > 0:
    print(f"   Warning: {remaining_zero_var} zero-variance features remain.")

# ---------------------- Optuna Hyperparameter Optimization ----------------------
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 128),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = []
    for train_idx, val_idx in kf.split(X_train_filtered):
        X_train_cv, X_val_cv = X_train_filtered[train_idx], X_train_filtered[val_idx]
        y_train_cv, y_val_cv = y_train_full[train_idx], y_train_full[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_cv, y_train_cv,
                  eval_set=[(X_val_cv, y_val_cv)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        preds = model.predict(X_val_cv)
        rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, preds)))
    return np.mean(rmse_scores)

print("\nStarting Optuna hyperparameter optimization...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=3600) 
best_params = study.best_params
print(f"\nBest hyperparameters: {best_params}")

# ---------------------- Model Training & Evaluation ----------------------
print("\nStarting final model training...")
final_model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
final_model.fit(X_train_filtered, y_train_full)

print("Model training completed.")
print(f"   Actual number of trees used: {final_model.n_estimators_}")

def evaluate(model, X, y, name):
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"\n[{name}] Evaluation results:")
    print(f"   MSE = {mse:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   MAE = {mae:.4f}")
    print(f"   R2 = {r2:.4f}")
    return pred

y_train_pred = evaluate(final_model, X_train_filtered, y_train_full, 'Train Set')
y_test_pred = evaluate(final_model, X_test_filtered, y_test, 'Test Set')

# ====================== Experimental vs Predicted Plot ========================
print("\nGenerating visualization plots...")

plt.figure(figsize=(6, 6))
train_color = '#1f77b4'
test_color = '#ff7f0e'

plt.scatter(y_train_full, y_train_pred, color=train_color, alpha=0.6, label='Train', s=25)
plt.scatter(y_test, y_test_pred, color=test_color, alpha=0.6, label='Test', s=25)
plt.plot([-9.5, 4.5], [-9.5, 4.5], 'k--', linewidth=1.2)

plt.xlabel('Experimental value', fontsize=16)
plt.ylabel('Predicted value', fontsize=16)
plt.title('LightGBM Model', fontsize=16)
plt.legend(loc='upper left', fontsize=12, frameon=True)
plt.xlim(-9.5, 4.5)
plt.ylim(-9.5, 4.5)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
plt.gca().yaxis.set_major_locator(MultipleLocator(2))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.tick_params(axis='both', which='major', direction='out', length=6, width=1.2, labelsize=14)
plt.tick_params(axis='both', which='minor', direction='out', length=3, width=1.0)
for spine in ['top', 'right', 'bottom', 'left']:
    plt.gca().spines[spine].set_visible(True)
    plt.gca().spines[spine].set_linewidth(1.2)
plt.tight_layout()
plt.savefig("results/lgb_scatter_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# ====================== Line Plot ========================
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Experimental values', color='blue', linewidth=2)
plt.plot(y_test_pred, label='Predicted values', color='red', linewidth=2)
plt.xlabel('Sample Index', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.title('LightGBM Model', fontsize=16)
plt.legend(fontsize=12)
plt.xlim(-100, 3100)
plt.ylim(-9.5, 4.5)
plt.xticks(np.arange(0, 3001, 500), fontsize=14)
plt.yticks(np.arange(-9, 5, 2), fontsize=14)
plt.gca().xaxis.set_minor_locator(MultipleLocator(250))
plt.gca().yaxis.set_major_locator(MultipleLocator(2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.tick_params(axis='both', which='major', direction='out', length=6, width=1.2, labelsize=14)
plt.tick_params(axis='both', which='minor', direction='out', length=3, width=1.0)
for spine in ['top', 'right', 'bottom', 'left']:
    plt.gca().spines[spine].set_visible(True)
    plt.gca().spines[spine].set_linewidth(1.2)
plt.tight_layout()
plt.savefig("results/lgb_line_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# ====================== Residual Plot ========================
train_residuals = y_train_full - y_train_pred
test_residuals = y_test - y_test_pred

plt.figure(figsize=(6, 4))
plt.scatter(y_train_full, train_residuals, alpha=0.7, color=train_color, label='Train', s=20)
plt.scatter(y_test, test_residuals, alpha=0.7, color=test_color, label='Test', s=20)
plt.axhline(0, color='black', linestyle='-', linewidth=1)

plt.xlabel('Experimental value', fontsize=16)
plt.ylabel('Prediction error', fontsize=16)
plt.title('LightGBM Model', fontsize=16)
plt.legend(loc='upper right', fontsize=12, frameon=True)
plt.xlim(-9.5, 4.5)
plt.ylim(-4.5, 4.5)
plt.gca().xaxis.set_major_locator(MultipleLocator(2))
plt.gca().yaxis.set_major_locator(MultipleLocator(2))
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.tick_params(axis='both', which='major', direction='out', length=6, width=1.2, labelsize=14)
plt.tick_params(axis='both', which='minor', direction='out', length=3, width=1.0)
for spine in ['top', 'right', 'bottom', 'left']:
    plt.gca().spines[spine].set_visible(True)
    plt.gca().spines[spine].set_linewidth(1.2)
plt.tight_layout()
plt.savefig("results/lgb_residual_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# ====================== SHAP Summary Plot ========================
print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test_filtered)

# Select top 20 important features
importance = np.abs(shap_values).mean(axis=0)
top_indices = np.argsort(importance)[-20:][::-1]
top_features = [selected_feature_names[i] for i in top_indices]
X_test_top = X_test_filtered[:, top_indices]
shap_values_top = shap_values[:, top_indices]

print("Top 20 most important features:")
for i, (idx, feat_name) in enumerate(zip(top_indices, top_features)):
    print(f"   {i + 1:2d}. {feat_name} (Importance: {importance[idx]:.6f})")

# Plot SHAP Summary
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values_top, X_test_top, feature_names=top_features, show=False)

ax = plt.gca()
fig = plt.gcf()

ax.set_xlabel("SHAP value (impact on model output)", fontsize=18, fontname="Times New Roman")
ax.tick_params(axis='x', which='major', direction='out', length=6, width=1.2, labelsize=16)
ax.tick_params(axis='y', which='major', labelsize=16)
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.axvline(0, color='gray', linewidth=1.2, linestyle='-')

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontsize(16)

for text in ax.texts:
    text.set_fontname("Times New Roman")
    text.set_fontsize(16)

for color_ax in fig.axes:
    if color_ax != ax:
        for tick in color_ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(16)
        for text in color_ax.texts:
            text.set_fontname("Times New Roman")
            text.set_fontsize(16)
        for label in color_ax.get_xticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(16)
        if hasattr(color_ax, 'yaxis') and color_ax.get_ylabel():
            color_ax.set_ylabel(color_ax.get_ylabel(), fontname="Times New Roman", fontsize=16)

plt.tight_layout()
plt.savefig("results/lgb_shap_summary.png", dpi=300, bbox_inches='tight')
plt.show()

# ====================== SHAP Bar Plot ========================
mean_shap = np.mean(np.abs(shap_values_top), axis=0)
colors_bar = ['steelblue'] * len(mean_shap)

fig, ax = plt.subplots(figsize=(8, 9), facecolor='white')
y_pos = np.arange(len(top_features))
ax.barh(y_pos, mean_shap[::-1], color=colors_bar[::-1], height=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_features[::-1], fontsize=16)
ax.tick_params(axis='y', left=False)
ax.set_xlabel("Mean |SHAP| Value", labelpad=10, fontsize=18)

ax.xaxis.set_major_locator(MultipleLocator(0.04))
ax.tick_params(axis='x', which='major', direction='out', length=6, width=1.2, labelsize=16)
ax.tick_params(axis='x', which='minor', direction='out', length=3, width=1.0)

ax.set_facecolor('white')
ax.grid(False)
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_linewidth(1.2)

plt.tight_layout()
plt.savefig("results/lgb_shap_bar.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")

# ====================== Save Feature Selection Info ========================
selected_features_info = pd.DataFrame({
    'Feature_Index': selected_indices,
    'Feature_Name': selected_feature_names,
    'Variance': np.var(X_train_full[:, selected_indices], axis=0)
})
selected_features_info.to_csv("results/lgb_selected_features_info.csv", index=False)
print("Feature selection info saved.")

# ====================== 1. Save Scatter Plot Data ========================
scatter_df = pd.DataFrame({
    "Set": ["Train"] * len(y_train_full) + ["Test"] * len(y_test),
    "Experimental Value": np.concatenate([y_train_full, y_test]),
    "Predicted Value": np.concatenate([y_train_pred, y_test_pred])
})
scatter_df.to_csv("results/lgb_scatter_data.csv", index=False)
print("Scatter plot data saved.")

# ====================== 2. Save Residual Data ========================
residuals_df = pd.DataFrame({
    "Set": ["Train"] * len(y_train_full) + ["Test"] * len(y_test),
    "Experimental Value": np.concatenate([y_train_full, y_test]),
    "Predicted Value": np.concatenate([y_train_pred, y_test_pred]),
    "Residual": np.concatenate([train_residuals, test_residuals])
})
residuals_df.to_csv("results/lgb_residual_data.csv", index=False)
print("Residual data saved.")

# ====================== 3. Test Set Residuals Boxplot ========================
plt.figure(figsize=(6, 5))
sns.boxplot(y=test_residuals, color='lightblue', width=0.4)
plt.axhline(0, color='black', linestyle='--', linewidth=1.2)
plt.ylabel('Residuals', fontsize=16)
plt.title('Test Set Residuals Boxplot', fontsize=16)
plt.tick_params(axis='y', labelsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("results/lgb_test_residuals_boxplot.png", dpi=300, bbox_inches='tight')
plt.show()

pd.DataFrame({"Test Residual": test_residuals}).to_csv(
    "results/lgb_test_residuals_boxplot.csv", index=False)
print("Test set residuals boxplot and data saved.")

# ====================== 4. Correlation Matrix of Top 20 Features ========================
top_n = 20
X_train_df = pd.DataFrame(X_train_filtered, columns=selected_feature_names)
feature_corr_with_pred = X_train_df.apply(lambda col: np.corrcoef(col, y_train_pred)[0, 1])
top_features_corr = feature_corr_with_pred.abs().sort_values(ascending=False).head(top_n).index.tolist()
top_features_df = X_train_df[top_features_corr]
correlation_matrix_top = top_features_df.corr()

correlation_matrix_top.to_csv(f"results/lgb_top{top_n}_features_correlation_matrix.csv")
print(f"Top {top_n} features correlation matrix saved.")

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_top,
            cmap='coolwarm',
            annot=False,
            fmt=".2f",
            cbar_kws={"shrink": 0.8})
plt.title(f"Top {top_n} Features Correlation Matrix", fontsize=16)
plt.tight_layout()
plt.savefig(f"results/lgb_top{top_n}_features_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"Top {top_n} features correlation matrix heatmap saved.")

print("\nAll tasks completed!")
print(f"   Original feature count: {X.shape[1]}")
print(f"   Selected feature count: {X_train_filtered.shape[1]}")
print(f"   Reduction ratio: {(X.shape[1] - X_train_filtered.shape[1]) / X.shape[1] * 100:.1f}%")
print(f"   Removed {zero_var_count} zero-variance features.")
