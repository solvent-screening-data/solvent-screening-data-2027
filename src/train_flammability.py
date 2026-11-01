import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_curve, auc)
import shap
import warnings

warnings.filterwarnings("ignore")
# Load dataset
df = pd.read_csv('D:/SU/date/flammable/input-classify.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

valid_classes = [0, 1, 2]
mask = np.isin(y, valid_classes)
X = X[mask]
y = y[mask]

# Feature selection based on Pearson correlation
correlations = []
for i in range(X.shape[1]):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    correlations.append((i, corr))


k = min(135, X.shape[1])
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
top_k_features = [x[0] for x in correlations[:k]]
X_selected = X[:, top_k_features]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Data splitting (Train 80%, Val 10%, Test 10%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# SVM Model and Grid Search
svc = SVC(probability=True,random_state=42)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}


grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"Best hyperparameters: {grid_search.best_params_}")
best_svc = grid_search.best_estimator_

# 5-fold cross-validation
cv_scores = cross_val_score(best_svc, X_train, y_train, cv=5, scoring='accuracy')
print(f"Best CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Model training
best_svc.fit(X_train, y_train)

# Predictions
y_train_pred = best_svc.predict(X_train)
y_val_pred = best_svc.predict(X_val)
y_test_pred = best_svc.predict(X_test)

# Dataset sizes
print(f"Train set size: {X_train.shape[0]}")
print(f"Val set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Accuracy evaluation
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Val Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Other metrics
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Classification report
label_names = ['NFL', 'HFL', 'FL']
print("\nClassification Report (Test set):")
print(classification_report(y_test, y_test_pred, target_names=label_names))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d',
            xticklabels=[f'Class {i}' for i in valid_classes],
            yticklabels=[f'Class {i}' for i in valid_classes],
            cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

plt.rcParams['font.family'] = 'Times New Roman'

# -------------------- SHAP Analysis --------------------

num_show_features = 20
feature_indices_show = top_k_features[:num_show_features]

# Extract top 20 features for SHAP visualization
X_test_show = X_test[:, :num_show_features]
feature_names_show = [df.columns[idx] for idx in feature_indices_show]

# Wrapper function to pad the 20-feature subset back to the original dimension for the trained model
def model_predict_proba(X_subset):
    X_full = np.zeros((X_subset.shape[0], X_train.shape[1]))
    X_full[:, :num_show_features] = X_subset
    return best_svc.predict_proba(X_full)

# Kernel SHAP explanation
explainer = shap.KernelExplainer(
    model_predict_proba,
    shap.kmeans(X_train[:, :num_show_features], 10)
)

shap_values = explainer.shap_values(X_test_show, nsamples=100)

# Calculate mean absolute SHAP values per class (shape: 3 x 20)
mean_abs_shap_per_class = np.array([
    np.mean(np.abs(class_shap), axis=0) for class_shap in shap_values
]) 

# Aggregate feature importance
total_importance = np.sum(mean_abs_shap_per_class, axis=0)

# Sort features by descending importance
sorted_indices = np.argsort(total_importance)[::-1]
mean_abs_shap_per_class_sorted = mean_abs_shap_per_class[:, sorted_indices]
feature_names_sorted = [feature_names_show[i] for i in sorted_indices]

# Reverse index for top-to-bottom display
reversed_idx = list(range(num_show_features))[::-1]

# Stacked horizontal bar plot
fig, ax = plt.subplots(figsize=(8, 6))
left = np.zeros(num_show_features)

colors = ['#2d5c87', '#D75615', '#C28B00']
labels = ['NFL', 'HFL', 'FL']

for i in range(len(shap_values)):
    ax.barh(reversed_idx, mean_abs_shap_per_class_sorted[i],
            left=left, color=colors[i], label=labels[i])
    left += mean_abs_shap_per_class_sorted[i]

ax.set_yticks(reversed_idx)
ax.set_yticklabels([feature_names_sorted[i] for i in range(num_show_features)], fontsize=10.5)
ax.tick_params(axis='y', length=0)
ax.set_xlabel('Mean |SHAP value|', fontsize=12)
ax.tick_params(axis='x', labelsize=10.5)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(-1, num_show_features)

plt.tight_layout(pad=0.5)
plt.show()
