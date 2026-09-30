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

# 读取数据
df = pd.read_csv('D:/SU/date/flammable/input-classify.csv')

# 特征和标签
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 将标签只保留三类：不燃液体(0), 高度易燃液体(1), 易燃液体(2)
valid_classes = [0, 1, 2]
mask = np.isin(y, valid_classes)
X = X[mask]
y = y[mask]

# 计算每个特征与目标变量的皮尔逊相关系数
correlations = []
for i in range(X.shape[1]):
    corr = np.corrcoef(X[:, i], y)[0, 1]
    correlations.append((i, corr))

# 选择相关性最大的k个特征
k = min(135, X.shape[1])
correlations.sort(key=lambda x: abs(x[1]), reverse=True)
top_k_features = [x[0] for x in correlations[:k]]
X_selected = X[:, top_k_features]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# 划分数据集
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# 定义模型与参数网格
svc = SVC(probability=True,random_state=42)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['linear', 'rbf']
}

# 网格搜索
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"最佳超参数: {grid_search.best_params_}")
best_svc = grid_search.best_estimator_

# 交叉验证
cv_scores = cross_val_score(best_svc, X_train, y_train, cv=5, scoring='accuracy')
print(f"最佳交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 模型训练
best_svc.fit(X_train, y_train)

# 预测
y_train_pred = best_svc.predict(X_train)
y_val_pred = best_svc.predict(X_val)
y_test_pred = best_svc.predict(X_test)

# 错误样本分析
X_test_df = pd.DataFrame(X_test, columns=[f'Feature_{i+1}' for i in range(X_test.shape[1])])
incorrect_indices = np.where(y_test != y_test_pred)[0]
incorrect_data = X_test_df.iloc[incorrect_indices].copy()
incorrect_data['True Label'] = y_test[incorrect_indices]
incorrect_data['Predicted'] = y_test_pred[incorrect_indices]

# 数据集大小
print(f"训练集大小: {X_train.shape[0]}")
print(f"验证集大小: {X_val.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 精度评估
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"训练集准确性: {train_accuracy:.4f}")
print(f"验证集准确性: {val_accuracy:.4f}")
print(f"测试集准确性: {test_accuracy:.4f}")

# 其他指标
precision = precision_score(y_test, y_test_pred, average='weighted')
recall = recall_score(y_test, y_test_pred, average='weighted')
f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"精确度: {precision:.2f}")
print(f"召回率: {recall:.2f}")
print(f"F1 分数: {f1:.2f}")

# 分类报告
label_names = ['不燃液体0', '高度易燃液体1', '易燃液体2']
print("\n分类报告 (测试集):")
print(classification_report(y_test, y_test_pred, target_names=label_names))

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_test_pred)
print("混淆矩阵:")
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

plt.rcParams['font.family'] = 'Times New Roman'  # 全局设置字体为 Times New Roman

# -------------------- SHAP 计算及堆叠条形图 --------------------

num_show_features = 20
feature_indices_show = top_k_features[:num_show_features]

# 提取用于展示的测试集特征（前20个特征）
X_test_show = X_test[:, :num_show_features]

# 原始英文特征名
feature_names_show = [df.columns[idx] for idx in feature_indices_show]

# 包装模型预测函数，使其支持仅前20维特征作为输入
def model_predict_proba(X_subset):
    X_full = np.zeros((X_subset.shape[0], X_train.shape[1]))
    X_full[:, :num_show_features] = X_subset
    return best_svc.predict_proba(X_full)

# 构建 KernelExplainer
explainer = shap.KernelExplainer(
    model_predict_proba,
    shap.kmeans(X_train[:, :num_show_features], 10)
)

# 计算 SHAP 值
shap_values = explainer.shap_values(X_test_show, nsamples=100)

# 平均绝对 SHAP 值，按类别分别求均值
mean_abs_shap_per_class = np.array([
    np.mean(np.abs(class_shap), axis=0) for class_shap in shap_values
])  # shape: (3, 20)

# 计算每个特征的整体重要性（3类加和）
total_importance = np.sum(mean_abs_shap_per_class, axis=0)

# 排序索引（按重要性从高到低）
sorted_indices = np.argsort(total_importance)[::-1]

# 重新排序 SHAP 值、特征名
mean_abs_shap_per_class_sorted = mean_abs_shap_per_class[:, sorted_indices]
feature_names_sorted = [feature_names_show[i] for i in sorted_indices]

# 倒序索引（用于从上到下显示）
reversed_idx = list(range(num_show_features))[::-1]

# 绘制堆叠条形图
fig, ax = plt.subplots(figsize=(8, 6))
left = np.zeros(num_show_features)

colors = ['#2d5c87', '#D75615', '#C28B00']
labels = ['NFL ', 'HFL', 'FL']

for i in range(len(shap_values)):
    ax.barh(reversed_idx, mean_abs_shap_per_class_sorted[i],
            left=left, color=colors[i], label=labels[i])
    left += mean_abs_shap_per_class_sorted[i]

# 设置 Y 轴标签为倒序排列的特征名
ax.set_yticks(reversed_idx)
ax.set_yticklabels([feature_names_sorted[i] for i in range(num_show_features)], fontsize=10.5)

# 去除 y 轴向外凸起的小刻度线
ax.tick_params(axis='y', length=0)

# 设置 X 轴
ax.set_xlabel('Mean |SHAP value|', fontsize=12)
ax.tick_params(axis='x', labelsize=10.5)

# 图例设置
ax.legend(loc='lower right', fontsize=10)

# 调整 y 轴显示范围，让上下更紧凑
ax.set_ylim(-1, num_show_features)  # -1 比 0 更贴近底部，num_show_features 保持顶部紧凑

# 调整子图边距
plt.tight_layout(pad=0.5)  # pad 控制整体紧凑程度
plt.show()