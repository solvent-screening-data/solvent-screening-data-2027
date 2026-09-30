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
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# 全局字体设置
plt.rcParams['font.family'] = 'Times New Roman'

# ---------------------- 数据读取与处理 ----------------------
start_time = time.time()
df = pd.read_csv('D:/SU/date/feature-date/input-total.csv')
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
feature_names = df.columns[1:].tolist()

print(f"✅ 数据读取完成")
print(f"   原始数据形状: {df.shape} (样本数×特征数+1)")
print(f"   总特征数: {X.shape[1]}")

# 数据划分
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ 数据划分完成")
print(f"   训练集形状: {X_train_full.shape}")
print(f"   测试集形状: {X_test.shape}")

# ====================== 特征选择（改进版） ======================
print(f"\n🔧 开始特征选择...")

# 1. 计算方差
feature_variances = np.var(X_train_full, axis=0)

# 2. 先去掉零方差特征
non_zero_indices = np.where(feature_variances > 0)[0]
zero_var_count = X_train_full.shape[1] - len(non_zero_indices)
print(f"   零方差特征数: {zero_var_count}")
print(f"   非零方差特征数: {len(non_zero_indices)}")

if zero_var_count > 0 and zero_var_count <= 10:
    zero_var_names = [feature_names[i] for i in np.where(feature_variances == 0)[0][:10]]
    print(f"   零方差特征示例: {zero_var_names}")

# 3. 在非零特征中选择前90%
if len(non_zero_indices) > 0:
    non_zero_variances = feature_variances[non_zero_indices]
    var_threshold = np.percentile(non_zero_variances, 10)  # 第10百分位数
    print(f"   非零特征的第10百分位数: {var_threshold:.10f}")

    # 选择方差大于阈值的特征（使用 >= 避免浮点数精度问题）
    epsilon = 1e-12
    selected_mask = non_zero_variances >= var_threshold - epsilon
    selected_indices = non_zero_indices[selected_mask]

    # 安全验证
    if len(selected_indices) == 0:
        print("⚠️ 警告：没有特征被选中，使用所有非零方差特征")
        selected_indices = non_zero_indices
else:
    print("⚠️ 所有特征方差都为0，使用所有特征")
    selected_indices = np.arange(X_train_full.shape[1])

# 确保至少有一些特征被选中
if len(selected_indices) == 0:
    print("❌ 严重错误：特征选择失败，使用前100个特征")
    selected_indices = np.arange(min(100, X_train_full.shape[1]))

print(f"✅ 特征选择完成:")
print(f"   原始特征数: {X_train_full.shape[1]}")
print(f"   选中特征数: {len(selected_indices)}")
print(f"   特征减少比例: {(X_train_full.shape[1] - len(selected_indices)) / X_train_full.shape[1] * 100:.1f}%")
print(f"   移除了 {zero_var_count} 个零方差特征")

# 应用特征选择
X_train_filtered = X_train_full[:, selected_indices]
X_test_filtered = X_test[:, selected_indices]
selected_feature_names = [feature_names[i] for i in selected_indices]

# ====================== 特征质量检查 ======================
print(f"\n🔍 特征质量检查:")
selected_variances = np.var(X_train_filtered, axis=0)
print(f"   选中特征的方差范围: [{selected_variances.min():.10f}, {selected_variances.max():.10f}]")
print(f"   选中特征的方差均值: {selected_variances.mean():.10f}")

# 检查选中特征是否还有零方差（应该没有）
remaining_zero_var = np.sum(selected_variances == 0)
if remaining_zero_var > 0:
    print(f"   ⚠️ 警告：选中的特征中还有 {remaining_zero_var} 个零方差特征")


# ====================== 继续原有代码 ======================

# ---------------------- Optuna 超参数优化 ----------------------
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


print(f"\n🔍 开始Optuna超参数优化...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100, timeout=3600)  # 超参数搜索1小时或100次
best_params = study.best_params
print(f"\n✅ [最佳超参数] {best_params}")

# ---------------------- 模型训练与评估 ----------------------
print(f"\n🔍 开始模型训练...")
final_model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
final_model.fit(X_train_filtered, y_train_full)

print(f"✅ 模型训练完成")
print(f"   实际使用的树的数量: {final_model.n_estimators_}")


def evaluate(model, X, y, name):
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    print(f"\n📊 [{name}] 评估结果:")
    print(f"   MSE = {mse:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   MAE = {mae:.4f}")
    print(f"   R² = {r2:.4f}")
    return pred


y_train_pred = evaluate(final_model, X_train_filtered, y_train_full, '训练集')
y_test_pred = evaluate(final_model, X_test_filtered, y_test, '测试集')

# ====================== 实验值 vs 预测值图（合并） ========================
print(f"\n📈 生成可视化图表...")

plt.figure(figsize=(6, 6))

# 单独定义颜色
train_color = '#1f77b4'  # 蓝色
test_color = '#ff7f0e'  # 橙色

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
plt.savefig("D:/SU/date/solubility-tu/lgb_scatter_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# ====================== 折线图 ========================
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
plt.savefig("D:/SU/date/solubility-tu/lgb_line_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# ====================== 残差图 ========================
train_residuals = y_train_full - y_train_pred
test_residuals = y_test - y_test_pred

plt.figure(figsize=(6, 4))

# 单独定义颜色
train_color = '#1f77b4'  # 蓝色
test_color = '#ff7f0e'  # 橙色

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
plt.savefig("D:/SU/date/solubility-tu/lgb_residual_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# ====================== SHAP Summary Plot ========================
print(f"\n🔍 计算SHAP值...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test_filtered)

# 取前 20 个最重要特征
importance = np.abs(shap_values).mean(axis=0)
top_indices = np.argsort(importance)[-20:][::-1]
top_features = [selected_feature_names[i] for i in top_indices]
X_test_top = X_test_filtered[:, top_indices]
shap_values_top = shap_values[:, top_indices]

print(f"✅ 最重要的20个特征:")
for i, (idx, feat_name) in enumerate(zip(top_indices, top_features)):
    print(f"   {i + 1:2d}. {feat_name} (重要性: {importance[idx]:.6f})")

# 绘制 SHAP Summary Plot
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values_top, X_test_top, feature_names=top_features, show=False)

# 获取主图坐标轴和图像对象
ax = plt.gca()
fig = plt.gcf()

# 设置 x 和 y 轴标签
ax.set_xlabel("SHAP value (impact on model output)", fontsize=18, fontname="Times New Roman")

# 设置坐标轴刻度
ax.tick_params(axis='x', which='major', direction='out', length=6, width=1.2, labelsize=16)
ax.tick_params(axis='y', which='major', labelsize=16)
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.axvline(0, color='gray', linewidth=1.2, linestyle='-')

# 强制设置坐标轴刻度字体
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname("Times New Roman")
    label.set_fontsize(16)

# 设置 y 轴特征名字体
for text in ax.texts:
    text.set_fontname("Times New Roman")
    text.set_fontsize(16)

# 修改右侧 colorbar（颜色条）的字体，包括标签、"High"/"Low"、刻度
for color_ax in fig.axes:
    if color_ax != ax:
        # 修改 colorbar 刻度字体
        for tick in color_ax.get_yticklabels():
            tick.set_fontname("Times New Roman")
            tick.set_fontsize(16)

        # 修改 colorbar 中的 "High"/"Low" 和 "Feature value"
        for text in color_ax.texts:
            text.set_fontname("Times New Roman")
            text.set_fontsize(16)

        # 设置 colorbar 横向坐标刻度（有时也会出现）
        for label in color_ax.get_xticklabels():
            label.set_fontname("Times New Roman")
            label.set_fontsize(16)

        # 显式设置 colorbar label（Feature value）字体
        if hasattr(color_ax, 'yaxis') and color_ax.get_ylabel():
            color_ax.set_ylabel(color_ax.get_ylabel(), fontname="Times New Roman", fontsize=16)

plt.tight_layout()
plt.savefig("D:/SU/date/solubility-tu/lgb_shap_summary.png", dpi=300, bbox_inches='tight')
plt.show()

# ====================== SHAP 柱状图 ========================
# 取 SHAP 值绝对值的均值
mean_shap = np.mean(np.abs(shap_values_top), axis=0)
colors_bar = ['steelblue'] * len(mean_shap)  # 全部设为蓝色

fig, ax = plt.subplots(figsize=(8, 9), facecolor='white')
y_pos = np.arange(len(top_features))
bars = ax.barh(y_pos, mean_shap[::-1], color=colors_bar[::-1], height=0.8)

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
plt.savefig("D:/SU/date/solubility-tu/lgb_shap_bar.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"\n[运行时间] 总耗时: {time.time() - start_time:.2f} 秒")

# ====================== 保存特征选择信息 ========================
# 保存选中的特征信息
selected_features_info = pd.DataFrame({
    'Feature_Index': selected_indices,
    'Feature_Name': selected_feature_names,
    'Variance': np.var(X_train_full[:, selected_indices], axis=0)
})
selected_features_info.to_csv("D:/SU/date/solubility-tu/lgb_selected_features_info.csv", index=False)
print("✅ 特征选择信息已保存")

# ====================== 1. 保存散点图数据 ========================
scatter_df = pd.DataFrame({
    "Set": ["Train"] * len(y_train_full) + ["Test"] * len(y_test),
    "Experimental Value": np.concatenate([y_train_full, y_test]),
    "Predicted Value": np.concatenate([y_train_pred, y_test_pred])
})
scatter_df.to_csv("D:/SU/date/solubility-tu/lgb_sandian_data-shanchu.csv", index=False)
print("✅ 散点图数据已保存")

# ====================== 2. 保存残差数据 ========================
residuals_df = pd.DataFrame({
    "Set": ["Train"] * len(y_train_full) + ["Test"] * len(y_test),
    "Experimental Value": np.concatenate([y_train_full, y_test]),
    "Predicted Value": np.concatenate([y_train_pred, y_test_pred]),
    "Residual": np.concatenate([train_residuals, test_residuals])
})
residuals_df.to_csv("D:/SU/date/solubility-tu/lgb_cancha_data-shanchu.csv", index=False)
print("✅ 残差数据已保存")

# ====================== 3. 测试集残差箱线图 ========================
plt.figure(figsize=(6, 5))
sns.boxplot(y=test_residuals, color='lightblue', width=0.4)
plt.axhline(0, color='black', linestyle='--', linewidth=1.2)
plt.ylabel('Residuals', fontsize=16)
plt.title('Test Set Residuals Boxplot', fontsize=16)
plt.tick_params(axis='y', labelsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("D:/SU/date/solubility-tu/lgb-test_residuals_boxplot-shanchu.png", dpi=300, bbox_inches='tight')
plt.show()

# 保存测试集残差数据
pd.DataFrame({"Test Residual": test_residuals}).to_csv(
    "D:/SU/date/solubility-tu/lgb-test_residuals_boxplot-shanchu.csv", index=False)
print("✅ 测试集残差箱线图已保存")
print("✅ 测试集残差数据已保存")

# ====================== 4. 与预测值最相关的前20个特征相关性矩阵 ========================
top_n = 20

# 将训练集特征转成 DataFrame
X_train_df = pd.DataFrame(X_train_filtered, columns=selected_feature_names)

# 计算每个特征与 y_train_pred 的相关系数
feature_corr_with_pred = X_train_df.apply(lambda col: np.corrcoef(col, y_train_pred)[0, 1])

# 取绝对值最大的前 top_n 特征
top_features_corr = feature_corr_with_pred.abs().sort_values(ascending=False).head(top_n).index.tolist()

# 提取这 N 个特征
top_features_df = X_train_df[top_features_corr]

# 计算相关性矩阵
correlation_matrix_top = top_features_df.corr()

# 保存相关性矩阵为 CSV
correlation_matrix_top.to_csv(f"D:/SU/date/solubility-tu/lgb-top{top_n}_features_correlation_matrix.csv")
print(f"✅ 前 {top_n} 个特征相关性矩阵已保存")

# 绘制热图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_top,
            cmap='coolwarm',
            annot=False,
            fmt=".2f",
            cbar_kws={"shrink": 0.8})
plt.title(f"Top {top_n} Features Correlation Matrix", fontsize=16)
plt.tight_layout()
plt.savefig(f"D:/SU/date/solubility-tu/lgb-top{top_n}_features_correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ 前 {top_n} 个特征相关性矩阵热图已保存")

print(f"\n🎉 所有任务完成！")
print(f"   原始特征数: {X.shape[1]}")
print(f"   选中特征数: {X_train_filtered.shape[1]}")
print(f"   特征减少比例: {(X.shape[1] - X_train_filtered.shape[1]) / X.shape[1] * 100:.1f}%")
print(f"   移除了 {zero_var_count} 个零方差特征")


