# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:07:41 2025

@author: Movement Rehab Lab
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
from sklearn.pipeline import make_pipeline


# 设置随机种子
np.random.seed(42)

clinical_scale = pd.read_csv('F:/response_model/clinical_scale.csv')
# 假设数据集有 200 个样本，每个样本 10 维特征，二分类任务
exclude_columns = ["ID", "name","UPDRS","response"]  # Adjust based on your dataset

# Automatically select feature columns
X = clinical_scale.drop(columns=exclude_columns).values
y = clinical_scale.iloc[:, 3].values  # 二分类标签




# 定义分类模型列表（包含预处理管道）
models = [
    ('Linear SVM', make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', probability=True, random_state=42)
    )),
    ('RBF SVM', make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', probability=True, random_state=42)
    )),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Stochastic GB', GradientBoostingClassifier(
        subsample=0.8,  # 启用随机子采样
        random_state=42
    )),
    ('XGBoost', XGBClassifier(
        
        eval_metric='logloss',
        random_state=42
    )),
]


# 存储结果的列表
results = []

# 定义5折交叉验证
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 遍历所有模型进行训练和评估
for name, model in models:
    # 交叉验证评估
    cv_metrics = cross_validate(
        model,
        X, y,
        cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        return_train_score=False,
        n_jobs=-1  # 使用所有可用的CPU核心加速计算
    )
    
    # 计算平均指标
    test_metrics = {
        'Model': name,
        'Accuracy': np.mean(cv_metrics['test_accuracy']),
        'Precision': np.mean(cv_metrics['test_precision']),
        'Recall': np.mean(cv_metrics['test_recall']),
        'F1': np.mean(cv_metrics['test_f1']),
        'ROC AUC': np.mean(cv_metrics['test_roc_auc']),
        'CV Accuracy (Std)': np.std(cv_metrics['test_accuracy']),
        'CV Precision (Std)': np.std(cv_metrics['test_precision']),
        'CV Recall (Std)': np.std(cv_metrics['test_recall']),
        'CV F1 (Std)': np.std(cv_metrics['test_f1']),
        'CV ROC AUC (Std)': np.std(cv_metrics['test_roc_auc'])
    }
    
    # 保存结果
    results.append(test_metrics)
    
    # 打印详细报告
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print("Cross-Validation Metrics:")
    print(f"Accuracy: {test_metrics['Accuracy']:.4f} (±{test_metrics['CV Accuracy (Std)']:.4f})")
    print(f"Precision: {test_metrics['Precision']:.4f} (±{test_metrics['CV Precision (Std)']:.4f})")
    print(f"Recall: {test_metrics['Recall']:.4f} (±{test_metrics['CV Recall (Std)']:.4f})")
    print(f"F1: {test_metrics['F1']:.4f} (±{test_metrics['CV F1 (Std)']:.4f})")
    print(f"ROC AUC: {test_metrics['ROC AUC']:.4f} (±{test_metrics['CV ROC AUC (Std)']:.4f})")

# 转换为DataFrame并展示结果
results_df = pd.DataFrame(results)
print("\n\n汇总结果:")
print(results_df.round(4))

# 可视化结果（示例：比较交叉验证准确率）
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(results_df['Model'], results_df['Accuracy'], color='skyblue')
plt.xlabel('Accuracy')
plt.title('Model Comparison (5-Fold CV Accuracy)')
plt.show()