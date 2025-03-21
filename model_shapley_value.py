# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:07:41 2025

@author: Movement Rehab Lab
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import shap
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(42)

# 数据加载与预处理（保持原样）
clinical_scale = pd.read_csv('F:/response_model/clinical_scale.csv')
EEG_feature = pd.read_csv('F:/response_model/EEG_feature.csv')
fc_feature = pd.read_csv('F:/20250306response/fc_outcomes.csv')

exclude_columns0 = ["ID", "name","UPDRS","response"]
exclude_columns1 = ["ID", "response"]

X = pd.concat([clinical_scale.drop(columns=exclude_columns0), 
               EEG_feature.drop(columns=exclude_columns1)], axis=1)
feature_names = X.columns.tolist()  # 保存特征名称
X = X.values
y = clinical_scale.iloc[:, 3].values

# 定义分类模型列表（保持原样）
models = [
    ('Linear SVM', make_pipeline(
        StandardScaler(),
        SVC(kernel='linear', probability=True, random_state=42,C=10)
    )),
    ('RBF SVM', make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', probability=True, random_state=42)
    )),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Stochastic GB', GradientBoostingClassifier(
        subsample=0.8,
        random_state=42
    )),
    ('XGBoost', XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )),
]

# 初始化结果存储结构
results = []
shap_dict = {}  # 存储各模型的SHAP值

# 定义5折交叉验证
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 遍历所有模型
for model_name, model in models:
    print(f"\n{'='*50}\nProcessing {model_name}\n{'='*50}")
    
    fold_shap_values = []  # 存储当前模型各折的SHAP值
    fold_test_indices = []  # 存储测试集索引
    
    # 手动实现交叉验证循环
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"\nFold {fold_idx+1}/5")
        
        # 划分数据
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 计算SHAP值（根据模型类型选择解释器）
        if 'SVM' in model_name:
            # 对SVM使用KernelExplainer（注意：计算较慢）
            explainer = shap.KernelExplainer(model.predict_proba, X_train)
            shap_values = explainer.shap_values(X_test)[:,:,1]  # 取正类的SHAP值
        elif 'Random Forest' in model_name:
            # 对树模型使用TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)[:,:,1]  # 取正类的SHAP值
        else:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)  # 取正类的SHAP值
        # 保存当前折的SHAP值和索引
        fold_shap_values.append(shap_values)
        fold_test_indices.append(test_idx)
    
    # 合并所有折的SHAP值
    all_test_indices = np.concatenate(fold_test_indices)
    all_shap_values = np.concatenate(fold_shap_values)
    
    # 按原始数据顺序重新排列SHAP值
    sorted_indices = np.argsort(all_test_indices)
    final_shap_values = all_shap_values[sorted_indices]
    
    # 存储到字典
    shap_dict[model_name] = {
        'values': final_shap_values,
        'feature_names': feature_names
    }
    
    # # 可视化当前模型的SHAP摘要图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(final_shap_values, X, feature_names=feature_names, show=False)
    plt.title(f"SHAP Summary - {model_name}")
    plt.tight_layout()
    plt.show()

# 后续分析示例：可以访问shap_dict进行各模型的SHAP分析
# 例如：shap_dict['Random Forest']['values'] 包含随机森林的合并SHAP值

# （原评估指标计算部分需要调整，此处省略，保持原逻辑即可）