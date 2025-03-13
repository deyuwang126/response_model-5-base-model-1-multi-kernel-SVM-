# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:03:46 2025

@author: Movement Rehab Lab
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# # 1️⃣ 读取数据
# data = pd.read_csv("your_data.csv")

# # 2️⃣ 选择特征
# exclude_cols = ["name", "UPDRS"]
# feature_cols = [col for col in data.columns if col not in exclude_cols]

# # 假设 EEG 数据在前 5 列，其他数据在后面
num_eeg_features = 5  
# X = data[feature_cols].values
# y = data["label"].values
X = np.random.rand(200, 10)  # 特征矩阵
y = np.random.randint(0, 2, 200)  # 随机二分类标签
# 3️⃣ 定义5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 4️⃣ 存储每折的结果
results = []

# 5️⃣ 遍历每一折
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\n{'='*50}\nFold {fold + 1}\n{'='*50}")
    
    # 划分训练集和测试集
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 6️⃣ 标准化数据
    scaler_eeg = StandardScaler()
    scaler_motion = StandardScaler()
    
    # 标准化训练数据
    X_eeg_train = scaler_eeg.fit_transform(X_train[:, :num_eeg_features])
    X_motion_train = scaler_motion.fit_transform(X_train[:, num_eeg_features:])
    
    # 标准化测试数据
    X_eeg_test = scaler_eeg.transform(X_test[:, :num_eeg_features])
    X_motion_test = scaler_motion.transform(X_test[:, num_eeg_features:])
    
    # 7️⃣ 计算 RBF 核
    gamma_eeg = 1.0 / X_eeg_train.shape[1]
    gamma_motion = 1.0 / X_motion_train.shape[1]
    
    # 训练数据的核矩阵
    kernel_eeg_train = rbf_kernel(X_eeg_train, gamma=gamma_eeg)
    kernel_motion_train = rbf_kernel(X_motion_train, gamma=gamma_motion)
    
    # 测试数据的核矩阵
    kernel_eeg_test = rbf_kernel(X_eeg_test, X_eeg_train, gamma=gamma_eeg)
    kernel_motion_test = rbf_kernel(X_motion_test, X_motion_train, gamma=gamma_motion)
    
    # 8️⃣ 多核融合
    w_eeg = 0.6  # EEG 核权重
    w_motion = 0.4  # 运动数据核权重
    
    K_fused_train = w_eeg * kernel_eeg_train + w_motion * kernel_motion_train
    K_fused_test = w_eeg * kernel_eeg_test + w_motion * kernel_motion_test
    
    # 9️⃣ 训练 SVM
    svm_model = SVC(kernel="precomputed", C=1.0)
    svm_model.fit(K_fused_train, y_train)
    
    # 🔟 预测
    y_pred = svm_model.predict(K_fused_test)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # 保存结果
    results.append(accuracy)

# 汇总结果
print("\n\n5-Fold Cross-Validation Results:")
print(f"Mean Accuracy: {np.mean(results):.4f} (±{np.std(results):.4f})")