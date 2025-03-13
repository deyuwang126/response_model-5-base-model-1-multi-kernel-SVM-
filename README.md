Multi-Source Feature Classification for Predicting Response Groups

Overview

In this analysis, we developed a classification model to predict patient response (low-response vs. high-response) based on different feature sources, including EEG features and motion features. The response variable is defined as (POST - PRE) UPDRS, and the classification target is binary (y=0 for low response, y=1 for high response).

Input Data

The input dataset contains various features from multiple sources. Specifically:

Features excluding: 'ID', 'name', and 'UPDRS'

EEG data: first 5 columns of the dataset

Motion features: remaining columns

Labels: Binary labels (0 = low response, 1 = high response)

Classification Models Used

We evaluated multiple classification algorithms for this task:

Linear SVM (Support Vector Machine)

RBF (Radial Basis Function) SVM

Random Forest (RF)

Stochastic Gradient Boosting (GBM)

Extreme Gradient Boosting (XGBoost)

Multi-Kernel Learning SVM (MKL-SVM)

The MKL-SVM approach was particularly important since we integrated features from multiple sources (EEG and motion data) with different feature dimensions. The steps included:

Preprocessing: Standardization of EEG and motion data separately.

Kernel Computation:

Compute RBF kernel for EEG features and motion features separately.

Use a weighted sum of these individual kernels to form a fused kernel.

Multi-Kernel SVM Training:

Train an SVM with a precomputed kernel using the combined kernel matrix.

Perform cross-validation (LOOCV) for evaluation.

Handling Different Feature Dimensions

Since EEG and motion data have different feature dimensions ( for EEG and  features for motion data), SVM’s multi-kernel learning is affected. The solution is:

Apply a separate kernel function to each feature set.

Normalize features (e.g., using StandardScaler).

Compute separate kernel matrices for EEG and motion features.

Apply a weighted combination: fused_kernel = w1 × K_eeg + w_motion × K_motion.

Use the fused kernel in the SVM model.

Alternative Approach: Multi-Model Fusion with XGBoost

For more complex cases or larger datasets, we also explored a deep learning-based fusion method:

Train an MLP model on EEG features and extract intermediate feature representations.

Train a Random Forest model on motion features and extract class probabilities.

Concatenate features from both sources.

Use Extreme Gradient Boosting (XGBoost) as the final classifier.

Evaluate the model using accuracy, classification reports, and Kfold.
