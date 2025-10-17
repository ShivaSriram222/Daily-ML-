# Common evaluation metrics Confusion matrix: This table helps visualize the performance of a classification model. From it, you can calculate:Accuracy: The percentage of all predictions the model got right. While a good starting point, it can be misleading with imbalanced datasets.Precision: The percentage of positive predictions that were actually correct. This is important when the cost of a false positive is high.Recall (Sensitivity): The percentage of actual positive cases that were correctly identified. This is crucial when the cost of a false negative is high.F1-Score: The harmonic mean of precision and recall. It is especially useful for imbalanced datasets.ROC curve and AUC:Receiver Operating Characteristic (ROC) curve: A graph showing the model's performance at all possible classification thresholds.Area Under the Curve (AUC): A metric that summarizes the ROC curve, representing the model's ability to distinguish between the positive and negative classes. A value of 1.0 is a perfect model, while 0.5 is no better than random guessing.Log Loss: Also known as cross-entropy loss, this metric measures the performance of a classifier where the prediction is a probability. The lower the log loss, the better the model's predictions.Statistical significance: You can evaluate the statistical significance of individual predictors using p-values and odds ratios.p-values: A low p-value (typically \(<0.05\)) for a variable suggests it has a meaningful impact on the outcome and is statistically significant.Odds ratio: Represents the change in odds of the outcome for a one-unit change in the predictor variable. For example, an odds ratio of 1.3 means a one-unit increase in the predictor increases the odds of the outcome by 30%. 


import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score

# Load California Housing Dataset
data = fetch_california_housing(as_frame=True)
df = data.frame

# Convert target to binary for classification
threshold = df['MedHouseVal'].median()
df['MedHouseVal_bin'] = (df['MedHouseVal'] > threshold).astype(int)

X = df.drop(['MedHouseVal', 'MedHouseVal_bin'], axis=1)
y = df['MedHouseVal_bin']

# Split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression without scaling
logreg_no_scaling = LogisticRegression(max_iter=1000)
logreg_no_scaling.fit(X_train, y_train)
y_pred_no_scaling = logreg_no_scaling.predict(X_test)
acc_no_scaling = accuracy_score(y_test, y_pred_no_scaling)
print(f"No Scaling Accuracy: {acc_no_scaling:.4f}")

# Logistic Regression with Min-Max Normalization
scaler_minmax = MinMaxScaler()
X_train_norm = scaler_minmax.fit_transform(X_train)
X_test_norm = scaler_minmax.transform(X_test)
logreg_norm = LogisticRegression(max_iter=1000)
logreg_norm.fit(X_train_norm, y_train)
y_pred_norm = logreg_norm.predict(X_test_norm)
acc_norm = accuracy_score(y_test, y_pred_norm)
print(f"Min-Max Normalization Accuracy: {acc_norm:.4f}")

# Logistic Regression with Standardization (Z-score)
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)
logreg_std = LogisticRegression(max_iter=1000)
logreg_std.fit(X_train_std, y_train)
y_pred_std = logreg_std.predict(X_test_std)
acc_std = accuracy_score(y_test, y_pred_std)
print(f"Standardization Accuracy: {acc_std:.4f}")

# Summary

# Logistic Regression only needs the relative relationships between features, not absolute scales.
# If all features are roughly in the same range, scaling won’t drastically change the decision boundary.




# SVM 
from sklearn.svm import SVC

# SVM without scaling
svm_no_scaling = SVC(kernel='rbf', C=1, gamma='scale')
svm_no_scaling.fit(X_train,y_train)
y_pred_no_scaling = svm_no_scaling.predict(X_test)
acc_no_scaling = accuracy_score(y_test, y_pred_no_scaling)
print(f"No Scaling Accuracy: {acc_no_scaling:.4f}")



# SVM with Min-Max Normalization

scaler_minmax = MinMaxScaler()
X_train_norm = scaler_minmax.fit_transform(X_train)
X_test_norm = scaler_minmax.transform(X_test)
svm_norm =  SVC(kernel='rbf', C=1, gamma='scale')
svm_norm.fit(X_train_norm, y_train)
y_pred_norm = svm_norm.predict(X_test_norm)
acc_norm = accuracy_score(y_test, y_pred_norm)
print(f"Min-Max Normalization Accuracy: {acc_norm:.4f}")


# SVM with Standardization (Z-score)

scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)
svm_std =  SVC(kernel='rbf', C=1, gamma='scale')
svm_std.fit(X_train_std, y_train)
y_pred_std = svm_std.predict(X_test_std)
acc_std = accuracy_score(y_test, y_pred_std)
print(f"Standardization Accuracy: {acc_std:.4f}")

# Summary

# No scaling (0.5174): SVM struggles because features with larger magnitudes dominate the kernel calculation, so the decision boundary is skewed.
# Min-Max (0.8241): Rescaling to [0, 1] balances feature contributions; SVM finds a much better boundary.
# Standardization (0.8602): Centering and scaling to unit variance further improves stability for gradient-based optimization; gives the best performance here.