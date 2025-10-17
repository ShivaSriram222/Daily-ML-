# ---

# ## Target Leakage
# **Target leakage** is a specific and common subtype of data leakage.  
# It happens when one or more training features are **directly or indirectly derived from the target variable** or contain information about it that should only be known *after* the event being predicted.

# For example:
# - Predicting pneumonia: using a feature like `took_antibiotic_medicine` leaks the outcome, since patients take antibiotics *after* diagnosis.  
# - Predicting customer churn: including `number_of_service_calls_after_cancellation` gives away post-churn behavior.

# ---

# ## Common Types of Data Leakage

# | Leakage Type | Description | Example |
# |--------------|-------------|---------|
# | Train–Test Contamination | Applying preprocessing (e.g., scaling, imputation) on the **entire dataset before splitting**. The test set influences training indirectly. | Fitting `StandardScaler` on the full dataset before `train_test_split`. |
# | Temporal Leakage | Using **future information** to predict past or present outcomes. | In a time-series model, using 2021 data to predict 2020 values. |
# | Target Leakage | A feature includes information that is a **direct proxy for the target**. | Adding a column like `SalePrice + noise` when predicting house prices. |
# | Post-Event Feature Leakage | Using attributes that are only known **after** the prediction event. | Using “Sale Condition” to predict `SalePrice`. |
# | Aggregation Leakage | Creating features using **global dataset stats** that include test/future data. | “Average SalePrice by Neighborhood” computed over all rows. |

# ---

# ## Why It Matters
# Data leakage:
# - Inflates validation metrics (e.g., R² ≈ 0.99 instead of realistic ≈ 0.8).  
# - Misleads model selection and hyperparameter tuning.  
# - Leads to poor generalization and unreliable deployment results.

# ---

# ## Prevention Checklist
# - Always **split data first**, then perform preprocessing (scaling, encoding, imputation).  
# - For time-based data, use **temporal validation** (train on past, test on future).  
# - Avoid including **post-event** or **target-derived** features.  
# - When encoding categorical variables using target statistics, compute encodings **only on the training fold**.  
# - Validate with realistic setups — if you couldn’t have known the data at prediction time, don’t include it.

# ---

# Key takeaway:  
# Data leakage is the silent performance booster — it makes models look smart in notebooks but fails in production.  
# Always ask: “Would this feature be known when making a real-world prediction?”


#  Manually introducing leakage scenarios 

# Scenario 1: Target Correlation Leak Example

# California Housing Dataset

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import numpy as np

# 1. Load dataset
data = fetch_california_housing(as_frame=True)
df = data.frame
with pd.option_context('display.max_columns', None):
    print(df.head())

# 2. Create a target leakage feature
# We create a feature that has information from the target (price) + small random noise
np.random.seed(42)
df["LeakageFeature"] = df["MedHouseVal"] + np.random.normal(0, 0.1, size=len(df))

# 3. Separate features and target
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Simple preprocessing + model pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# 6. Train model
pipeline.fit(X_train, y_train)

# 7. Evaluate model
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(" With Target Correlation Leak")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# 8. Now remove the leak and compare
X_no_leak = X.drop(columns=["LeakageFeature"])
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_no_leak, y, test_size=0.2, random_state=42)

pipeline.fit(X_train2, y_train2)
y_pred2 = pipeline.predict(X_test2)

r2_clean = r2_score(y_test2, y_pred2)
mae_clean = mean_absolute_error(y_test2, y_pred2)

print("\n Without Leakage Feature")
print(f"R² Score: {r2_clean:.4f}")
print(f"MAE: {mae_clean:.4f}")



# === With Target Correlation Leak ===
# R² Score: 0.9925
# MAE: 0.0789

# === Without Leakage Feature ===
# R² Score: 0.5758
# MAE: 0.5332


# Scenario 2: Aggregated Future Info Leak — Data leakage through aggregation
    # Aggregations must be computed only on the training data, and then mapped onto the test data.

# Scenario 3: Domain Feature Leak — Indirect information leak

# Scenario 4: Post-processing Label Leak — Pipeline leakage
    # Post-processing label leak occurs when data transformations, feature selection, or dimensionality reduction are applied before splitting the dataset into training and test sets, and those transformations indirectly use information from the target in the test set.


# Scenario 5: Outlier Label Influence Leak — Label-influenced preprocessing


# Summary

# Operations SAFE to DO Before Splitting. -> Anything which is TARGET INDEPENDENT 

# | Operation                                         | Why Safe                  | Examples                                     |
# | ------------------------------------------------- | ------------------------- | -------------------------------------------- |
# | Drop irrelevant columns or IDs                    | Doesn’t affect target     | `CustomerID`, `RowNumber`                    |
# | Remove duplicates                                 | No target info needed     | Remove identical rows                        |
# | Basic cleaning                                    | Doesn’t use target info   | Correct typos, fix formatting                |
# | Feature engineering from independent features     | Only uses predictors      | Ratios, sums, differences between predictors |
# | Convert categorical labels to numbers arbitrarily | Does not reference target | Map `Yes/No` → `1/0`                         |
# | Exploratory analysis / visualizations             | No model training yet     | Histograms, boxplots, correlation matrices   |


# Operations That Must Be Done After Splitting. -> Anything which is TARGET DEPENDENT

# | Operation                              | Leakage Risk                                            | Safe Approach                                                 |
# | -------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------- |
# | Scaling / normalization                | Uses mean/std of entire data → test info leaks          | Fit scaler on `X_train` → transform both `X_train` & `X_test` |
# | Imputation (mean/median)               | If based on entire dataset, test info leaks             | Compute on `X_train` only → apply to `X_test`                 |
# | Target encoding / aggregation features | Encoding uses target values → direct leak               | Compute on `X_train` only → map to `X_test`                   |
# | Feature selection based on target      | Selecting features using full dataset reveals test info | Select features using `X_train` & `y_train` only              |
# | Dimensionality reduction (PCA, SVD)    | Captures variance directions of test data               | Fit on `X_train` → transform both sets                        |
# | Handling outliers based on target      | Removing/extreme values after seeing target             | Apply rules based on `X_train` only                           |