

## Titanic — Data Sanity, Exploration & Baseline Modeling

*(Challenge / Prep Notebook)*

---

### Load & Inspect

```python
import pandas as pd
data = pd.read_csv('/Users/shivasriram/Desktop/Daily ML/Excercise1/Titanic.csv')
data.head()

print("Columns:", data.columns)
print(data.describe(include='all'))

# Missing values
missing = data.isnull().sum()
print("Missing Values:\n", missing)
```

---

### Missing Values Visualization & Handling

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.xlabel('Features')
plt.show()

# Imputation
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Cabin'].fillna('Unknown', inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

print("After Handling:\n", data.isnull().sum())
```

---

### Data Distribution — Initial Feature Understanding

```python
# Numerical distributions
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
fig, axes = plt.subplots(2,2, figsize=(10,6))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(data[col], bins=15, edgecolor='black', color="#27982B")
    axes[i].set_title(f'Distribution of {col}')
plt.tight_layout(); plt.show()

# Categorical distributions
cat_cols = ['Sex', 'Pclass', 'Embarked', 'Survived']
fig, axes = plt.subplots(2,2, figsize=(10,6))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    sns.countplot(data=data, x=col, ax=axes[i], palette='Set2')
    axes[i].set_title(f'Count of {col}')
plt.tight_layout(); plt.show()
```

**Key Observations**

* Skewness & outliers inform scaling or log-transform decisions
* Encoding strategy choice (avoid high-cardinality dummy expansion)
* Helps spot potential imbalances (e.g., survival distribution)

---

### Correlation Analysis

```python
# Numerical correlation
corr_matrix = data[num_cols].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix (Numerical)')
plt.show()

# Encode for mixed correlation exploration
data_encoded = data.copy()
data_encoded['Sex'] = data_encoded['Sex'].map({'male': 0, 'female': 1})
data_encoded['Embarked'] = data_encoded['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data_encoded['Survived'] = data_encoded['Survived'].astype(int)

# Example: Gender vs Fare
corr_gender_fare = data_encoded[['Sex','Fare']].corr().iloc[0,1]
sns.boxplot(x='Sex', y='Fare', data=data_encoded, palette='Set3')
plt.title(f'Fare by Gender (corr={corr_gender_fare:.2f})')
plt.xlabel('Gender (0=Male,1=Female)')
plt.show()
```

**Next Possible Statistical Tests**

* Embarked ↔ Fare → ANOVA
* Pclass ↔ Fare → Spearman
* Gender ↔ Age → T-test
* Categorical ↔ Categorical → Chi-Square + Cramér’s V

---

### Feature Selection Summary

| Category                     | Method                   | Notes                              |
| ---------------------------- | ------------------------ | ---------------------------------- |
| **Filter**                   | Correlation, ANOVA, Chi² | Identify strong predictors         |
| **Wrapper**                  | RFE                      | Iteratively drop least important   |
| **Embedded**                 | Lasso, Tree-based        | Built-in importance                |
| **Dimensionality Reduction** | PCA                      | Retain variance, reduce redundancy |
| **Interaction**              | Feature crosses          | Capture compound effects           |

**Key Takeaway:**
The **Logistic Regression model in scikit-learn** inherently uses an **Embedded feature selection technique**.
It assigns coefficients (weights) to features and applies **regularization** to control importance:

* `penalty='l1'` → performs **Lasso-style selection**, driving some weights to zero (explicit feature removal).
* `penalty='l2'` → performs **Ridge-style shrinkage**, reducing magnitude but keeping all features.

Therefore, **by default (`penalty='l2'`)**, Logistic Regression applies an **Embedded Method** for feature selection.

---

### Baseline Model — Logistic Regression (with Scaling)

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Features and Target
X = data[["Pclass", "Age", "SibSp", "Parch", "Fare"]]  # keep as DataFrame
y = data["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Optional Scaling for baseline
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# 3. Baseline Logistic Regression
log_reg = LogisticRegression(max_iter=2000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Metrics
print("Baseline Logistic Regression Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.2f}")
print(f"Precision: {precision_score(y_test, y_pred_log_reg):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_log_reg):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_log_reg):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_log_reg):.2f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))
print("Classification Report:\n", classification_report(y_test, y_pred_log_reg))
```

**Sample Output:**

```
Baseline Logistic Regression Metrics:
Accuracy: 0.62
Precision: 0.67
Recall: 0.12
F1 Score: 0.20
ROC-AUC: 0.54
Confusion Matrix:
 [[48  2]
 [30  4]]
Classification Report:
               precision    recall  f1-score   support

           0       0.62      0.96      0.75        50
           1       0.67      0.12      0.20        34

    accuracy                           0.62        84
   macro avg       0.64      0.54      0.47        84
weighted avg       0.64      0.62      0.53        84
```

---

### Sanity Check — Model Corruption & Debugging

```python
# 4. Function to break the model
def break_model(X_train_df, y_train, X_test_df, y_test, corruption_type, model_class=LogisticRegression):
    """
    Simulates different types of data corruption to test model sanity.
    corruption_type: "shuffle_labels", "drop_age", "extreme_scaling", "combined"
    """
    X_train_corrupted = X_train_df.copy()
    X_test_corrupted = X_test_df.copy()
    y_train_corrupted = y_train.copy()

    if corruption_type == "shuffle_labels":
        y_train_corrupted = np.random.permutation(y_train)
    elif corruption_type == "drop_age":
        X_train_corrupted = X_train_corrupted.drop(columns=["Age"])
        X_test_corrupted = X_test_corrupted.drop(columns=["Age"])
    elif corruption_type == "extreme_scaling":
        X_train_corrupted["Fare"] *= 1000
        X_test_corrupted["Fare"] *= 1000
    elif corruption_type == "combined":
        y_train_corrupted = np.random.permutation(y_train)
        X_train_corrupted = X_train_corrupted.drop(columns=["Age"])
        X_test_corrupted = X_test_corrupted.drop(columns=["Age"])
        X_train_corrupted["Fare"] *= 1000
        X_test_corrupted["Fare"] *= 1000
    else:
        raise ValueError("Invalid corruption type")

    model_instance = model_class(max_iter=2000, random_state=42)
    model_instance.fit(X_train_corrupted, y_train_corrupted)
    y_pred = model_instance.predict(X_test_corrupted)

    return accuracy_score(y_test, y_pred)

# Run corruption experiments
results = {
    "Baseline Accuracy": accuracy_score(y_test, y_pred_log_reg),
    "Shuffled Labels Accuracy": break_model(X_train, y_train, X_test, y_test, "shuffle_labels"),
    "Dropped Feature Accuracy": break_model(X_train, y_train, X_test, y_test, "drop_age"),
    "Extreme Scaling Accuracy": break_model(X_train, y_train, X_test, y_test, "extreme_scaling"),
    "Combined Corruption Accuracy": break_model(X_train, y_train, X_test, y_test, "combined"),
}

# Print comparison
print("\n--- Model Corruption Results ---")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.2f}")
```

**Expected Behavior:**

* **Baseline Accuracy** should remain highest (~0.6–0.7).
* **Shuffled Labels** should drop to ≈0.5 → confirms no data leakage.
* **Dropped Feature** → slight decrease (Age holds signal).
* **Extreme Scaling** → unstable convergence or reduced accuracy.
* **Combined Corruption** → worst performance → sanity check success.

---

### Final Takeaways

* Scaling stabilizes training but doesn’t fix poor feature quality.
* Sanity checks confirm whether your model learns *signal* or *noise*.
* If shuffling labels still yields high accuracy → **data leakage** exists.
* Baseline sanity ensures later complex models have meaningful validation.
* Logistic Regression = **Embedded feature selection** through regularization.

---


