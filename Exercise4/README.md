# Feature Preprocessing: Manual Scaling/Encoding vs scikit-learn Transformers

**Challenge / Prep Book README**
---

## 1) What this notebook/script does

1. Loads a subset of Ames Housing columns and defines predictors/target.
2. Implements **manual preprocessing**:

   * Min–Max scaling for numeric features and target (with train-fitted mins/maxes).
   * One-hot encoding for categorical features using **training-only categories** (unseen test categories default to 0).
3. Trains **LinearRegression** on manually scaled numeric features and evaluates:

   * RMSE, MAE, and adjusted R² (with an explicit function).
4. Re-implements preprocessing via **scikit-learn**:

   * `StandardScaler` for numeric, `OneHotEncoder` for categorical inside a `ColumnTransformer` + `Pipeline`.
   * Evaluates with scikit-learn’s `Pipeline.score` (R²).

---

## 2) Data and columns used

* File: `AmesHousing.csv`
* Subset columns you selected:

  * Numeric: `Lot Area`, `Gr Liv Area`, `SalePrice`, `Overall Qual`
  * Categorical: `MS Zoning`, `Neighborhood`, `Bldg Type`, `House Style`, `Exter Qual`

Split: `train_test_split(test_size=0.2, random_state=0)`.

---

## 3) Manual preprocessing (exactly as implemented)

### 3.1 Numeric scaling (Min–Max)

* Fit min and max on **training** numeric columns only:

  ```
  X_train_min = X_train[num_cols].min()
  X_train_max = X_train[num_cols].max()
  denom = X_train_max - X_train_min; denom[denom==0] = 1
  X_train_scaled = (X_train[num_cols] - X_train_min) / denom
  X_test_scaled  = (X_test[num_cols]  - X_train_min) / denom
  ```
* Target scaling:

  ```
  y_min, y_max = y_train.min(), y_train.max()
  y_train_scaled = (y_train - y_min) / (y_max - y_min)
  y_test_scaled  = (y_test  - y_min) / (y_max - y_min)
  ```
* Prediction inverse-transform:

  ```
  y_pred = y_min + y_pred_scaled * (y_max - y_min)
  ```

### 3.2 Categorical encoding (One-hot)

* Build training-driven one-hot columns per categorical feature:

  * Enumerate **unique categories from training**.
  * Create `col_category` indicator columns in train and test.
  * Test rows with unseen categories get zeros in that feature’s one-hot set.

*(Note: your loop structure intends per-column one-hot using training categories.)*

---

## 4) Manual model and evaluation

* Model: `LinearRegression()` trained on **manually scaled numeric** features.
* Metrics printed:

  * RMSE: `sqrt(mean_squared_error(y_test, y_pred))`
  * MAE: `mean_absolute_error(y_test, y_pred)`
  * Adjusted R² (custom function):

    ```
    1 - (1 - R²) * (n - 1) / (n - p - 1)
    ```

    where `p` = number of features you pass into the model.

---

## 5) scikit-learn transformers and pipeline

### 5.1 Preprocessors

* Numeric transformer: `StandardScaler()`
* Categorical transformer: `OneHotEncoder(handle_unknown="ignore")`

### 5.2 ColumnTransformer and Pipeline

```
preprocessor = ColumnTransformer(
  transformers=[
    ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
  ]
)

clf = Pipeline(steps=[
  ("preprocessor", preprocessor),
  ("regressor", LinearRegression())
])
```

Train/evaluate:

```
clf.fit(X_train_sk, y_train_sk)
rsquared_sklearn = clf.score(X_train, y_train)  # R²
```

---



##  One-screen summary

```
Feature Preprocessing — Manual vs scikit-learn
• Manual: Min–Max scaling for numeric; train-only one-hot for categorical; LinearRegression; RMSE/MAE/adj-R²
• sklearn: StandardScaler + OneHotEncoder in ColumnTransformer; LinearRegression; R² via Pipeline.score
• Guardrails: fit scalers/encoders on training only; inverse-transform target for error metrics
• Next: join manual one-hot with scaled numeric, add CV, residual diagnostics, and regularization
```
