# Exercise 3.2  — Classification Metrics & Feature Scaling (Logistic Regression vs SVM)

**Challenge / Prep Book README**

---

## 1) What this notebook/script does

1. Summarizes common **classification evaluation metrics** (confusion matrix, accuracy, precision, recall, F1, ROC–AUC, log loss, statistical significance via p-values/odds ratios).
2. Converts **California Housing** into a **binary classification** task (above/below median price).
3. Compares **Logistic Regression** and **SVM (RBF)** under three preprocessing regimes:

   * No scaling
   * **Min-Max normalization**
   * **Standardization (Z-score)**
4. Reports **accuracy** across settings and discusses why scaling affects SVM more than Logistic Regression.

---

## 2) Evaluation metrics (from your notes)

* **Confusion matrix**: basis for Accuracy, Precision, Recall, F1.
* **Accuracy**: share of correct predictions (can mislead on imbalanced data).
* **Precision**: among predicted positives, how many are truly positive (important when false positives are costly).
* **Recall (Sensitivity)**: among actual positives, how many are found (important when false negatives are costly).
* **F1-Score**: harmonic mean of precision and recall (useful with imbalance).
* **ROC curve & AUC**: threshold-sweep performance; AUC near 1.0 is strong, 0.5 is random.
* **Log Loss (cross-entropy)**: evaluates probabilistic predictions; lower is better.
* **Statistical significance** (for models that expose it, e.g., logistic with stats packages):

  * **p-values**: small p (<0.05) suggests predictor is statistically significant.
  * **Odds ratios**: effect size; e.g., OR=1.3 → one-unit increase raises odds by 30%.

---

## 3) Dataset & setup

* **Dataset**: `fetch_california_housing(as_frame=True)`.
* **Target**: `MedHouseVal_bin` = 1 if `MedHouseVal` > median else 0.
* **Features**: all continuous predictors except `MedHouseVal` and `MedHouseVal_bin`.
* **Split**: `train_test_split(test_size=0.2, random_state=42)`.

---

## 4) Experiments

### 4.1 Logistic Regression (max_iter=1000)

You ran LR in three conditions:

1. **No scaling**

```python
logreg_no_scaling = LogisticRegression(max_iter=1000)
logreg_no_scaling.fit(X_train, y_train)
y_pred = logreg_no_scaling.predict(X_test)
acc_no_scaling = accuracy_score(y_test, y_pred)
```

2. **Min-Max normalization**

```python
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)
logreg_norm = LogisticRegression(max_iter=1000).fit(X_train_norm, y_train)
acc_norm = accuracy_score(y_test, logreg_norm.predict(X_test_norm))
```

3. **Standardization (Z-score)**

```python
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)
logreg_std = LogisticRegression(max_iter=1000).fit(X_train_std, y_train)
acc_std = accuracy_score(y_test, logreg_std.predict(X_test_std))
```

** summary:**
Logistic Regression only needs the relative relationships between features, not absolute scales. If all features are roughly in the same range, scaling won’t drastically change the decision boundary.

---

### 4.2 SVM (RBF kernel, C=1, gamma='scale')

You repeated the same three regimes:

1. **No scaling**

```python
svm_no_scaling = SVC(kernel='rbf', C=1, gamma='scale')
svm_no_scaling.fit(X_train, y_train)
acc_no_scaling = accuracy_score(y_test, svm_no_scaling.predict(X_test))
```

2. **Min-Max normalization**

```python
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_test_norm  = scaler.transform(X_test)
svm_norm = SVC(kernel='rbf', C=1, gamma='scale').fit(X_train_norm, y_train)
acc_norm = accuracy_score(y_test, svm_norm.predict(X_test_norm))
```

3. **Standardization (Z-score)**

```python
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std  = scaler.transform(X_test)
svm_std = SVC(kernel='rbf', C=1, gamma='scale').fit(X_train_std, y_train)
acc_std = accuracy_score(y_test, svm_std.predict(X_test_std))
```

**summary:**

* No scaling (≈0.52): SVM struggles because features with larger magnitudes dominate the kernel calculation; the boundary is skewed.
* Min-Max (≈0.82): balancing feature ranges helps SVM find a better boundary.
* Standardization (≈0.86): centering and unit variance improves stability; best performance here.

---

## 5) Expected outputs / checkpoints

* Three **accuracy** values for Logistic Regression: no scaling, Min-Max, Standardization.
* Three **accuracy** values for SVM (RBF): no scaling, Min-Max, Standardization.
* Interpretation that LR is comparatively less sensitive to scaling than SVM on this task, while SVM benefits strongly from normalization/standardization.

---

## 6) Takeaways (from your comments)

* **Scaling matters** more for SVM (especially RBF kernels) than for Logistic Regression.
* **Standardization** often provides the most stable improvement for kernel methods.
* When reporting results on classification tasks, include **precision, recall, F1, ROC-AUC, and log loss** (not just accuracy), especially if class balance may vary.

---

## 7) Minimal extensions (optional, still aligned with what you built)

* Add:

  * `precision_score`, `recall_score`, `f1_score`, `roc_auc_score` (with `predict_proba`/`decision_function`).
  * `log_loss` for probabilistic evaluation.
* Plot ROC curves across the three regimes for each model.
* Compare SVM with linear kernel vs RBF under scaling.
* Add class balance check and, if imbalanced, try class weights (`class_weight='balanced'`).



## 8) One-screen summary

```
Classification Metrics & Scaling (California Housing, binary target)
• Metrics: Accuracy, Precision, Recall, F1, ROC–AUC, Log Loss, plus p-values/odds ratios where applicable
• Models: Logistic Regression vs SVM (RBF)
• Regimes: No scaling, Min-Max, Standardization
• Result: LR mildly affected by scaling; SVM strongly improved by normalization/standardization
• Next: Add full metric suite and ROC plots; explore class weights if imbalance arises
```
