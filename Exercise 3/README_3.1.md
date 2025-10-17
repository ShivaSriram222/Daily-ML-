# Exercise 3.1 — Data Leakage & Target Leakage (Prep README)

**Challenge / Prep Book**

---

## 1) What this notebook/script does

1. Defines **target leakage** and related leakage types.
2. Provides a **prevention checklist** and safe/unsafe operation tables.
3. **Demonstrates target leakage** on California Housing by adding a target-proxy feature and comparing inflated vs. clean metrics.
4. Lists additional leakage scenarios to extend later.

---

## 2) Core concepts

### Target Leakage

A feature directly/indirectly reveals the target or uses information only available after the prediction event.

Examples:

* Pneumonia prediction with `took_antibiotic_medicine` (post-diagnosis).
* Churn prediction with `service_calls_after_cancellation` (post-churn).

### Common leakage types

* Train–Test Contamination (fit transforms on full data before split)
* Temporal Leakage (use future info to predict past/present)
* Target Leakage (feature proxies the target)
* Post-Event Feature Leakage (feature available only after outcome)
* Aggregation Leakage (global stats computed with test/future rows)

**Why it matters:** Inflated validation (e.g., R² ≈ 0.99 vs ≈ 0.8), bad model selection, poor deployment performance.

**Prevention checklist:**
Split first; time-aware validation; exclude post-event/target-derived features; compute encodings/stats on **training only**; validate with realistic data availability.

---

## 3) Safe vs. unsafe operations

**Safe before split (target independent):** drop IDs, remove duplicates, basic cleaning, FE from predictors only, simple label maps, EDA.

**Must be after split (target dependent):** scaling/normalization, imputation, target/aggregation encodings, target-based feature selection, PCA/SVD, target-based outlier handling.
Always **fit on X_train** then transform train/test.

---

## 4) Reproducible demo: Target correlation leak (implemented)

**Dataset:** California Housing (`fetch_california_housing`)

**Steps:**

1. Create `LeakageFeature = MedHouseVal + small_noise`.
2. Train/test split; `StandardScaler` → `LinearRegression`.
3. Evaluate with leak; remove `LeakageFeature` and re-evaluate.

**Expected metrics from your run:**

```
With Leakage
R² ≈ 0.9925, MAE ≈ 0.0789

Without Leakage
R² ≈ 0.5758, MAE ≈ 0.5332
```

**Interpretation:** The leakage feature creates unrealistic, inflated performance.

---


## 5) Additional scenarios to extend (placeholders in your notes)

* Aggregated future info leak (compute aggregates on training only, map to test).
* Domain feature leak (post-event attributes).
* Post-processing label leak (transforms applied pre-split).
* Outlier label influence leak (rules derived using target).

---

## 6) One-screen summary

```
Exercise 3.1 — Data Leakage (Target Leakage Focus)
• Demo: California Housing with synthetic target-proxy feature
• Result: R² ~0.99 with leak vs ~0.58 without; MAE far lower with leak
• Prevention: Split first; fit stats/encodings on training only; exclude post-event features
• Next: Add aggregation/domain/pipeline leakage demos with same pattern
```
