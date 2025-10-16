# Ames Housing Price: Data Cleaning → Encoding → Linear Regression

**Challenge / Prep Book README**

---

## 1) What this notebook/script does

End-to-end **tabular ML pipeline** on **AmesHousing.csv**:

1. **Load & prune IDs** (`Order`, `PID`).
2. **Missing-data audit** → quantify/visualize → **targeted imputation** (domain-aware).
3. **Outlier handling** → **log1p** on skewed features + **IQR capping** on heavy-tailed areas.
4. **Correlation heatmap** → **multicollinearity reduction** (drop >0.85 pairwise corr).
5. **Encoding plan by cardinality/semantics** (one-hot, frequency, target, ordinal).
6. **ColumnTransformer + Pipeline** → **LinearRegression**.
7. **Hold-out evaluation** with **R²** and **MAE**.

---

## 2) Files & prerequisites

* **Input**: `AmesHousing.csv` (loaded with `na_values="?"`).
* **Python**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `category_encoders`.
* **Path**: update the CSV path in the loader if needed.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn category_encoders
```

---

## 3) Data preparation (what you implemented)

### 3.1 ID removal

* Dropped non-predictive identifiers: **`Order`**, **`PID`**.

### 3.2 Missingness analysis

* Computed per-column **missing counts** and **% missing**.
* Visualized via **bar plot** and simple **heatmap** of `% missing`.

### 3.3 Imputation strategy (exact columns from your code)

* **Fill “None” (structural absences)**:
  `Pool QC, Misc Feature, Alley, Fence, Mas Vnr Type, Fireplace Qu, Garage Cond, Garage Qual, Garage Finish, Garage Yr Blt, Garage Type, Bsmt Exposure, BsmtFin Type 2, Bsmt Cond, Bsmt Qual, BsmtFin Type 1`
* **Fill 0 (true numeric absence)**:
  `Mas Vnr Area, Bsmt Half Bath, Bsmt Full Bath`
* **Fill median (continuous)**:
  `BsmtFin SF 1, Garage Cars, Garage Area, Total Bsmt SF, Bsmt Unf SF, BsmtFin SF 2`
* **Fill mode (categorical)**:
  `Electrical`
* **Group-wise median by `Neighborhood`**:
  `Lot Frontage`

> After imputation, you rechecked residual missingness.

### 3.4 Define features/target early

* `X = ames_housing.drop('SalePrice')`, `y = ames_housing['SalePrice']`
  (so all later transforms apply **only** to `X` and avoid leakage into `y`).

### 3.5 Outliers & transformations

* **Log1p compression** to reduce skew:
  `Enclosed Porch, Screen Porch, Mas Vnr Area, Open Porch SF, Lot Area, Gr Liv Area, Wood Deck SF, Garage Area, 3Ssn Porch`
* **IQR capping** (clip to [Q1−1.5·IQR, Q3+1.5·IQR]):
  `BsmtFin SF 2, Lot Frontage, Total Bsmt SF, Bsmt Unf SF, 1st Flr SF, BsmtFin SF 1, 2nd Flr SF`
* Printed **before/after `.describe()`** for all transformed features.

**Outcome:** Skewness reduced, heavy tails controlled, central tendency preserved.

### 3.6 Correlation & multicollinearity

* Heatmap on numeric features **after outlier treatment**.
* Built upper-triangle matrix and **dropped columns with any corr > 0.85** → `X_reduced`.

### 3.7 Safety NaN audit (post-reduction)

* **Numeric**: fill remaining NaNs with **median**.
* **Non-numeric**: fill with **"Unknown"**.
* Verified **residual total NaNs = 0**.

---

## 4) Encoding & modeling (what you implemented)

### 4.1 Feature groups (by your plan)

* **Low-cardinality (one-hot)**:
  `Street, Alley, Utilities, Central Air, Paved Drive, Mo Sold, Yr Sold, Sale Condition`
* **Medium-cardinality (frequency/count encode)**:
  `MS Zoning, Condition 1, Condition 2, Bldg Type, House Style, Roof Style, Exterior 1st, Exterior 2nd, Foundation, Heating, Electrical, Functional, Garage Type, Sale Type, Land Contour, Lot Config, Roof Matl, Mas Vnr Type, Misc Feature`
* **High-cardinality (target/mean encode)**:
  `Neighborhood, MS SubClass`
* **Ordinal (ranked)** (ordinal encoder):
  `Lot Shape, Land Slope, Exter Qual, Exter Cond, Bsmt Qual, Bsmt Cond, Bsmt Exposure, BsmtFin Type 1, BsmtFin Type 2, Heating QC, Kitchen Qual, Fireplace Qu, Garage Qual, Garage Cond, Garage Finish, Fence, Pool QC`
* **Numeric**: `StandardScaler`

### 4.2 ColumnTransformer + Pipeline

* `OneHotEncoder(handle_unknown='ignore')`
* `ce.CountEncoder(normalize=True)`
* `ce.TargetEncoder(smoothing=2)`
* `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`
* `StandardScaler()`
* **Regressor**: `LinearRegression()`

### 4.3 Train/test split & fit

* `train_test_split(test_size=0.2, random_state=42)`
* `model.fit(X_train, y_train) → predict → R² & MAE` printed.

---

## 5) How to run (step-by-step)

1. Place `AmesHousing.csv` at the path used in your script (or update the path).
2. Install deps (see **Prereqs**).
3. Run cells top-to-bottom (notebook) or execute the script.
4. Watch for plots (missingness bar/heatmap, correlation heatmap) and metrics at the end:

   * **R²**: overall variance explained by linear model.
   * **MAE**: average absolute prediction error (in SalePrice units).

---

## 6) Expected outputs / checkpoints

* Missingness table: sorted by `% missing`.
* Two visuals: missingness bar; correlation heatmap (post-treatment).
* Outlier report: IQR outlier counts per numeric feature; outliers in `y`.
* Before/After `.describe()` for transformed features (sanity check).
* Dropped col list from >0.85 correlation threshold.
* Final metrics: `R²`, `MAE` on the test set.

(Exact values depend on the CSV and random split; you print them at runtime.)

---

## 7) Sanity & leakage guards (already in your code)

* Define `X`/`y` before modeling; all transforms go through `ColumnTransformer` inside the training Pipeline → prevents leakage.
* `handle_unknown='ignore'` for one-hot; `unknown_value=-1` for ordinals → robust to rare/unseen categories.
* Safety NaN patch just before split ensures the pipeline won’t crash on leftover holes.

---

## 8) Notes & reflections (from your comments)

* Log + IQR reduced skew and stabilized scales; area/size features benefited most.
* Multicollinearity is strong among footprint/area features (e.g., `1st Flr SF`, `Total Bsmt SF`, `Gr Liv Area`), and Garage Cars ↔ Garage Area.
* Overall Qual correlates with many drivers of `SalePrice`.
* Encoding strategy avoids one-hot explosion (freq/target encoding for bigger cardinals) while preserving ordinal meaning where applicable.

---

## 9) Known limitations (as of this challenge stage)

* LinearRegression only (no regularization); multicollinearity can still affect stability even after drops.
* TargetEncoder applied in the pipeline—good—but no explicit CV inside encoding beyond smoothing (fine for this stage).
* No feature importance/coeff interpretation yet (optional next step).
* No residual diagnostics or error analysis by segment (e.g., neighborhood, house age).

---

## 10) Minimal extensions (optional, still aligned with what you built)

* Swap `LinearRegression` → Ridge/Lasso; compare R² / MAE.
* Permutation importance on the preprocessed features.
* Residual plots vs. fitted values and key predictors.
* Cross-validation (`cross_val_score`) to stabilize estimates.

---

## 11) Quick rubric (pass/fail for this prep)

* [ ] IDs dropped (`Order`, `PID`)
* [ ] Missingness quantified & visualized
* [ ] Imputation (None/0/median/mode + group-median for `Lot Frontage`)
* [ ] Skewed features log-transformed; heavy tails capped
* [ ] Correlation heatmap; features dropped at >0.85
* [ ] Safe NaN patch (numeric medians / “Unknown” for cats)
* [ ] ColumnTransformer encodings as specified
* [ ] LinearRegression pipeline trained & evaluated (R², MAE printed)

---

## 12) One-screen summary

```
Ames Housing — Linear Regression Pipeline
• Cleaning: domain-aware imputations + group median for Lot Frontage
• Outliers: log1p (porches/areas), IQR capping (basement/floor areas)
• Multicollinearity: drop cols with corr > 0.85
• Encoding: OneHot (low), Count (med), Target (high), Ordinal (ranked)
• Scaling: StandardScaler for numeric
• Model: LinearRegression (train/test split)
• Metrics: R², MAE (hold-out)
```

---
