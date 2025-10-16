import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import category_encoders as ce
import numpy as np

# Load Data
ames_housing = pd.read_csv(
    "/Users/shivasriram/Desktop/Daily ML/Exercise2/AmesHousing.csv",
    na_values="?"
)
print(ames_housing.head())
print(ames_housing.columns.tolist())

# Remove identifier columns
ames_housing.drop(columns=['Order', 'PID'], inplace=True)
print(ames_housing.columns.tolist())

# A) Data Cleaning

# Check missing values
missing_counts = ames_housing.isnull().sum()
missing_columns = missing_counts[missing_counts > 0]
print(missing_columns)

# Quantify missingness
total_rows = len(ames_housing)
missing_info = missing_columns.to_frame(name='Missing_counts')
missing_info['Percentage Missings'] = (missing_info['Missing_counts'] / total_rows * 100)
missing_info_sorted = missing_info.sort_values(by='Percentage Missings', ascending=False)
print(missing_info_sorted)

# Bar Plot of missing percentages
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_info_sorted.index, y=missing_info_sorted['Percentage Missings'])
plt.title('Percentage Missings')
plt.xlabel('Features')
plt.ylabel('Percentage')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Heatmap of missing percentages
plt.figure(figsize=(10, 6))
sns.heatmap(data=missing_info_sorted[['Percentage Missings']])
plt.tight_layout()
plt.show()

# Missing Value Imputation Checklist
# ### Quantify & Visualize
# - Use `.isnull().sum()` and `% missing`.
# - Visualize patterns → `missingno.matrix()`, heatmap.
# - Focus on columns with >10% missingness.

# ### Identify Missingness Type
# | Type | Meaning | Example | Imputation |
# |------|----------|----------|-------------|
# | **MCAR** | Random, unrelated to any feature | Randomly lost entry | Mean/Median/Mode |
# | **MAR** | Depends on *observed* feature | `GarageType` ↔ `GarageArea` | Group-based / KNN |
# | **MNAR** | Depends on *unobserved/self* value | High income not reported | Model-based / flag as info |

# ### Structural vs. True Missing
# | Case | Example | Action |
# |------|----------|---------|
# | Structural | No garage → `GarageType=NaN` | Fill `"None"` / `0` |
# | True missing | `LotFrontage` not recorded | Group median / KNN |
# | Outlier removal | Extreme values dropped | Flag / Model-based |

# ### Imputation Strategy by Data Type
# | Type | <10% | 10–40% | >40% |
# |------|------|--------|------|
# | **Numeric** | Mean/Median | KNN / Group Median | Drop / Flag |
# | **Categorical** | Mode | Conditional / "Unknown" | Drop / Flag |
# | **Structural** | "None" / 0 | Same | Same |

# ### Validate
# - Compare distributions (before vs. after imputation).  
# - Check correlation shifts (`.corr()` before/after).  
# - Add missing flags for MAR/MNAR

# Impute missing values
fill_none_features = [
    "Pool QC", "Misc Feature", "Alley", "Fence",
    "Mas Vnr Type", "Fireplace Qu",
    "Garage Cond", "Garage Qual", "Garage Finish",
    "Garage Yr Blt", "Garage Type",
    "Bsmt Exposure", "BsmtFin Type 2", "Bsmt Cond",
    "Bsmt Qual", "BsmtFin Type 1"
]
ames_housing[fill_none_features] = ames_housing[fill_none_features].fillna("None")

fill_zero_features = ["Mas Vnr Area", "Bsmt Half Bath", "Bsmt Full Bath"]
ames_housing[fill_zero_features] = ames_housing[fill_zero_features].fillna(0)

fill_median_features = [
    "BsmtFin SF 1", "Garage Cars", "Garage Area",
    "Total Bsmt SF", "Bsmt Unf SF", "BsmtFin SF 2"
]
ames_housing[fill_median_features] = ames_housing[fill_median_features].fillna(
    ames_housing[fill_median_features].median()
)

ames_housing["Electrical"] = ames_housing["Electrical"].fillna(ames_housing["Electrical"].mode()[0])

ames_housing["Lot Frontage"] = ames_housing.groupby("Neighborhood")["Lot Frontage"].transform(
    lambda x: x.fillna(x.median())
)

missing_columns_after = ames_housing.isnull().sum()
print(missing_columns_after[missing_columns_after > 0])

# ======= PHASE 2: DEFINE X/y EARLY (so transforms below apply to X) =======
X = ames_housing.drop('SalePrice', axis=1) # X -> dataframe with handled null values.
y = ames_housing['SalePrice']

# EDA on Numeric Features
# Uncomment below to visualize distributions and boxplots

# # Histograms
# for feature in numeric_features:
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data=X, x=feature, bins=10)
#     plt.title(f'Histogram of {feature}')
#     plt.show()

# # Boxplots
# for feature in numeric_features:
#     plt.figure(figsize=(10,6))
#     sns.boxplot(data=X, y=feature)
#     plt.title(f"Boxplot of {feature}")
#     plt.show()

# Outlier Detection (IQR)
numeric_features = X.select_dtypes(include=['number']).columns.tolist()
outlier_counts_iqr = {}
for feature in numeric_features:
    Q1 = X[feature].quantile(0.25)
    Q3 = X[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    num_outliers = ((X[feature] < lower_bound) | (X[feature] > upper_bound)).sum()
    outlier_counts_iqr[feature] = num_outliers

outlier_series_iqr = pd.Series(outlier_counts_iqr).sort_values(ascending=False)
print("Outlier counts per feature (IQR Method):")
print(outlier_series_iqr)


Q1 = y.quantile(0.25)
Q3 = y.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_y = y[(y < lower_bound) | (y > upper_bound)]
print(f"Number of outliers in y: {len(outliers_y)}")


# Log transform
log_features = ['Enclosed Porch', 'Screen Porch', 'Mas Vnr Area', 'Open Porch SF',
                'Lot Area', 'Gr Liv Area', 'Wood Deck SF', 'Garage Area', '3Ssn Porch']
for feature in log_features:
    X[feature] = np.log1p(X[feature])


# Cap using IQR
cap_features = ['BsmtFin SF 2', 'Lot Frontage', 'Total Bsmt SF', 
                'Bsmt Unf SF', '1st Flr SF', 'BsmtFin SF 1', '2nd Flr SF']
for feature in cap_features:
    Q1 = X[feature].quantile(0.25)
    Q3 = X[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    X[feature] = X[feature].clip(lower, upper)

    
# Compare statistical summaries
features_to_check = log_features + cap_features

for feature in features_to_check:
    print(f"\n--- {feature} ---")
    print("Before:")
    print(ames_housing[feature].describe(percentiles=[0.25, 0.5, 0.75]))
    print("After:")
    print(X[feature].describe(percentiles=[0.25, 0.5, 0.75]))

# Visualize effect of transformations
# for feature in features_to_check:
#     plt.figure(figsize=(10, 5))

#     plt.subplot(1, 2, 1)
#     sns.boxplot(y=ames_housing[feature])
#     plt.title(f"Before Transformation - {feature}")

#     plt.subplot(1, 2, 2)
#     sns.boxplot(y=X[feature])
#     plt.title(f"After Transformation - {feature}")

#     plt.tight_layout()
#     plt.show()

# Outlier transformations (log and IQR capping) effectively reduced skewness and stabilized feature distributions.  
# Extreme values were compressed while preserving the central tendencies of each variable, leading to tighter spreads and more consistent scales.  
# Features such as `Lot Area`, `Gr Liv Area`, and `Mas Vnr Area` benefited most from log compression, while structural metrics like `Total Bsmt SF` and `1st Flr SF` showed improved uniformity after IQR capping.  
# Overall, the preprocessing achieved robust control of outliers without distorting genuine data patterns — ensuring cleaner, model-ready inputs for training.

# Heat Map 
numeric_df = X.select_dtypes(include=['int64', 'float64'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap (After Outlier Treatment)", fontsize=14)
plt.show()

# Strong positive correlations (deep red diagonals beyond the main one) suggest multicollinearity among structural area features:

# 1st Flr SF, Total Bsmt SF, and Gr Liv Area are highly correlated — all represent size of the house.

# Garage Cars ↔ Garage Area — very strong correlation.

# Overall Qual shows moderate correlation with several size-related and finish-related features — important predictor of SalePrice.

# Weak or near-zero correlations (grey zones) indicate independent predictors, which add diversity to the model.

upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]

X_reduced = X.drop(columns=to_drop) # X_reduced -> handled null values, no outliers. but not yet encoded.
print(X_reduced)
X_reduced.describe()

# PHASE 3: ENCODING + MODEL

# Encoding Categorical Variables
categorical_cols = ames_housing.select_dtypes(include="category").columns
print(categorical_cols)

# Cardinality
cardinality_cat_cols = ames_housing[categorical_cols].nunique()
for column, cardinality in cardinality_cat_cols.items():
    print(f"{column}: {cardinality}")

# Recommended encoding strategies
# | Feature Type                             | Example Features | Cardinality | Recommended Encoding | Reasoning |
# | ---------------------------------------- | ---------------- | ----------- | ------------------ | --------- |
# | **Low-cardinality nominal (≤10)**        | `Street`, `Alley`, `Central Air` | ≤10 | One-Hot | Simple categorical features |
# | **Medium-cardinality nominal (10–20)**   | `MS Zoning`, `Condition 1`, `House Style` | 10–20 | Frequency / Count Encoding | Avoids one-hot explosion |
# | **High-cardinality nominal (>20)**       | `Neighborhood` | >20 | Target / Mean Encoding | Keeps cardinality manageable |
# | **Ordinal (clear rank order)**           | `Exter Qual`, `Bsmt Qual`, `Garage Qual` | Varies | Ordinal Mapping | Preserve qualitative ranking |
# | **Numeric Continuous**                   | `Lot Area`, `Gr Liv Area` | Continuous | StandardScaler | Scale-sensitive models |
# | **Numeric Discrete (count-based)**       | `Overall Qual`, `Fireplaces` | Low-Mid | None | Already numeric |
# | **Temporal**                             | `Year Built`, `Yr Sold` | 5–12 | Time-based features | e.g., Age, RemodelTime |
# | **Binary categorical**                   | `Central Air`, `Street` | 2–3 | Label Encoding | Simple 0/1 encoding |

# Feature Splitting
# NEW: rebind X so lists & numeric_features align with reduced columns
X = X_reduced.copy()

# --- SAFETY: quick NaN audit & minimal patch (temporary for debugging) ---
nan_counts = X.isna().sum().sort_values(ascending=False)
print("Top columns with NaN (before patch):")
print(nan_counts[nan_counts > 0].head(25))

# minimal, schema-preserving filler: numeric -> column medians; non-numeric -> "Unknown"
num_cols = X.select_dtypes(include=[np.number]).columns
cat_cols = X.columns.difference(num_cols)

X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("Unknown")

print("Residual total NaNs after patch:", int(X.isna().sum().sum()))
# --- end safety ---


low_cardinality_features = ["Street", "Alley", "Utilities", "Central Air",
                            "Paved Drive", "Mo Sold", "Yr Sold", "Sale Condition"]

medium_cardinality_features = [
    "MS Zoning", "Condition 1", "Condition 2", "Bldg Type", "House Style",
    "Roof Style", "Exterior 1st", "Exterior 2nd", "Foundation", "Heating",
    "Electrical", "Functional", "Garage Type", "Sale Type",
    "Land Contour", "Lot Config", "Roof Matl", "Mas Vnr Type", "Misc Feature"
]

high_cardinality_features = ["Neighborhood", "MS SubClass"]

ordinal_features = [
    'Lot Shape', 'Land Slope', 'Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond',
    'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2', 'Heating QC', 'Kitchen Qual',
    'Fireplace Qu', 'Garage Qual', 'Garage Cond', 'Garage Finish', 'Fence', 'Pool QC'
]

numeric_features = X.select_dtypes(include=['number']).columns.tolist()

# Preprocessing Pipelines
one_hot_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
ordinal_pipeline = Pipeline([
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
frequency_pipeline = Pipeline([('freq', ce.CountEncoder(normalize=True))])
target_encode_pipeline = Pipeline([('target', ce.TargetEncoder(smoothing=2))])
numeric_pipeline = Pipeline([('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', one_hot_pipeline, low_cardinality_features),
        ('freq', frequency_pipeline, medium_cardinality_features),
        ('target', target_encode_pipeline, high_cardinality_features),
        ('ordinal', ordinal_pipeline, ordinal_features),
        ('numeric', numeric_pipeline, numeric_features)
    ],
    remainder='drop'
)

# Transformation doen at last due to data leakage problem.

# Train/Test Split 

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
