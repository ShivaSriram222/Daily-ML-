# Feature Preprocessing — Manual Implementation vs Scikit-Learn Transformers


# Manual implementation
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/shivasriram/Desktop/Daily ML/Exercise4/AmesHousing.csv")
print(df.head())
print(df.columns)

# Columns used for this challenge
subset_cols = [
    'Lot Area',  
    'Gr Liv Area',           
    'SalePrice',      
    'Overall Qual',      
    'MS Zoning',      
    'Neighborhood',   
    'Bldg Type',       
    'House Style',    
    'Exter Qual',     
]

df = df[subset_cols]
print(df.head())
print(df.shape)
print(df.columns)






# Reference 

# | Case                           | Encoding Type                        | When to Use                                  | Example Columns                                              | Notes / Tradeoffs                                                                                                                                            |
# | ------------------------------ | ------------------------------------ | -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
# | Nominal (no order)             | One-Hot Encoding                     | Categories are few and unordered             | MS Zoning, Neighborhood, BldgType                            | Creates one binary column per class. Works well if unique values ≤ ~20. Not ideal for high-cardinality features (many unique categories).                    |
# | Ordinal (ordered)              | Ordinal Encoding                     | Categories have natural ranking              | ExterQual (Po < Fa < TA < Gd < Ex), KitchenQual, OverallCond | Encodes categories as integers (1,2,3,...). Retains rank information. Do not use One-Hot here, since it loses order semantics.                               |
# | High Cardinality (many unique) | Target Encoding / Frequency Encoding | More than ~20–30 unique values or text-based | Neighborhood (if many unique), StreetName, ProductID         | Replace category with mean of target (supervised) or frequency count (unsupervised). Reduces dimensionality. Must avoid leakage — fit only on training data. |
# | Binary                         | Label Encoding                       | Only 2 unique values                         | CentralAir (Y/N), PavedDrive (Y/N)                           | Converts to 0/1. Both One-Hot and LabelEncoding yield same effect here.                                                                                      |


# | Scaling Method                           | When to Use                                                                  | Formula                            | Effect on Data                                   | Typical Models / Use Cases                                                |
# | ---------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------- |
# | Z-Score Standardization (StandardScaler) | When data is approximately normal (bell-shaped) and contains outliers        | x' = (x - μ) / σ                   | Centers data around 0 with unit variance         | Linear Regression, Logistic Regression, SVM, PCA, KNN, Neural Networks    |
# | Min-Max Scaling (Normalization)          | When you want all values in [0,1] or when feature bounds are known and fixed | x' = (x - x_min) / (x_max - x_min) | Compresses all values between 0 and 1            | Neural Networks, Gradient Descent-based models, Image data (pixel values) |
# | Robust Scaling                           | When data has strong outliers                                                | x' = (x - median) / IQR            | Uses median and IQR → less sensitive to outliers | Datasets with heavy tails or skewed distributions                         |
# | Log / Power Transformation               | When distribution is skewed (non-Gaussian)                                   | x' = log(x + 1)                    | Normalizes right-skewed data                     | Price, Income, or Count features before applying StandardScaler           |


from sklearn.model_selection import train_test_split


y = df['SalePrice']
X = df.drop(columns =['SalePrice'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Identify numeric & categorical columns
num_cols = X_train.select_dtypes(include=['int', 'float']).columns
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

# Feature Preprocessing — Manual Implementation
# Min-Max Scaling:
# Fit scaler only on training data
X_train_min = X_train[num_cols].min()
X_train_max = X_train[num_cols].max()

# Avoid division by zero
denom = X_train_max - X_train_min
denom[denom == 0] = 1

X_train_scaled = (X_train[num_cols] - X_train_min) / denom
X_test_scaled = (X_test[num_cols] - X_train_min) / denom    

y_train = y_train.astype(float)
y_test = y_test.astype(float)

y_min = y_train.min()
y_max = y_train.max()
y_train_scaled = (y_train - y_min) / (y_max - y_min)
y_test_scaled = (y_test - y_min) / (y_max - y_min)


# One hot encoding 
train_onehot = pd.DataFrame()
test_onehot = pd.DataFrame()

for col in cat_cols:
    unique_cats = X_train[col].dropna().unique()  # only train categories

for cat in unique_cats:
    new_col = f"{col}_{cat}"
    train_onehot[new_col] = (X_train[col] == cat).astype(int)
    test_onehot[new_col] = (X_test[col] == cat).astype(int) # For test: unseen categories will just be 0
    

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score
 

model = LinearRegression()
model.fit(X_train_scaled, y_train_scaled)

y_pred_scaled = model.predict(X_test_scaled)
y_pred = y_min + y_pred_scaled * (y_max - y_min)  # inverse transform

print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:", mean_absolute_error(y_test, y_pred))


def adjusted_r2(y_test, y_pred, n_features):
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

print("rsquared:", adjusted_r2(y_test, y_pred, X.shape[1]))   



#sklearn implementation 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

y = df['SalePrice']
X = df.drop(columns =['SalePrice'])

X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(X,y, test_size=0.2, random_state=0 )


numeric_transformer = Pipeline(
steps = [(("scaler", StandardScaler()))]
)

categorical_transformer = Pipeline(
steps = [("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

clf = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]
)


# Training and Evaluation 
clf.fit(X_train_sk, y_train_sk)
rsquared_sklearn = clf.score(X_train, y_train) #r_sqaured metric 
print("rsqaured_sklearn", rsquared_sklearn)








# # ===========================================
# # Feature Preprocessing: Manual vs Sklearn
# # ===========================================

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# # -----------------------------
# # Load Data
# # -----------------------------
# df = pd.read_csv("/Users/shivasriram/Desktop/Daily ML/Exercise4/AmesHousing.csv")

# subset_cols = [
#     'Lot Area',
#     'Gr Liv Area',
#     'SalePrice',
#     'Overall Qual',
#     'MS Zoning',
#     'Neighborhood',
#     'Bldg Type',
#     'House Style',
#     'Exter Qual',
# ]
# df = df[subset_cols]

# # -----------------------------
# # Train-Test Split
# # -----------------------------
# y = df['SalePrice']
# X = df.drop(columns=['SalePrice'])
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=0
# )

# # Identify numeric & categorical columns
# num_cols = X_train.select_dtypes(include=['int', 'float']).columns
# cat_cols = X_train.select_dtypes(include=['object', 'category']).columns

# # ===========================================
# # Manual Preprocessing
# # ===========================================

# # 1️⃣ Numeric Scaling (Min-Max)
# X_train_min = X_train[num_cols].min()
# X_train_max = X_train[num_cols].max()
# denom = X_train_max - X_train_min
# denom[denom == 0] = 1  # avoid division by zero

# X_train_num_scaled = (X_train[num_cols] - X_train_min) / denom
# X_test_num_scaled = (X_test[num_cols] - X_train_min) / denom

# # 2️⃣ One-Hot Encoding (categorical)
# train_onehot = pd.get_dummies(X_train[cat_cols], drop_first=False)
# test_onehot = pd.get_dummies(X_test[cat_cols], drop_first=False)

# # Align test set columns with train set
# test_onehot = test_onehot.reindex(columns=train_onehot.columns, fill_value=0)

# # 3️⃣ Combine numeric + categorical
# X_train_manual = pd.concat([X_train_num_scaled, train_onehot], axis=1)
# X_test_manual = pd.concat([X_test_num_scaled, test_onehot], axis=1)

# # 4️⃣ Train Linear Regression (manual)
# model_manual = LinearRegression()
# model_manual.fit(X_train_manual, y_train)

# y_pred_manual = model_manual.predict(X_test_manual)

# # Metrics
# rmse_manual = np.sqrt(mean_squared_error(y_test, y_pred_manual))
# mae_manual = mean_absolute_error(y_test, y_pred_manual)
# r2_manual = r2_score(y_test, y_pred_manual)

# print("----- Manual Preprocessing -----")
# print("RMSE:", rmse_manual)
# print("MAE:", mae_manual)
# print("R²:", r2_manual)

# # ===========================================
# # Scikit-Learn Preprocessing
# # ===========================================

# # ColumnTransformer: numeric -> StandardScaler, categorical -> OneHotEncoder
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), num_cols),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
#     ]
# )

# # Full pipeline with Linear Regression
# clf = Pipeline(
#     steps=[
#         ("preprocessor", preprocessor),
#         ("regressor", LinearRegression())
#     ]
# )

# # Train
# clf.fit(X_train, y_train)

# # Predict
# y_pred_sklearn = clf.predict(X_test)

# # Metrics
# rmse_sklearn = np.sqrt(mean_squared_error(y_test, y_pred_sklearn))
# mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)
# r2_sklearn = r2_score(y_test, y_pred_sklearn)

# print("\n----- Sklearn Preprocessing -----")
# print("RMSE:", rmse_sklearn)
# print("MAE:", mae_sklearn)
# print("R²:", r2_sklearn)

# # ===========================================
# # Comparison Notes:
# # - Both now use numeric + categorical features
# # - Manual uses Min-Max, sklearn uses StandardScaler (slight difference)
# # - Metrics should be very close
# # ===========================================
