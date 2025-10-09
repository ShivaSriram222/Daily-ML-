# A) Data Sanity & Exploration

import pandas as pd

# Load the dataset
data = pd.read_csv('/Users/shivasriram/Desktop/Daily ML/Exercise1/Titanic.csv')
data.head()

#columns in the dataset
print("Columns in the dataset:", data.columns)

# Basic statistics of the dataset
print(data.describe(include='all'))

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Plotting Missing Values
import matplotlib.pyplot as plt
import seaborn as sns 

plt.figure(figsize=(10,6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.xlabel('Features')
plt.show()

# Handling Missing Values
# Fill missing 'Age' with median age
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing 'Cabin' with 'Unknown'
data['Cabin'].fillna('Unknown', inplace=True)


data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)


data['Fare'].fillna(data['Fare'].median(), inplace=True)

# Verify no missing values remain
print("Missing Values after handling:\n", data.isnull().sum())


# Check for duplicates, invalid entries, and unrealistic values.


# Data Distribution
numerical_columns = ['Age', 'SibSp', 'Parch', 'Fare']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
axes = axes.flatten()
for i, col in enumerate(numerical_columns):
    ax = axes[i]
    
    # Plot the histogram on the current subplot
    ax.hist(data[col], bins=15, edgecolor='black', color="#27982B")
    
    # Set the title and labels for the subplot
    ax.set_title(f'Distribution of {col}', fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

categorical_columns = ['Sex', 'Pclass', 'Embarked', 'Survived']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
axes = axes.flatten()
for i, col in enumerate(categorical_columns):
    ax = axes[i]
    
    # Plot the countplot on the current subplot
    sns.countplot(data=data, x=col, ax=ax, palette='Set2')
    
    # Set the title and labels for the subplot
    ax.set_title(f'Count of {col}', fontsize=12)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.show()  

# Reveals skewness and outliers
# Informs feature scaling
# Guides encoding strategy
# Supports model selection

# Transform features: Apply transformations like logarithm or square root to fix skewed distributions in numerical features.
# Handle outliers: Decide whether to remove, cap, or transform features with significant outliers.
# Scale features: Apply the appropriate scaler (e.g., StandardScaler, MinMaxScaler) based on the distribution.
# Refine encoding: Use the right encoding strategy for your categorical data to avoid creating too many features.
# Re-evaluate correlations: After transforming your data, your correlation plots might change. A previously low correlation might become stronger after a non-linear transformation.






# Correlation Analysis

# Correlation Between Numerical Features
corr_matrix = data[numerical_columns].corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Correlation Between Numerical Features and Categorical Features
# Encoding categorical features for correlation analysis
data_encoded = data.copy()
data_encoded['Sex'] = data_encoded['Sex'].map({'male': 0, 'female': 1})
data_encoded['Embarked'] = data_encoded['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
data_encoded['Pclass'] = data_encoded['Pclass'].map({1: 1, 2: 2, 3: 3})
data_encoded['Survived'] = data_encoded['Survived'].map({0: 0, 1: 1})

# Gender and Fare: Was there a difference in the amount of money paid for tickets between men and women?
 # Used perason correlation coefficient, generally used for continuous variables, but can be used here as well since gender is binary and pearson becomes equivalent to point-biserial correlation.
corr_gender_fare = data_encoded[['Sex', 'Fare']].corr().iloc[0,1]
plt.figure(figsize=(10,6))
sns.boxplot(x='Sex', y='Fare', data=data_encoded, palette='Set3')
plt.title(f'Boxplot of  Fare by Gender (Correlation: {corr_gender_fare:.2f})')
plt.xlabel('Gender (0: Male, 1: Female)')
plt.ylabel('Fare')
plt.show()

# Embarked and Fare: Did passengers embarking from different ports have different economic profiles, reflected in their fare?
# perform ANOVA, because pearson correlation is not suitable for categorical variables with more than two categories 

# Pclass and Fare: Was there a clear relationship between the passenger's class and the fare they paid? 
# perform Spearman correlation, because Pclass is ordinal categorical variable

# Gender and Age: Was there a noticeable age difference between male and female passengers?
# perform T-test, since it is more direct and interpretable.

# Correlation Between Categorical Features
# Using Chi-Squared Test for independence and Cramér's V for strength of association.

# Feature Selection Techniques
# Filter Methods - Use statistical tests to select features that have the strongest relationship with the target variable
# Correlation Scores - Rank features and identify the most important ones for prediction.
# Wrapper Methods - Recursive Feature Elimination (RFE) rain a model on all features, rank them by importance, and then recursively remove the least important ones.
# Embedded Methods - Use algorithms that have built-in feature selection, such as Lasso (L1 regularization) add penalty or tree-based methods like Random Forests and Gradient Boosting.
# Dimensionality Reduction - Techniques like Principal Component Analysis (PCA) transform the features into a lower-dimensional space while retaining most of the variance in the data.
# Interaction Features - Create new features by combining existing ones, such as multiplying or adding them together



print(data.head())

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
log_reg.fit(X_train_scaled, y_train)  # scaled features for better convergence
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

# 4. Function to break the model
def break_model(X_train_df, y_train, X_test_df, y_test, corruption_type, model_class=LogisticRegression):
    """
    X_train_df, X_test_df should be pandas DataFrames (unscaled is fine)
    corruption_type: "shuffle_labels", "drop_age", "extreme_scaling", "combined"
    """
    X_train_corrupted = X_train_df.copy()
    X_test_corrupted = X_test_df.copy()
    y_train_corrupted = y_train.copy()

    # Apply corruption
    if corruption_type == "shuffle_labels":
        y_train_corrupted = np.random.permutation(y_train)
    elif corruption_type == "drop_age":
        X_train_corrupted = X_train_corrupted.drop(columns=["Age"])
        X_test_corrupted = X_test_corrupted.drop(columns=["Age"])
    elif corruption_type == "extreme_scaling":
        X_train_corrupted["Fare"] = X_train_corrupted["Fare"] * 1000
        X_test_corrupted["Fare"] = X_test_corrupted["Fare"] * 1000
    elif corruption_type == "combined":
        y_train_corrupted = np.random.permutation(y_train)
        X_train_corrupted = X_train_corrupted.drop(columns=["Age"])
        X_test_corrupted = X_test_corrupted.drop(columns=["Age"])
        X_train_corrupted["Fare"] = X_train_corrupted["Fare"] * 1000
        X_test_corrupted["Fare"] = X_test_corrupted["Fare"] * 1000
    else:
        raise ValueError("Invalid corruption type")

    # Train a new model for each corruption
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
