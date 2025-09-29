# Practice debugging model performance

# Understand how data leakage, label corruption, and feature removal affect models

# Build intuition for root cause analysis, a key ML engineer skill

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("Titanic.csv")
# Preview data
print(df.head())

# Fill missing numeric columns with median
for col in ["Age", "Fare"]:  # add any numeric columns with NaN
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical columns with mode
for col in ["Embarked"]:  # add any categorical columns with NaN
    df[col] = df[col].fillna(df[col].mode()[0])

# Prepare features and target
X = df[["Pclass", "Age", "SibSp", "Parch", "Fare"   ]]  # example
y = df["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model trained successfully!")

# Evaluate model
y_pred1 = model.predict(X_test)
Baseline_accuracy = accuracy_score(y_test, y_pred1)
print("Baseline Accuracy:", Baseline_accuracy)

# Step 3 – Break the Model on Purpose

# a. Shuffle Labels

import numpy as np

y_train_shuffled = np.random.permutation(y_train)
model.fit(X_train, y_train_shuffled)
y_pred2 = model.predict(X_test)


Shuffled_accuracy1 = accuracy_score(y_test, y_pred2)
print("Shuffled Labels Accuracy:", Shuffled_accuracy1)


#b. Drop Key Feature (Sex)

X_train_dropped = X_train.drop(columns=["Age"])  # example of dropping a feature
X_test_dropped = X_test.drop(columns= ["Age"])
model.fit(X_train_dropped, y_train)
y_pred3 = model.predict(X_test_dropped)


Shuffled_accuracy2 = accuracy_score(y_test, y_pred3)
print("Dropped Feature Accuracy:", Shuffled_accuracy2)

# c. Wrong Scaling (introduce extreme values)

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled["Fare"] = X_train_scaled["Fare"] * 1000  # extreme scaling
X_test_scaled["Fare"] = X_test_scaled["Fare"] * 1000
model.fit(X_train_scaled, y_train)
y_pred4 = model.predict(X_test_scaled)
Shuflled_accuracy3 = accuracy_score(y_test, y_pred4)
print("Extreme Scaling Accuracy:", Shuflled_accuracy3)


# d. Combining all the changes


# Drop Age
X_train_dropped_final = X_train.drop(columns=["Age"])
X_test_dropped_final = X_test.drop(columns=["Age"])

# Extreme scaling on Fare
X_train_dropped_final["Fare"] = X_train_dropped_final["Fare"] * 1000
X_test_dropped_final["Fare"] = X_test_dropped_final["Fare"] * 1000

# Shuffle labels
y_train_shuffled = np.random.permutation(y_train)

# Fit model
model.fit(X_train_dropped_final, y_train_shuffled)
y_pred5 = model.predict(X_test_dropped_final)

Shuffled_accuracy4 = accuracy_score(y_test, y_pred5)
print("Combined changed model accuracy:", Shuffled_accuracy4)


# Comparing results 

print("Baseline Accuracy:" ,Baseline_accuracy)
print("Shuffled Labels Accuracy:", Shuffled_accuracy1)
print("Dropped Feature Accuracy:", Shuffled_accuracy2)
print("Extreme Scaling Accuracy:", Shuflled_accuracy3)
print("Combined changed model accuracy:", Shuffled_accuracy4)
