import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("Titanic.csv")

# Select simple features
features = ["Pclass", "Sex", "Age", "SibSp", "Fare"]
df = df[features + ["Survived"]]

# Encode categorical
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model training and evaluation
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Baseline Accuracy:", accuracy_score(y_test, y_pred))
