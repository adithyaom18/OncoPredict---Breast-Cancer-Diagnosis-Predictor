# model/main.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/data.csv")

# Remove unnamed columns
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Encode target variable
df["diagnosis"] = df["diagnosis"].map({"B": 0, "M": 1})

# Drop ID column if exists
df.drop(columns=["id"], errors="ignore", inplace=True)

# Split features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# -----------------------------
# Handle Missing Values
# -----------------------------
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# -----------------------------
# Feature Scaling
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Logistic Regression + GridSearch
# -----------------------------
param_grid = {
    "C": [0.1, 1.0, 10],
    "solver": ["lbfgs", "liblinear"]
}

logreg = LogisticRegression(max_iter=1000)

grid = GridSearchCV(
    logreg,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# -----------------------------
# Evaluation
# -----------------------------
y_pred = best_model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "best_params": grid.best_params_
}

# -----------------------------
# Save Artifacts
# -----------------------------
with open("model/logreg.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/imputer.pkl", "wb") as f:
    pickle.dump(imputer, f)

with open("model/metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

with open("model/scores.pkl", "wb") as f:
    pickle.dump({"logreg": metrics["accuracy"]}, f)

print("✅ Best Logistic Regression model saved successfully")
print("Accuracy:", metrics["accuracy"])
print("Best Parameters:", metrics["best_params"])
