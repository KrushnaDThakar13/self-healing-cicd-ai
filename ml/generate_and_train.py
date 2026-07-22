import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

np.random.seed(42)
N = 500  # number of synthetic past deployments

# --- Generate synthetic features ---
data = pd.DataFrame({
    "files_changed": np.random.randint(1, 50, N),
    "services_modified": np.random.randint(1, 6, N),
    "commit_hour": np.random.randint(0, 24, N),
    "test_coverage_delta": np.round(np.random.uniform(-15, 5, N), 2),
    "failed_tests": np.random.randint(0, 5, N),
    "build_duration_min": np.round(np.random.uniform(1, 20, N), 1),
    "prev_deploy_success_rate": np.round(np.random.uniform(0.5, 1.0, N), 2),
    "author_failure_rate": np.round(np.random.uniform(0.0, 0.4, N), 2),
    "days_since_last_deploy": np.random.randint(0, 14, N),
})

# --- Create a realistic risk formula to generate labels ---
risk_score = (
    data["files_changed"] * 0.02 +
    data["services_modified"] * 0.15 +
    (data["commit_hour"].apply(lambda h: 1 if h >= 22 or h <= 5 else 0)) * 0.8 +
    (-data["test_coverage_delta"]) * 0.05 +
    data["failed_tests"] * 0.6 +
    (1 - data["prev_deploy_success_rate"]) * 2 +
    data["author_failure_rate"] * 2 +
    (data["days_since_last_deploy"] > 7).astype(int) * 0.5 +
    np.random.normal(0, 0.5, N)  # noise
)
data["failed"] = (risk_score > risk_score.median()).astype(int)

# --- Train/test split ---
X = data.drop("failed", axis=1)
y = data["failed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Baseline: Logistic Regression ---
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
print(f"Logistic Regression (baseline) -> Accuracy: {accuracy_score(y_test, lr_preds):.2f}, F1: {f1_score(y_test, lr_preds):.2f}")

# --- Main model: Random Forest ---
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print(f"Random Forest (main model)     -> Accuracy: {accuracy_score(y_test, rf_preds):.2f}, F1: {f1_score(y_test, rf_preds):.2f}")

# --- Feature importance ---
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:")
print(importances)

# --- Save the model and dataset ---
joblib.dump(rf, "risk_model.pkl")
data.to_csv("synthetic_deployments.csv", index=False)
print("\nModel saved as risk_model.pkl")