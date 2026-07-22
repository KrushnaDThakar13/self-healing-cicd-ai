import joblib
import pandas as pd
import sys
import json

# Load the trained model
model = joblib.load("risk_model.pkl")

def predict_risk(features: dict):
    df = pd.DataFrame([features])
    proba = model.predict_proba(df)[0][1]  # probability of "failed"
    risk_percent = round(proba * 100, 1)

    if risk_percent >= 60:
        level = "HIGH"
    elif risk_percent >= 30:
        level = "MEDIUM"
    else:
        level = "LOW"

    return risk_percent, level

if __name__ == "__main__":
    example_commit = {
        "files_changed": 42,
        "services_modified": 4,
        "commit_hour": 23,
        "test_coverage_delta": -8.5,
        "failed_tests": 2,
        "build_duration_min": 12.3,
        "prev_deploy_success_rate": 0.65,
        "author_failure_rate": 0.25,
        "days_since_last_deploy": 9,
    }

    risk_percent, level = predict_risk(example_commit)
    result = {"risk_percent": risk_percent, "risk_level": level}
    print(json.dumps(result, indent=2))

    if level == "HIGH":
        sys.exit(1)
    else:
        sys.exit(0)