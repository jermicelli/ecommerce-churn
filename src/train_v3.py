"""Feature set v3: v2 minus the features SHAP showed as ~0. Trains the final random
forest, logs it to MLflow, and saves it for the API.

Run:  python src/train_v3.py   (MLflow server must be running on localhost:5000)
Output: models/churn_model.joblib
"""
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURE_SET = "v3"
# SHAP importance ~0 on v2. purchase_top_category is covered by category_repeat_rate.
DROPPED = ["purchase_top_category", "n_carted_not_bought", "n_cart_adds", "first_seen_day", "n_views"]
V2_ROC_AUC = 0.6995  # random_forest_none on v2, for comparison

df = pd.read_parquet("data/training_table_v2.parquet")
X = df.drop(columns=["user_id", "churned"] + DROPPED)   # only numbers left, so no get_dummies needed
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=20,
                               n_jobs=-1, random_state=42)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn_prediction")

with mlflow.start_run(run_name="random_forest_none_v3"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)

    mlflow.log_params({
        "model_type": "random_forest",
        "imbalance": "none",
        "feature_set": FEATURE_SET,
        "n_features": X.shape[1],
        "features": ", ".join(X.columns),
        "dropped": ", ".join(DROPPED),
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
    })
    mlflow.log_metrics({
        "accuracy":      accuracy_score(y_test, y_pred),
        "precision":     precision_score(y_test, y_pred),
        "recall":        recall_score(y_test, y_pred),
        "recall_stayed": recall_score(y_test, y_pred, pos_label=0),
        "f1":            f1_score(y_test, y_pred),
        "roc_auc":       roc_auc,
    })

print(f"{X.shape[1]} features: {list(X.columns)}")
print(f"ROC-AUC v3: {roc_auc:.4f}  (v2: {V2_ROC_AUC}, change: {roc_auc - V2_ROC_AUC:+.4f})")

# Save the model with its column order: the API must send features in exactly this order.
os.makedirs("models", exist_ok=True)
path = "models/churn_model.joblib"
joblib.dump({"model": model, "features": list(X.columns)}, path, compress=3)

# Check the saved file gives the same predictions as the model in memory.
saved = joblib.load(path)
# allclose, not ==: trees are summed in parallel, so the last decimal can differ between calls.
assert np.allclose(saved["model"].predict_proba(X_test[saved["features"]])[:, 1], y_proba)
print(f"saved {path} ({os.path.getsize(path) / 1e6:.0f} MB)")
