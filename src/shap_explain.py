import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_parquet("data/training_table_v2.parquet")
X = df.drop(columns=["user_id", "churned"])
X = pd.get_dummies(X, dtype=int)
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=20,
                               n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

X_sample = X_test.sample(2000, random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)
shap_churn = shap_values[:, :, 1]

shap.plots.bar(shap_churn, show=False, max_display=40)
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()

shap.plots.beeswarm(shap_churn, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

shap.plots.waterfall(shap_churn[0], show=False)
plt.tight_layout()
plt.savefig("shap_one_customer.png", dpi=150, bbox_inches="tight")
plt.close()

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run(run_id="95e22ea7d3cb4566917be9c6a2397c17"):
    mlflow.log_artifact("shap_importance.png", artifact_path="shap")
    mlflow.log_artifact("shap_beeswarm.png", artifact_path="shap")
    mlflow.log_artifact("shap_one_customer.png", artifact_path="shap")