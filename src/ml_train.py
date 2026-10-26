import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost



df = pd.read_parquet("data/training_table.parquet")
X = df.drop(columns=["user_id", "churned"])   # the 15 features
y = df["churned"]                             # predict label

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,  # 80% train, 20% test 
                                                    stratify=y,     # Keep same proportion of churned users in train and test
                                                    random_state=42)


models = {
    "baseline":      DummyClassifier(strategy="most_frequent"),
    "logistic":      make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                                   LogisticRegression(max_iter=1000, random_state=42)),
    "random_forest": RandomForestClassifier(n_estimators=100, max_depth=25, min_samples_leaf=20,
                                            n_jobs=-1, random_state=42),
    "xgboost":       XGBClassifier(n_estimators=100, n_jobs=-1, random_state=42),
}

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn_prediction")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"\n===== {name} =====")
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        mlflow.log_param("model_type", name)
        mlflow.log_param("imbalance", "none")
        if name in ("random_forest", "xgboost"):
            mlflow.log_param("n_estimators", model.n_estimators)
        if name == "random_forest":
            mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_metrics({
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_proba),
        })
        if name == "xgboost":
                mlflow.xgboost.log_model(model, name="model")
        else:
                mlflow.sklearn.log_model(model, name="model",
                                         skops_trusted_types=["sklearn.tree._tree.Tree"])


# baseline_model = DummyClassifier(strategy="most_frequent")  # predict the most frequent class
# baseline_model.fit(X_train, y_train)
# print("Baseline model accuracy on test set: {:.2f}%".format(baseline_model.score(X_test, y_test) * 100))

# # Model 1: Logistic Regression
# logistic_model = make_pipeline(FunctionTransformer(np.log1p),
#                                StandardScaler(),
#                                LogisticRegression(random_state=42))

# logistic_model.fit(X_train, y_train)
# y_pred_logistic = logistic_model.predict(X_test)
# y_pred_proba_logistic = logistic_model.predict_proba(X_test)[:, 1]
# print("Logistic Regression model accuracy on test set: {:.2f}%".format(logistic_model.score(X_test, y_test) * 100))

# print("Classification report for Logistic Regression model:")
# print(classification_report(y_test, y_pred_logistic))

# print("Confusion matrix for Logistic Regression model:")
# print(confusion_matrix(y_test, y_pred_logistic))

# print("ROC AUC score for Logistic Regression model: {:.2f}".format(roc_auc_score(y_test, y_pred_proba_logistic)))

# # Model 2: Random Forest
# rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
# rf_model.fit(X_train, y_train)
# y_pred_rf = rf_model.predict(X_test)
# y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
# print("Random Forest model accuracy on test set: {:.2f}%".format(rf_model.score(X_test, y_test) * 100))

# print("Classification report for Random Forest model:")
# print(classification_report(y_test, y_pred_rf))

# print("Confusion matrix for Random Forest model:")
# print(confusion_matrix(y_test, y_pred_rf))

# print("ROC AUC score for Random Forest model: {:.2f}".format(roc_auc_score(y_test, y_pred_proba_rf)))

# # Model 3: XGBoost
# xgb_model = XGBClassifier(n_estimators=100, n_jobs=-1, random_state=42)
# xgb_model.fit(X_train, y_train)
# y_pred_xgb = xgb_model.predict(X_test)
# y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
# print("XGBoost model accuracy on test set: {:.2f}%".format(xgb_model.score(X_test, y_test) * 100))

# print("Classification report for XGBoost model:")
# print(classification_report(y_test, y_pred_xgb))

# print("Confusion matrix for XGBoost model:")
# print(confusion_matrix(y_test, y_pred_xgb))

# print("ROC AUC score for XGBoost model: {:.2f}".format(roc_auc_score(y_test, y_pred_proba_xgb)))
