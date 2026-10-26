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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
import mlflow




FEATURE_SET = "v2"                                   # "v1" = 14 features, "v2" = 14 + 9 new
DATA_PATH = {"v1": "data/training_table.parquet",
             "v2": "data/training_table_v2.parquet"}[FEATURE_SET]

df = pd.read_parquet(DATA_PATH)
X = df.drop(columns=["user_id", "churned"])
X = pd.get_dummies(X, dtype=int)   # text columns (purchase_top_category) -> one 0/1 column per value
y = df["churned"]                             # predict label

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,  # 80% train, 20% test 
                                                    stratify=y,     # Keep same proportion of churned users in train and test
                                                    random_state=42)

ratio = (y_train == 0).sum() / (y_train == 1).sum()
print(ratio)

experiments = [
    ("baseline", "none", DummyClassifier(strategy="most_frequent")),

    # --- Logistic regression ---
    ("logistic", "none",
        make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                      LogisticRegression(max_iter=1000, random_state=42))),

    ("logistic", "class_weight",
        make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                      LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))),

    ("logistic", "smote",
        imbalanced_make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                          SMOTE(random_state=42),
                          LogisticRegression(max_iter=1000, random_state=42))),

    # --- Random forest ---
    ("random_forest", "none",
        RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=20,
                               n_jobs=-1, random_state=42)),

    ("random_forest", "class_weight",
        RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=20,
                               class_weight="balanced", n_jobs=-1, random_state=42)),

    ("random_forest", "smote",
        imbalanced_make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                          SMOTE(random_state=42),
                          RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=20,
                                                 n_jobs=-1, random_state=42))),

    # --- XGBoost ---
    ("xgboost", "none",
        XGBClassifier(n_estimators=300, n_jobs=-1, random_state=42)),

    ("xgboost", "class_weight",
        XGBClassifier(n_estimators=300, scale_pos_weight=ratio, n_jobs=-1, random_state=42)),

    ("xgboost", "smote",
        imbalanced_make_pipeline(FunctionTransformer(np.log1p), StandardScaler(),
                          SMOTE(random_state=42),
                          XGBClassifier(n_estimators=300, n_jobs=-1, random_state=42))),
]


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn_prediction")

for name, imbalance, model in experiments:
    with mlflow.start_run(run_name=f"{name}_{imbalance}"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print(f"\n===== {name}/{imbalance} =====")
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))

        mlflow.log_param("model_type", name)
        mlflow.log_param("imbalance", imbalance)
        mlflow.log_param("feature_set", FEATURE_SET)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("features", ", ".join(X.columns))
        final = model[-1] if hasattr(model, "steps") else model  
        if name in ("random_forest", "xgboost"):
            mlflow.log_param("n_estimators", final.n_estimators)
        if name == "random_forest":
            mlflow.log_param("max_depth", final.max_depth)
        mlflow.log_metrics({
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "recall_stayed": recall_score(y_test, y_pred, pos_label=0),
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_proba),
        })


