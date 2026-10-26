"""Churn prediction API.

Run (from the project folder):  uvicorn app.main:app --reload
Test page:                       http://127.0.0.1:8000/docs
"""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

saved = joblib.load("models/churn_model.joblib")  # loaded once, when the server starts
model, FEATURES = saved["model"], saved["features"]
THRESHOLD = 0.5  # flag as churn when probability >= this; see README for how to pick it

app = FastAPI(title="Churn prediction API")


class Customer(BaseModel):
    """One October buyer's 18 features. Ranges match the checks in data_process_v2.py."""
    days_since_last_purchase: int = Field(ge=1, le=31)
    days_since_last_visit: int = Field(ge=1, le=31)
    n_purchases: int = Field(ge=1)
    n_sessions: int = Field(ge=1)
    n_active_days: int = Field(ge=1, le=31)
    total_spent: float = Field(ge=0)
    avg_purchase_price: float = Field(ge=0)
    n_unique_products: int = Field(ge=0)
    n_unique_brands: int = Field(ge=0)
    n_unique_categories: int = Field(ge=0)
    events_last_7_days: int = Field(ge=0)
    conversion_rate: float = Field(ge=0)
    category_repeat_rate: float = Field(ge=0, le=1)
    brand_repeat_rate: float = Field(ge=0, le=1)
    views_after_last_purchase: int = Field(ge=0)
    same_category_views_after_purchase: int = Field(ge=0)
    views_before_first_purchase: int = Field(ge=0)
    days_first_to_last_visit: int = Field(ge=0, le=30)


# Fail at start-up, not on the first request, if the model and the input fields don't match.
assert set(Customer.model_fields) == set(FEATURES), "Customer fields differ from the model's features"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])[FEATURES]  # same column order as training
    prob = float(model.predict_proba(row)[0, 1])
    return {
        "churn_probability": round(prob, 3),
        "churn_prediction": prob >= THRESHOLD,
        "threshold": THRESHOLD,
    }
