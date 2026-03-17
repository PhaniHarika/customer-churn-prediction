from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="Customer Churn Prediction API")

model = pickle.load(open('model/churn_model.pkl', 'rb'))
encoders = pickle.load(open('model/encoders.pkl', 'rb'))
feature_cols = pickle.load(open('model/feature_cols.pkl', 'rb'))

class Customer(BaseModel):
    gender: str
    senior_citizen: int
    partner: str
    dependents: str
    tenure: int
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float

@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    cat_cols = ['gender','partner','dependents','phone_service','multiple_lines',
                'internet_service','online_security','online_backup','device_protection',
                'tech_support','streaming_tv','streaming_movies','contract',
                'paperless_billing','payment_method']
    for col in cat_cols:
        le = encoders[col]
        df[col] = le.transform(df[col].astype(str))
    df = df[feature_cols]
    prob = model.predict_proba(df)[0][1]
    churn = "Yes" if prob >= 0.5 else "No"
    return {"churn_probability": round(float(prob), 4), "predicted_churn": churn}

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running!"}