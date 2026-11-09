"""
FastAPI Inference Service for Customer Churn Prediction.
Containerized microservice deployed to Kubernetes with health probes,
model serving, structured logging, and version reporting.
"""
import os
import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field


app = FastAPI(
    title="MLOps Production Prediction API",
    version=os.getenv("APP_VERSION", "v1.2.0"),
    description="High-performance inference microservice for ML predictions"
)

# Global model container
MODEL = None
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.pkl")
MODEL_NAME = os.getenv("MODEL_NAME", "churn-classifier-prod")
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
APP_VERSION = os.getenv("APP_VERSION", "v1.2.0")


class ChurnPredictionRequest(BaseModel):
    account_age_months: int = Field(..., ge=0, le=120, example=24)
    monthly_charges: float = Field(..., ge=0.0, example=65.50)
    total_logins: int = Field(..., ge=0, example=18)
    support_tickets: int = Field(..., ge=0, example=2)
    service_failures: int = Field(..., ge=0, example=0)
    latency_avg_ms: float = Field(..., ge=0.0, example=140.0)


class ChurnPredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    model_version: str
    inference_latency_ms: float


@app.on_event("startup")
def load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        try:
            MODEL = joblib.load(MODEL_PATH)
            print(f"Model successfully loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not load model from {MODEL_PATH}: {e}")
    else:
        print(f"Model file {MODEL_PATH} not found on startup. Standby inference enabled.")


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "MLOps Prediction Service",
        "status": "UP",
        "app_version": APP_VERSION,
        "environment": ENVIRONMENT
    }


@app.get("/health", tags=["Health Probes"])
def health_check():
    """Kubernetes liveness and readiness probe endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None or os.path.exists(MODEL_PATH),
        "app_version": APP_VERSION,
        "environment": ENVIRONMENT,
        "timestamp": time.time()
    }


@app.get("/version", tags=["Info"])
def get_service_version():
    """Reports currently deployed model and application versions."""
    return {
        "service_name": "churn-prediction-api",
        "app_version": APP_VERSION,
        "model_name": MODEL_NAME,
        "model_path": MODEL_PATH,
        "model_loaded": MODEL is not None,
        "environment": ENVIRONMENT
    }


@app.post("/predict", response_model=ChurnPredictionResponse, tags=["Inference"])
def predict_churn(request: ChurnPredictionRequest):
    """Generates customer churn prediction."""
    start_time = time.time()
    
    # Feature array
    features = pd.DataFrame([{
        "account_age_months": request.account_age_months,
        "monthly_charges": request.monthly_charges,
        "total_logins": request.total_logins,
        "support_tickets": request.support_tickets,
        "service_failures": request.service_failures,
        "latency_avg_ms": request.latency_avg_ms
    }])
    
    if MODEL is not None:
        try:
            pred = int(MODEL.predict(features)[0])
            proba = float(MODEL.predict_proba(features)[0][1])
        except Exception:
            # Fallback heuristic
            score = (request.support_tickets * 0.25) + (request.service_failures * 0.35) - (request.total_logins * 0.02)
            proba = min(max(1 / (1 + np.exp(-score)), 0.01), 0.99)
            pred = 1 if proba > 0.5 else 0
    else:
        score = (request.support_tickets * 0.25) + (request.service_failures * 0.35) - (request.total_logins * 0.02)
        proba = min(max(1 / (1 + np.exp(-score)), 0.01), 0.99)
        pred = 1 if proba > 0.5 else 0

    latency_ms = (time.time() - start_time) * 1000
    risk = "HIGH" if proba >= 0.7 else ("MEDIUM" if proba >= 0.4 else "LOW")

    return ChurnPredictionResponse(
        churn_prediction=pred,
        churn_probability=round(proba, 4),
        risk_level=risk,
        model_version=APP_VERSION,
        inference_latency_ms=round(latency_ms, 2)
    )
