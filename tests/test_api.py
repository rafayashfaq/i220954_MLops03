"""
Unit tests for FastAPI Inference Microservice (api/main.py).
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app_version" in data


def test_api_version_endpoint():
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["service_name"] == "churn-prediction-api"
    assert data["app_version"] == "v1.2.0"
    assert data["model_name"] == "churn-classifier-prod"


def test_api_prediction_endpoint():
    payload = {
        "account_age_months": 24,
        "monthly_charges": 65.50,
        "total_logins": 18,
        "support_tickets": 1,
        "service_failures": 0,
        "latency_avg_ms": 140.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert data["churn_prediction"] in {0, 1}
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert data["inference_latency_ms"] >= 0.0


def test_api_prediction_validation_error():
    """Invalid input schema must return HTTP 422 Unprocessable Entity."""
    bad_payload = {
        "account_age_months": -5,  # Invalid negative age
        "monthly_charges": "not-a-number"
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
