"""
API Integration Tests using FastAPI TestClient.
Tests endpoint status codes, schema validation errors, and readiness probes.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Provides a TestClient instance."""
    with TestClient(app) as test_client:
        yield test_client


def test_liveness_probe(client):
    """Verifies that /api/v1/health/live returns 200 OK."""
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readiness_probe(client):
    """Verifies that /api/v1/health/ready returns 200 OK."""
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_predict_validation_error(client):
    """Verifies that malformed input features return 422 Unprocessable Entity."""
    # Sending missing features payload
    response = client.post("/api/v1/predict", json={"features": {}})
    assert response.status_code == 422


def test_predict_success(client):
    """Verifies that valid input features return a 200 prediction response."""
    payload = {
        "features": {
            "mean_radius": 17.99,
            "mean_texture": 10.38,
            "mean_perimeter": 122.8,
            "mean_area": 1001.0,
            "mean_smoothness": 0.1184,
        }
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in (0, 1)
    assert data["diagnosis"] in ("Benign", "Malignant")
    assert 0.0 <= data["probability"] <= 1.0
    assert data["model_version"] == "1.0.0"
