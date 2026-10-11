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


def test_predict_validation_error(client):
    """Verifies that malformed input features return 422 Unprocessable Entity."""
    # Sending missing features payload
    response = client.post("/api/v1/predict", json={"features": {}})
    assert response.status_code == 422
