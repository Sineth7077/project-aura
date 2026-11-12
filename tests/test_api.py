"""Tests for the FastAPI application."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.config import load_api_config


def test_health_endpoint():
    """Test GET /health."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "project-aura"


def test_ready_endpoint():
    """Test GET /ready."""
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "components" in data


def test_model_info_endpoint():
    """Test GET /model/info."""
    client = TestClient(app)
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "num_classes" in data
    assert "loaded" in data


def test_inference_invalid_image():
    """Test POST /inference with invalid image data."""
    client = TestClient(app)
    response = client.post(
        "/inference",
        json={"image_base64": "not_valid_base64!!", "format": "jpeg"},
    )
    assert response.status_code == 400


def test_gesture_invalid_image():
    """Test POST /gesture with invalid image data."""
    client = TestClient(app)
    response = client.post(
        "/gesture",
        json={"image_base64": "not_valid_base64!!", "format": "jpeg"},
    )
    assert response.status_code == 400


def test_agent_decide():
    """Test POST /agent/decide."""
    client = TestClient(app)
    response = client.post(
        "/agent/decide",
        json={
            "observation_source": "test",
            "observation_type": "gesture",
            "data": {"gesture": "OPEN_PALM"},
            "confidence": 0.9,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "decision_id" in data
    assert "selected_action" in data
    assert data["selected_action"] == "pause"


def test_agent_decide_unknown():
    """Test POST /agent/decide with UNKNOWN gesture."""
    client = TestClient(app)
    response = client.post(
        "/agent/decide",
        json={
            "observation_source": "test",
            "observation_type": "gesture",
            "data": {"gesture": "UNKNOWN"},
            "confidence": 0.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["selected_action"] == "noop"


def test_agent_decide_confidence_out_of_range():
    """Test POST /agent/decide rejects confidence outside [0, 1]."""
    client = TestClient(app)
    for bad_conf in [-0.1, 1.5]:
        response = client.post(
            "/agent/decide",
            json={
                "observation_source": "test",
                "observation_type": "gesture",
                "data": {"gesture": "OPEN_PALM"},
                "confidence": bad_conf,
            },
        )
        assert response.status_code == 422, f"confidence {bad_conf} should be rejected"


def test_agent_decide_confidence_in_range():
    """Test POST /agent/decide accepts confidence within [0, 1]."""
    client = TestClient(app)
    for good_conf in [0.0, 0.5, 1.0]:
        response = client.post(
            "/agent/decide",
            json={
                "observation_source": "test",
                "observation_type": "gesture",
                "data": {"gesture": "OPEN_PALM"},
                "confidence": good_conf,
            },
        )
        assert response.status_code == 200, f"confidence {good_conf} should be accepted"


def test_api_config_defaults():
    """Test API config loading."""
    config = load_api_config()
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.environment in ("development", "production")


def test_api_config_env_override(monkeypatch):
    """Test API config reads from environment variables."""
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("ENVIRONMENT", "production")
    config = load_api_config()
    assert config.port == 9000
    assert config.environment == "production"
    assert config.is_production() is True
