"""Tests for custom exception hierarchy."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.core.exceptions import (
    AgentOrchestrationError,
    AuraException,
    CloudConnectionError,
    ConfigurationError,
    DataIngestionError,
    GestureRecognitionError,
    InferenceError,
    ModelNotFoundError,
)


def test_aura_exception_base():
    """Test base AuraException."""
    exc = AuraException("test error", error_code="TEST_001")
    assert str(exc) == "[TEST_001] test error"
    assert exc.message == "test error"
    assert exc.error_code == "TEST_001"


def test_aura_exception_no_code():
    """Test AuraException without error code."""
    exc = AuraException("test error")
    # The implementation defaults error_code to "AURA_UNKNOWN" when None is passed
    assert str(exc) == "[AURA_UNKNOWN] test error"


def test_configuration_error():
    """Test ConfigurationError."""
    exc = ConfigurationError("bad config")
    assert isinstance(exc, AuraException)
    assert exc.error_code == "AURA_CONFIG_ERROR"


def test_model_not_found_error():
    """Test ModelNotFoundError."""
    exc = ModelNotFoundError("resnet50")
    assert "resnet50" in str(exc)
    assert exc.error_code == "AURA_MODEL_NOT_FOUND"


def test_inference_error():
    """Test InferenceError."""
    exc = InferenceError("inference failed")
    assert exc.error_code == "AURA_INFERENCE_ERROR"


def test_data_ingestion_error():
    """Test DataIngestionError."""
    exc = DataIngestionError("data error")
    assert exc.error_code == "AURA_DATA_ERROR"


def test_cloud_connection_error():
    """Test CloudConnectionError."""
    exc = CloudConnectionError("connection failed")
    assert exc.error_code == "AURA_CLOUD_ERROR"


def test_gesture_recognition_error():
    """Test GestureRecognitionError."""
    exc = GestureRecognitionError("gesture failed")
    assert exc.error_code == "AURA_GESTURE_ERROR"


def test_agent_orchestration_error():
    """Test AgentOrchestrationError."""
    exc = AgentOrchestrationError("agent failed")
    assert exc.error_code == "AURA_AGENT_ERROR"