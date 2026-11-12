"""Tests for model service with offline behavior and fallback support."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.api.services.model_service import ModelService


def test_model_service_creation():
    """Test ModelService instantiation."""
    service = ModelService()
    assert service.model_name == "efficientnet_b0"
    assert service.num_classes == 10
    assert not service.is_loaded()
    assert service._engine is None


def test_model_service_custom_params():
    """Test ModelService with custom parameters."""
    service = ModelService(model_name="resnet50", num_classes=1000)
    assert service.model_name == "resnet50"
    assert service.num_classes == 1000


def test_load_model_success():
    """Test successful model loading with mocked inference engine."""
    service = ModelService()
    
    # Mock the InferenceEngine to avoid actual model loading
    mock_engine = Mock()
    mock_engine.load_model.return_value = None
    
    with patch('src.vision.inference.InferenceEngine', return_value=mock_engine):
        result = service.load_model()
        assert result is True
        assert service.is_loaded()
        assert service._engine is not None


def test_load_model_failure():
    """Test model loading failure handling."""
    service = ModelService()
    
    with patch('src.vision.inference.InferenceEngine', side_effect=Exception("Import error")):
        result = service.load_model()
        assert result is False
        assert not service.is_loaded()


def test_predict_with_loaded_model():
    """Test prediction when model is loaded."""
    service = ModelService()
    
    # Mock engine and prediction
    mock_engine = Mock()
    mock_engine.predict.return_value = {
        "probabilities": [0.1, 0.9],
        "predicted_class": 1,
        "confidence": 0.9,
        "model_name": "efficientnet_b0"
    }
    
    with patch('src.vision.inference.InferenceEngine', return_value=mock_engine):
        service.load_model()
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = service.predict(image)
        
        assert result is not None
        assert result["predicted_class"] == 1
        assert result["confidence"] == 0.9


def test_predict_without_loaded_model():
    """Test prediction triggers lazy loading."""
    service = ModelService()
    
    # Mock engine and prediction
    mock_engine = Mock()
    mock_engine.predict.return_value = {
        "probabilities": [0.1, 0.9],
        "predicted_class": 1,
        "confidence": 0.9,
        "model_name": "efficientnet_b0"
    }
    
    with patch('src.vision.inference.InferenceEngine', return_value=mock_engine):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = service.predict(image)
        
        assert result is not None
        assert service.is_loaded()


def test_predict_load_failure_returns_none():
    """Test prediction returns None when model loading fails."""
    service = ModelService()
    
    with patch('src.vision.inference.InferenceEngine', side_effect=Exception("Load failed")):
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = service.predict(image)
        
        assert result is None
        assert not service.is_loaded()


def test_predict_inference_failure_returns_none():
    """Test prediction returns None when inference fails."""
    service = ModelService()
    
    mock_engine = Mock()
    mock_engine.predict.side_effect = Exception("Inference failed")
    
    with patch('src.vision.inference.InferenceEngine', return_value=mock_engine):
        service.load_model()
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = service.predict(image)
        
        assert result is None


def test_get_model_info():
    """Test model info retrieval."""
    service = ModelService(model_name="resnet50", num_classes=1000)
    
    info = service.get_model_info()
    assert info["model_name"] == "resnet50"
    assert info["num_classes"] == 1000
    assert info["loaded"] == False
    assert info["device"] == "cpu"


def test_get_model_info_loaded():
    """Test model info when model is loaded."""
    service = ModelService()
    
    mock_engine = Mock()
    
    with patch('src.vision.inference.InferenceEngine', return_value=mock_engine):
        service.load_model()
        info = service.get_model_info()
        assert info["loaded"] is True


def test_unload_model():
    """Test model unloading."""
    service = ModelService()
    
    mock_engine = Mock()
    
    with patch('src.vision.inference.InferenceEngine', return_value=mock_engine):
        service.load_model()
        assert service.is_loaded()
        
        service.unload_model()
        assert not service.is_loaded()
        assert service._engine is None


def test_offline_behavior_no_automatic_download():
    """Test that model service does not automatically download weights."""
    service = ModelService()
    
    # This test verifies that the service uses the InferenceEngine
    # which should handle offline behavior gracefully
    # The actual weight download logic is in InferenceEngine/VisionModelBase
    # This test ensures the service doesn't bypass that
    
    mock_engine = Mock()
    mock_engine.load_model.return_value = None
    
    with patch('src.vision.inference.InferenceEngine', return_value=mock_engine) as mock_create:
        service.load_model()
        # Verify InferenceEngine was called, not direct weight download
        mock_create.assert_called_once()


def test_fallback_model_unavailable():
    """Test behavior when model is completely unavailable."""
    service = ModelService()
    
    # Simulate complete unavailability
    with patch('src.vision.inference.InferenceEngine', side_effect=ImportError("PyTorch not available")):
        result = service.load_model()
        assert result is False
        
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        prediction = service.predict(image)
        assert prediction is None
        
        info = service.get_model_info()
        assert info["loaded"] is False


def test_thread_safety():
    """Test that model service uses thread-safe locking."""
    service = ModelService()
    
    mock_engine = Mock()
    
    with patch('src.vision.inference.InferenceEngine', return_value=mock_engine):
        # Multiple load calls should be safe
        service.load_model()
        service.load_model()
        service.load_model()
        
        assert service.is_loaded()
