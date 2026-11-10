"""Tests for inference engine module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from src.core.config import VisionConfig
from src.vision.inference import InferenceEngine


def test_inference_engine_creation():
    """Test InferenceEngine instantiation."""
    config = VisionConfig(num_classes=5, model_name="efficientnet_b0")
    engine = InferenceEngine(config=config)
    assert engine.model_name == "efficientnet_b0"
    assert engine._model is None


def test_preprocess_single_image():
    """Test preprocessing a single image."""
    config = VisionConfig(input_size=(224, 224))
    engine = InferenceEngine(config=config)
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    processed = engine.preprocess(image)
    assert processed.shape == (3, 224, 224)
    assert processed.dtype == np.float32


def test_preprocess_batch():
    """Test preprocessing a batch of images."""
    config = VisionConfig(input_size=(224, 224))
    engine = InferenceEngine(config=config)
    images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
    batch = engine.preprocess_batch(images)
    assert batch.shape == (3, 3, 224, 224)
    assert batch.dtype == np.float32


def test_predict_with_fallback_model():
    """Test prediction using fallback model (no PyTorch)."""
    config = VisionConfig(num_classes=3, model_name="efficientnet_b0")
    engine = InferenceEngine(config=config)
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = engine.predict(image)
    assert "probabilities" in result
    assert "predicted_class" in result
    assert "confidence" in result
    assert "model_name" in result
    assert len(result["probabilities"]) == 3
    assert 0 <= result["predicted_class"] < 3
    assert 0 <= result["confidence"] <= 1.0