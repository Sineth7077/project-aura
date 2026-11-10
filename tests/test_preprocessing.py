"""Tests for image preprocessing module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from src.vision.preprocessing import (
    denormalize_image,
    normalize_image,
    prepare_batch,
    resize_image,
)


def test_resize_image_preserve_aspect_ratio():
    """Test resize with aspect ratio preservation."""
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = resize_image(image, target_size=(224, 224), preserve_aspect_ratio=True)
    assert resized.shape == (224, 224, 3)
    assert resized.dtype == np.uint8


def test_resize_image_no_preserve_aspect_ratio():
    """Test resize without aspect ratio preservation."""
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = resize_image(image, target_size=(224, 224), preserve_aspect_ratio=False)
    assert resized.shape == (224, 224, 3)


def test_normalize_image():
    """Test image normalization."""
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    normalized = normalize_image(image)
    assert normalized.dtype == np.float32
    assert normalized.shape == (224, 224, 3)


def test_prepare_batch():
    """Test batch preparation."""
    images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
    batch = prepare_batch(images, target_size=(224, 224))
    assert batch.shape == (4, 3, 224, 224)
    assert batch.dtype == np.float32


def test_denormalize_image():
    """Test denormalization."""
    normalized = np.random.randn(3, 224, 224).astype(np.float32)
    denormalized = denormalize_image(normalized)
    assert denormalized.dtype == np.uint8
    assert denormalized.shape == (224, 224, 3)
    assert denormalized.min() >= 0
    assert denormalized.max() <= 255


def test_resize_invalid_input():
    """Test resize raises on invalid input."""
    try:
        resize_image("not_an_array", target_size=(224, 224))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_prepare_batch_empty():
    """Test prepare_batch raises on empty list."""
    try:
        prepare_batch([], target_size=(224, 224))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass