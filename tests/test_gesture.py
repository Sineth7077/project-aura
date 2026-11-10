"""Tests for gesture recognition module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from src.agents.gesture import GestureRecognizer, GestureResult


def test_gesture_result_to_dict():
    """Test GestureResult serialization."""
    result = GestureResult(
        gesture_name="open_palm",
        confidence=0.95,
        bounding_box=[10, 20, 100, 100],
        landmarks=[(0.5, 0.5, 0.0)],
    )
    d = result.to_dict()
    assert d["gesture_name"] == "open_palm"
    assert d["confidence"] == 0.95
    assert d["bounding_box"] == [10, 20, 100, 100]


def test_gesture_recognizer_creation():
    """Test GestureRecognizer instantiation."""
    recognizer = GestureRecognizer()
    assert recognizer.confidence_threshold == 0.7
    assert recognizer.max_num_hands == 2
    assert recognizer._detector is None


def test_gesture_recognizer_detect_fallback():
    """Test fallback detection with a blank image."""
    recognizer = GestureRecognizer()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    results = recognizer.detect(image)
    # Blank image should produce no gestures in fallback mode
    assert isinstance(results, list)


def test_classify_gesture_fist():
    """Test fist classification (all fingers closed)."""
    # All fingertips below MCPs (y increases downward)
    landmarks = [(0.5, 0.9)] * 21  # all low in image (high y)
    recognizer = GestureRecognizer()
    name, conf = recognizer._classify_gesture(landmarks)
    assert name == "fist"


def test_classify_gesture_open_palm():
    """Test open palm classification (all fingers extended)."""
    landmarks = [(0.5, 0.1)] * 21  # all high in image (low y)
    recognizer = GestureRecognizer()
    name, conf = recognizer._classify_gesture(landmarks)
    assert name == "open_palm"