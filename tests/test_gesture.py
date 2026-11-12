"""Tests for gesture recognition module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from src.vision.gesture_recognizer import GestureRecognizer, GesturePrediction


def test_gesture_prediction_to_dict():
    """Test GesturePrediction serialization."""
    result = GesturePrediction(
        gesture="OPEN_PALM",
        confidence=0.95,
        handedness="Right",
        landmarks=[(0.5, 0.5, 0.0)],
    )
    d = result.to_dict()
    assert d["gesture"] == "OPEN_PALM"
    assert d["confidence"] == 0.95
    assert d["handedness"] == "Right"


def test_gesture_recognizer_creation():
    """Test GestureRecognizer instantiation."""
    recognizer = GestureRecognizer()
    assert recognizer.confidence_threshold == 0.7
    assert recognizer.pinch_distance_threshold == 0.05


def test_classify_gesture_fist():
    """Test fist classification (all fingers closed)."""
    # All fingertips below MCPs (y increases downward in image coordinates)
    # Realistic fist geometry: fingertips lower than their corresponding MCP joints
    landmarks = [(0.5, 0.5, 0.0)] * 21
    # Set fingertips (indices 4, 8, 12, 16, 20) to be lower (higher y) than MCPs
    landmarks[4] = (0.3, 0.9, 0.0)  # thumb tip (far left to avoid pinch)
    landmarks[8] = (0.5, 0.9, 0.0)  # index tip
    landmarks[12] = (0.5, 0.9, 0.0)  # middle tip
    landmarks[16] = (0.5, 0.9, 0.0)  # ring tip
    landmarks[20] = (0.5, 0.9, 0.0)  # pinky tip
    # Set MCPs higher (lower y)
    landmarks[2] = (0.3, 0.5, 0.0)  # thumb MCP
    landmarks[5] = (0.5, 0.5, 0.0)  # index MCP
    landmarks[9] = (0.5, 0.5, 0.0)  # middle MCP
    landmarks[13] = (0.5, 0.5, 0.0)  # ring MCP
    landmarks[17] = (0.5, 0.5, 0.0)  # pinky MCP
    # Set intermediate joints
    landmarks[3] = (0.3, 0.7, 0.0)  # thumb IP
    landmarks[7] = (0.5, 0.7, 0.0)  # index PIP
    
    recognizer = GestureRecognizer()
    prediction = recognizer.recognize(landmarks, handedness="Right")
    assert prediction.gesture == "FIST"
    assert prediction.confidence >= 0.7


def test_classify_gesture_open_palm():
    """Test open palm classification (all fingers extended)."""
    # Realistic open palm geometry: fingertips above MCP joints
    landmarks = [(0.5, 0.5, 0.0)] * 21
    # Set fingertips (indices 4, 8, 12, 16, 20) to be higher (lower y) than MCPs
    # For right hand, thumb extended means tip[0] > IP[0]
    landmarks[4] = (0.7, 0.1, 0.0)  # thumb tip (right of IP for extension)
    landmarks[8] = (0.5, 0.1, 0.0)  # index tip
    landmarks[12] = (0.5, 0.1, 0.0)  # middle tip
    landmarks[16] = (0.5, 0.1, 0.0)  # ring tip
    landmarks[20] = (0.5, 0.1, 0.0)  # pinky tip
    # Set MCPs lower (higher y)
    landmarks[2] = (0.5, 0.5, 0.0)  # thumb MCP
    landmarks[5] = (0.5, 0.5, 0.0)  # index MCP
    landmarks[9] = (0.5, 0.5, 0.0)  # middle MCP
    landmarks[13] = (0.5, 0.5, 0.0)  # ring MCP
    landmarks[17] = (0.5, 0.5, 0.0)  # pinky MCP
    # Set intermediate joints
    landmarks[3] = (0.6, 0.3, 0.0)  # thumb IP (left of tip for right hand)
    landmarks[7] = (0.5, 0.3, 0.0)  # index PIP
    
    recognizer = GestureRecognizer()
    prediction = recognizer.recognize(landmarks, handedness="Right")
    assert prediction.gesture == "OPEN_PALM"
    assert prediction.confidence >= 0.7