"""Tests for the gesture recognition module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from src.vision.gesture_recognizer import (
    GestureRecognizer,
    GesturePrediction,
    OPEN_PALM, FIST, THUMBS_UP, POINTING, VICTORY, PINCH, UNKNOWN,
    ALL_GESTURES,
)
from src.vision.landmark_processor import validate_landmarks


def _fist_landmarks():
    lm = [(0.5, 0.5, 0.0)] * 21
    lm[4] = (0.3, 0.9, 0.0); lm[3] = (0.4, 0.5, 0.0)
    lm[8] = (0.5, 0.9, 0.0); lm[7] = (0.5, 0.5, 0.0)
    lm[12] = (0.5, 0.9, 0.0); lm[11] = (0.5, 0.5, 0.0)
    lm[16] = (0.5, 0.9, 0.0); lm[15] = (0.5, 0.5, 0.0)
    lm[20] = (0.5, 0.9, 0.0); lm[19] = (0.5, 0.5, 0.0)
    return lm


def _open_palm_landmarks():
    lm = [(0.5, 0.5, 0.0)] * 21
    lm[4] = (0.7, 0.5, 0.0); lm[3] = (0.4, 0.5, 0.0)
    lm[8] = (0.5, 0.2, 0.0); lm[7] = (0.5, 0.5, 0.0)
    lm[12] = (0.5, 0.2, 0.0); lm[11] = (0.5, 0.5, 0.0)
    lm[16] = (0.5, 0.2, 0.0); lm[15] = (0.5, 0.5, 0.0)
    lm[20] = (0.5, 0.2, 0.0); lm[19] = (0.5, 0.5, 0.0)
    return lm


def _thumbs_up_landmarks():
    lm = [(0.5, 0.5, 0.0)] * 21
    lm[4] = (0.7, 0.3, 0.0); lm[3] = (0.4, 0.5, 0.0)
    lm[8] = (0.5, 0.9, 0.0); lm[7] = (0.5, 0.5, 0.0)
    lm[12] = (0.5, 0.9, 0.0); lm[11] = (0.5, 0.5, 0.0)
    lm[16] = (0.5, 0.9, 0.0); lm[15] = (0.5, 0.5, 0.0)
    lm[20] = (0.5, 0.9, 0.0); lm[19] = (0.5, 0.5, 0.0)
    return lm


def _pointing_landmarks():
    lm = [(0.5, 0.5, 0.0)] * 21
    lm[4] = (0.3, 0.5, 0.0); lm[3] = (0.4, 0.5, 0.0)
    lm[8] = (0.5, 0.2, 0.0); lm[7] = (0.5, 0.5, 0.0)
    lm[12] = (0.5, 0.9, 0.0); lm[11] = (0.5, 0.5, 0.0)
    lm[16] = (0.5, 0.9, 0.0); lm[15] = (0.5, 0.5, 0.0)
    lm[20] = (0.5, 0.9, 0.0); lm[19] = (0.5, 0.5, 0.0)
    return lm


def _victory_landmarks():
    lm = [(0.5, 0.5, 0.0)] * 21
    lm[4] = (0.3, 0.5, 0.0); lm[3] = (0.4, 0.5, 0.0)
    lm[8] = (0.5, 0.2, 0.0); lm[7] = (0.5, 0.5, 0.0)
    lm[12] = (0.5, 0.2, 0.0); lm[11] = (0.5, 0.5, 0.0)
    lm[16] = (0.5, 0.9, 0.0); lm[15] = (0.5, 0.5, 0.0)
    lm[20] = (0.5, 0.9, 0.0); lm[19] = (0.5, 0.5, 0.0)
    return lm


def _pinch_landmarks():
    lm = [(0.5, 0.5, 0.0)] * 21
    lm[4] = (0.5, 0.3, 0.0); lm[8] = (0.5, 0.3, 0.0)
    lm[12] = (0.5, 0.9, 0.0); lm[16] = (0.5, 0.9, 0.0)
    lm[20] = (0.5, 0.9, 0.0)
    return lm


def test_gesture_prediction_to_dict():
    pred = GesturePrediction(
        gesture=OPEN_PALM, confidence=0.95, handedness="Right",
        landmarks=[(0.5, 0.5, 0.0)] * 21, feature_scores={"num_extended": 5.0},
    )
    d = pred.to_dict()
    assert d["gesture"] == OPEN_PALM
    assert d["confidence"] == 0.95
    assert d["handedness"] == "Right"


def test_gesture_prediction_is_confident():
    pred = GesturePrediction(gesture=FIST, confidence=0.8)
    assert pred.is_confident(threshold=0.7) is True
    assert pred.is_confident(threshold=0.9) is False


def test_recognize_fist():
    recognizer = GestureRecognizer()
    pred = recognizer.recognize(_fist_landmarks(), handedness="Right")
    assert pred.gesture == FIST, f"Expected FIST, got {pred.gesture}"
    assert pred.confidence > 0.7


def test_recognize_open_palm():
    recognizer = GestureRecognizer()
    pred = recognizer.recognize(_open_palm_landmarks(), handedness="Right")
    assert pred.gesture == OPEN_PALM, f"Expected OPEN_PALM, got {pred.gesture}"
    assert pred.confidence > 0.7


def test_recognize_thumbs_up():
    recognizer = GestureRecognizer()
    pred = recognizer.recognize(_thumbs_up_landmarks(), handedness="Right")
    assert pred.gesture == THUMBS_UP, f"Expected THUMBS_UP, got {pred.gesture}"


def test_recognize_pointing():
    recognizer = GestureRecognizer()
    pred = recognizer.recognize(_pointing_landmarks(), handedness="Right")
    assert pred.gesture == POINTING, f"Expected POINTING, got {pred.gesture}"


def test_recognize_victory():
    recognizer = GestureRecognizer()
    pred = recognizer.recognize(_victory_landmarks(), handedness="Right")
    assert pred.gesture == VICTORY, f"Expected VICTORY, got {pred.gesture}"


def test_recognize_pinch():
    recognizer = GestureRecognizer()
    pred = recognizer.recognize(_pinch_landmarks(), handedness="Right")
    assert pred.gesture == PINCH, f"Expected PINCH, got {pred.gesture}"


def test_recognize_unknown_invalid_landmarks():
    recognizer = GestureRecognizer()
    with pytest.raises(Exception):
        recognizer.recognize([(0.5, 0.5)] * 5, handedness="Right")


def test_recognize_unknown_hand():
    recognizer = GestureRecognizer()
    landmarks = [(0.5, 0.5, 0.0)] * 21
    landmarks[8] = (0.5, 0.3, 0.0)
    landmarks[12] = (0.5, 0.4, 0.0)
    pred = recognizer.recognize(landmarks, handedness="Right")
    assert pred.gesture in (VICTORY, UNKNOWN, POINTING, OPEN_PALM)


def test_recognize_many():
    recognizer = GestureRecognizer()
    hands = [(_open_palm_landmarks(), "Right"), (_fist_landmarks(), "Right")]
    preds = recognizer.recognize_many(hands)
    assert len(preds) == 2
    assert preds[0].gesture == OPEN_PALM
    assert preds[1].gesture == FIST


def test_recognize_many_invalid_input():
    recognizer = GestureRecognizer()
    hands = [[(0.5, 0.5)] * 5, "Right"]
    preds = recognizer.recognize_many([hands])
    assert len(preds) == 1
    assert preds[0].gesture == UNKNOWN
    assert preds[0].confidence == 0.0


def test_validate_landmarks():
    assert validate_landmarks([(0.5, 0.5, 0.0)] * 21) is True
    assert validate_landmarks([(0.5, 0.5)] * 20) is False
    assert validate_landmarks([]) is False
    assert validate_landmarks("not a list") is False


def test_all_gestures_defined():
    assert OPEN_PALM == "OPEN_PALM"
    assert FIST == "FIST"
    assert THUMBS_UP == "THUMBS_UP"
    assert POINTING == "POINTING"
    assert VICTORY == "VICTORY"
    assert PINCH == "PINCH"
    assert UNKNOWN == "UNKNOWN"
    assert len(ALL_GESTURES) == 7

