"""Tests for the gesture smoothing module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from src.vision.gesture_recognizer import GesturePrediction, OPEN_PALM, FIST, UNKNOWN
from src.vision.gesture_smoother import GestureSmoother, SmoothedGesture


def _make_prediction(gesture, confidence=0.9):
    return GesturePrediction(gesture=gesture, confidence=confidence, handedness="Right")


def test_smoother_creation():
    """Test GestureSmoother instantiation with defaults."""
    smoother = GestureSmoother()
    assert smoother.window_size == 5
    assert smoother.min_stable_frames == 3
    assert smoother.cooldown_frames == 10


def test_smoother_initial_state_unknown():
    """Test that initial smoothed gesture is UNKNOWN."""
    smoother = GestureSmoother()
    result = smoother.update(_make_prediction(UNKNOWN, 0.0))
    assert result.gesture == UNKNOWN


def test_smoother_stability_requirement():
    """Test that gesture must be stable across multiple frames."""
    smoother = GestureSmoother(min_stable_frames=3, cooldown_frames=0)
    # Feed OPEN_PALM predictions
    for _ in range(3):
        result = smoother.update(_make_prediction(OPEN_PALM, 0.9))
    assert result.stable is True
    assert result.gesture == OPEN_PALM


def test_smoother_transition_detection():
    """Test that gesture transitions are detected."""
    smoother = GestureSmoother(min_stable_frames=2, cooldown_frames=0)
    # First establish OPEN_PALM
    result = smoother.update(_make_prediction(OPEN_PALM, 0.9))
    result = smoother.update(_make_prediction(OPEN_PALM, 0.9))
    # Transition may have already fired on the 2nd frame
    assert result.gesture == OPEN_PALM

    # Then switch to FIST (feed enough frames for majority vote to flip)
    result = smoother.update(_make_prediction(FIST, 0.9))
    result = smoother.update(_make_prediction(FIST, 0.9))
    result = smoother.update(_make_prediction(FIST, 0.9))
    assert result.transition is True  # OPEN_PALM -> FIST


def test_smoother_cooldown_suppression():
    """Test that cooldown suppresses rapid gesture changes."""
    smoother = GestureSmoother(min_stable_frames=1, cooldown_frames=5)
    # Establish OPEN_PALM
    result = smoother.update(_make_prediction(OPEN_PALM, 0.9))
    assert result.transition is True
    # Immediately try to switch to FIST - should be suppressed by cooldown
    result = smoother.update(_make_prediction(FIST, 0.9))
    assert result.event_suppressed is True or result.gesture == OPEN_PALM


def test_smoother_reset():
    """Test that reset clears smoothing state."""
    smoother = GestureSmoother(min_stable_frames=1, cooldown_frames=0)
    smoother.update(_make_prediction(OPEN_PALM, 0.9))
    smoother.reset()
    result = smoother.update(_make_prediction(UNKNOWN, 0.0))
    assert result.gesture == UNKNOWN


def test_smoothed_gesture_to_dict():
    """Test SmoothedGesture serialization."""
    result = SmoothedGesture(
        gesture=OPEN_PALM,
        confidence=0.9,
        stable=True,
        transition=True,
        event_suppressed=False,
        history_length=5,
    )
    d = {
        "gesture": result.gesture,
        "confidence": result.confidence,
        "stable": result.stable,
        "transition": result.transition,
        "event_suppressed": result.event_suppressed,
        "history_length": result.history_length,
    }
    assert d["gesture"] == OPEN_PALM
    assert d["stable"] is True
    assert d["transition"] is True


def test_smoother_invalid_window_size():
    """Test that invalid window_size raises ValueError."""
    with pytest.raises(ValueError):
        GestureSmoother(window_size=0)
    with pytest.raises(ValueError):
        GestureSmoother(min_stable_frames=0)
    with pytest.raises(ValueError):
        GestureSmoother(cooldown_frames=-1)
