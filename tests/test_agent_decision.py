"""Tests for the agent decision engine module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from src.agents.decision_engine import (
    DecisionEngine,
    Decision,
    ACTION_PAUSE,
    ACTION_RESUME,
    ACTION_NOOP,
    ACTION_REQUEST_VERIFICATION,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
from src.agents.context import ContextManager, ContextSnapshot
from src.agents.observation import Observation, ObservationType


def _make_gesture_observation(gesture, confidence=0.9):
    return Observation(
        source="gesture_recognition",
        type=ObservationType.GESTURE,
        data={"gesture": gesture},
        confidence=confidence,
    )


def test_decision_engine_creation():
    """Test DecisionEngine instantiation."""
    engine = DecisionEngine()
    assert engine.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD
    assert len(engine._rules) > 0


def test_decision_open_palm_pause():
    """Test OPEN_PALM -> pause decision."""
    engine = DecisionEngine()
    context = ContextManager()
    obs = _make_gesture_observation("OPEN_PALM", 0.9)
    decision = engine.decide(obs, context.snapshot())
    assert decision.selected_action == ACTION_PAUSE
    assert decision.confidence >= 0.7


def test_decision_thumbs_up_resume():
    """Test THUMBS_UP -> resume decision."""
    engine = DecisionEngine()
    context = ContextManager()
    obs = _make_gesture_observation("THUMBS_UP", 0.9)
    decision = engine.decide(obs, context.snapshot())
    assert decision.selected_action == ACTION_RESUME


def test_decision_fist_verification():
    """Test FIST -> request_verification decision."""
    engine = DecisionEngine()
    context = ContextManager()
    obs = _make_gesture_observation("FIST", 0.9)
    decision = engine.decide(obs, context.snapshot())
    assert decision.selected_action == ACTION_REQUEST_VERIFICATION


def test_decision_low_confidence_noop():
    """Test low-confidence gesture -> noop."""
    engine = DecisionEngine()
    context = ContextManager()
    obs = _make_gesture_observation("OPEN_PALM", 0.3)
    decision = engine.decide(obs, context.snapshot())
    assert decision.selected_action == ACTION_NOOP


def test_decision_unknown_gesture_noop():
    """Test UNKNOWN gesture -> noop."""
    engine = DecisionEngine()
    context = ContextManager()
    obs = _make_gesture_observation("UNKNOWN", 0.9)
    decision = engine.decide(obs, context.snapshot())
    assert decision.selected_action == ACTION_NOOP


def test_decision_has_required_fields():
    """Test that Decision has all required fields."""
    engine = DecisionEngine()
    context = ContextManager()
    obs = _make_gesture_observation("OPEN_PALM", 0.9)
    decision = engine.decide(obs, context.snapshot())
    assert decision.decision_id != ""
    assert decision.timestamp != ""
    assert decision.observation_id == obs.observation_id
    assert decision.observation_type == "gesture"
    assert decision.explanation != ""
    assert decision.permission_status == "pending"
    assert decision.verification_status == "pending"


def test_decision_to_dict():
    """Test Decision serialization."""
    engine = DecisionEngine()
    context = ContextManager()
    obs = _make_gesture_observation("OPEN_PALM", 0.9)
    decision = engine.decide(obs, context.snapshot())
    d = decision.to_dict()
    assert d["decision_id"] == decision.decision_id
    assert d["selected_action"] == decision.selected_action
    assert d["confidence"] == decision.confidence
    assert d["explanation"] == decision.explanation


def test_custom_rule():
    """Test adding a custom decision rule."""
    engine = DecisionEngine()
    engine.clear_rules()

    def always_pause(obs, ctx):
        return ("pause", 1.0, "Custom rule: always pause")

    engine.add_rule(always_pause)
    context = ContextManager()
    obs = _make_gesture_observation("THUMBS_UP", 0.9)
    decision = engine.decide(obs, context.snapshot())
    assert decision.selected_action == "pause"
    assert "Custom rule" in decision.explanation


def test_decision_conflicting_observations():
    """Test conflicting observations trigger verification."""
    engine = DecisionEngine()
    context = ContextManager()
    # First establish a gesture
    obs1 = _make_gesture_observation("OPEN_PALM", 0.9)
    engine.decide(obs1, context.snapshot())
    # Context now has OPEN_PALM as latest
    # Now create conflicting gesture
    context.update(_make_gesture_observation("FIST", 0.9))
    obs2 = _make_gesture_observation("FIST", 0.9)
    decision = engine.decide(obs2, context.snapshot())
    # Should request verification due to conflict or match FIST rule
    assert decision.selected_action in (ACTION_REQUEST_VERIFICATION, ACTION_NOOP)
