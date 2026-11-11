"""Tests for the agent observation module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from src.agents.observation import (
    Observation,
    ObservationType,
    validate_observation,
)


def test_observation_creation():
    """Test Observation instantiation."""
    obs = Observation(
        source="gesture_recognition",
        type=ObservationType.GESTURE,
        data={"gesture": "OPEN_PALM"},
        confidence=0.9,
    )
    assert obs.observation_id != ""
    assert obs.source == "gesture_recognition"
    assert obs.type == ObservationType.GESTURE
    assert obs.data["gesture"] == "OPEN_PALM"
    assert obs.confidence == 0.9


def test_observation_to_dict():
    """Test Observation serialization."""
    obs = Observation(
        source="test",
        type=ObservationType.INFERENCE,
        data={"label": "cat"},
        confidence=0.85,
    )
    d = obs.to_dict()
    assert d["source"] == "test"
    assert d["type"] == "inference"
    assert d["data"]["label"] == "cat"
    assert d["confidence"] == 0.85


def test_observation_from_dict():
    """Test Observation deserialization."""
    data = {
        "observation_id": "test-id",
        "timestamp": "2024-01-01T00:00:00+00:00",
        "source": "test",
        "type": "gesture",
        "data": {"gesture": "FIST"},
        "confidence": 0.9,
        "metadata": {},
    }
    obs = Observation.from_dict(data)
    assert obs.observation_id == "test-id"
    assert obs.type == ObservationType.GESTURE
    assert obs.data["gesture"] == "FIST"


def test_validate_observation_valid():
    """Test validation of a valid observation."""
    obs = Observation(
        source="test",
        type=ObservationType.GESTURE,
        data={"gesture": "OPEN_PALM"},
        confidence=0.9,
    )
    errors = validate_observation(obs)
    assert errors == []


def test_validate_observation_empty_id():
    """Test validation rejects empty observation_id."""
    obs = Observation(
        source="test",
        type=ObservationType.GESTURE,
        data={},
    )
    obs.observation_id = ""
    errors = validate_observation(obs)
    assert len(errors) > 0


def test_validate_observation_invalid_confidence():
    """Test validation rejects confidence outside [0, 1]."""
    obs = Observation(
        source="test",
        type=ObservationType.GESTURE,
        data={},
        confidence=1.5,
    )
    errors = validate_observation(obs)
    assert len(errors) > 0


def test_validate_observation_missing_source():
    """Test validation rejects empty source."""
    obs = Observation(
        source="",
        type=ObservationType.GESTURE,
        data={},
    )
    errors = validate_observation(obs)
    assert len(errors) > 0


def test_observation_types():
    """Test all observation types are valid."""
    for obs_type in ObservationType:
        obs = Observation(source="test", type=obs_type, data={})
        errors = validate_observation(obs)
        assert errors == [], f"Validation failed for {obs_type}"
