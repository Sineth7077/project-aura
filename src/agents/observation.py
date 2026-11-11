"""Standardized observation schema for the autonomous agent.

Defines the :class:`Observation` dataclass and validation helpers that form
the contract between computer vision outputs (Notebooks 04/05) and the agent
pipeline (Notebook 06).

Every observation carries:
- observation_id:  Unique UUID4 identifier.
- timestamp:       UTC ISO-8601 timestamp.
- source:          Originating component (e.g. "inference_engine", "gesture_recognition").
- type:            ObservationType enum value.
- data:            Typed payload dict.
- confidence:      Confidence in [0, 1] (optional).
- metadata:        Extra free-form context.

Validation ensures required fields are present and types are correct before
the observation enters the context manager or decision engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
from uuid import uuid4
from datetime import datetime, timezone


class ObservationType(str, Enum):
    """Categories of observations the agent can receive."""

    INFERENCE = "inference"          # Object / image classification result
    GESTURE = "gesture"              # Hand gesture recognition result
    OBJECT_DETECTION = "object_detection"  # Object detection result
    PERSON_COUNT = "person_count"    # Count of people in scene
    SYSTEM_STATE = "system_state"    # Internal system state change
    CUSTOM = "custom"                # Application-specific observation


@dataclass
class Observation:
    """A standardized observation fed into the agent pipeline.

    Attributes:
        observation_id: Unique identifier (auto-generated).
        timestamp: UTC ISO-8601 timestamp.
        source: Component that produced the observation.
        type: ObservationType category.
        data: Typed payload dictionary.
        confidence: Confidence score in [0, 1] (optional).
        metadata: Extra context dictionary.
    """

    observation_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = "unknown"
    type: ObservationType = ObservationType.CUSTOM
    data: dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "observation_id": self.observation_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "type": self.type.value,
            "data": self.data,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            observation_id=data.get("observation_id", str(uuid4())),
            timestamp=data.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            source=data.get("source", "unknown"),
            type=ObservationType(data.get("type", ObservationType.CUSTOM.value)),
            data=data.get("data", {}),
            confidence=data.get("confidence"),
            metadata=data.get("metadata", {}),
        )


def validate_observation(obs: Observation) -> list[str]:
    """Validate an observation and return a list of human-readable errors.

    An empty list means the observation is valid. This function never raises;
    callers decide how to handle the error list.

    Checks performed:
    - observation_id is a non-empty string.
    - timestamp is a non-empty string.
    - source is a non-empty string.
    - type is a valid ObservationType.
    - data is a dict.
    - confidence, when present, is a number in [0, 1].
    """
    errors: list[str] = []

    if not isinstance(obs.observation_id, str) or not obs.observation_id:
        errors.append("observation_id must be a non-empty string")
    if not isinstance(obs.timestamp, str) or not obs.timestamp:
        errors.append("timestamp must be a non-empty string")
    if not isinstance(obs.source, str) or not obs.source:
        errors.append("source must be a non-empty string")
    if not isinstance(obs.type, ObservationType):
        errors.append(f"type must be a valid ObservationType, got {type(obs.type)}")
    if not isinstance(obs.data, dict):
        errors.append("data must be a dictionary")
    if obs.confidence is not None:
        if not isinstance(obs.confidence, (int, float)):
            errors.append("confidence must be a number")
        elif not (0.0 <= float(obs.confidence) <= 1.0):
            errors.append("confidence must be in [0, 1]")
    if not isinstance(obs.metadata, dict):
        errors.append("metadata must be a dictionary")

    return errors
