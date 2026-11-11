"""Structured gesture event generation.

Defines the canonical event schema used by Notebook 05 (gesture recognition)
and consumed by Notebook 06 (autonomous agent). The schema is intentionally
flat and JSON-serializable so it can cross notebook, process, and network
boundaries without ambiguity.

Event fields:
- event_id:      Unique UUID4 string.
- timestamp:     UTC ISO-8601 timestamp.
- gesture:       Canonical gesture name (OPEN_PALM, FIST, etc.).
- hand:          Hand identifier ("Left", "Right", or "Unknown").
- confidence:    Recognition confidence in [0, 1].
- landmarks:     Normalized landmark list (optional).
- bbox:          Pixel bounding box [xmin, ymin, xmax, ymax] (optional).
- prev_gesture:  Previous stabilized gesture (for transition analysis).
- curr_gesture:  Current stabilized gesture.
- metadata:      Extra free-form context (source, frame_id, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4
from datetime import datetime, timezone


@dataclass
class GestureEvent:
    """A single structured gesture event.

    Attributes:
        event_id: Unique identifier (auto-generated).
        timestamp: UTC ISO-8601 timestamp string.
        gesture: Current gesture name.
        hand: Hand identifier ("Left", "Right", "Unknown").
        confidence: Recognition confidence in [0, 1].
        landmarks: Normalized 21-landmark list (optional).
        bbox: Pixel bounding box [xmin, ymin, xmax, ymax] (optional).
        prev_gesture: Previous stabilized gesture name.
        curr_gesture: Current stabilized gesture name.
        metadata: Extra context dictionary.
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    gesture: str = "UNKNOWN"
    hand: str = "Unknown"
    confidence: float = 0.0
    landmarks: Optional[list[tuple[float, float, float]]] = None
    bbox: Optional[list[int]] = None
    prev_gesture: str = "UNKNOWN"
    curr_gesture: str = "UNKNOWN"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "gesture": self.gesture,
            "hand": self.hand,
            "confidence": self.confidence,
            "landmarks": self.landmarks,
            "bbox": self.bbox,
            "prev_gesture": self.prev_gesture,
            "curr_gesture": self.curr_gesture,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GestureEvent":
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            event_id=data.get("event_id", str(uuid4())),
            timestamp=data.get(
                "timestamp", datetime.now(timezone.utc).isoformat()
            ),
            gesture=data.get("gesture", "UNKNOWN"),
            hand=data.get("hand", "Unknown"),
            confidence=float(data.get("confidence", 0.0)),
            landmarks=data.get("landmarks"),
            bbox=data.get("bbox"),
            prev_gesture=data.get("prev_gesture", "UNKNOWN"),
            curr_gesture=data.get("curr_gesture", "UNKNOWN"),
            metadata=data.get("metadata", {}),
        )


class GestureEventGenerator:
    """Builds :class:`GestureEvent` objects from smoothed gesture results.

    Tracks the previous gesture across calls so each event carries both
    ``prev_gesture`` and ``curr_gesture`` for transition analysis.
    """

    def __init__(self, source: str = "gesture_recognition") -> None:
        """Initialize the event generator.

        Args:
            source: Identifier for the event source (e.g. "webcam",
                "video_file", "uploaded_image").
        """
        self.source = source
        self._prev_gesture: str = "UNKNOWN"

    def create_event(
        self,
        gesture: str,
        confidence: float,
        hand: str = "Unknown",
        landmarks: Optional[list[tuple[float, float, float]]] = None,
        bbox: Optional[list[int]] = None,
        metadata: Optional[dict] = None,
    ) -> GestureEvent:
        """Create a new gesture event.

        Args:
            gesture: Current stabilized gesture name.
            confidence: Confidence score in [0, 1].
            hand: Hand identifier.
            landmarks: Optional normalized landmarks.
            bbox: Optional pixel bounding box.
            metadata: Optional extra context.

        Returns:
            A populated :class:`GestureEvent`.
        """
        event = GestureEvent(
            gesture=gesture,
            confidence=confidence,
            hand=hand,
            landmarks=landmarks,
            bbox=bbox,
            prev_gesture=self._prev_gesture,
            curr_gesture=gesture,
            metadata=metadata or {},
        )
        # Enrich metadata with source info.
        event.metadata["source"] = self.source
        # Update previous gesture for the next call.
        self._prev_gesture = gesture
        return event

    def reset(self) -> None:
        """Reset the previous-gesture tracker."""
        self._prev_gesture = "UNKNOWN"
