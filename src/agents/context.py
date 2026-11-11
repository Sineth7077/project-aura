"""Context manager for the autonomous AI agent.

Maintains short-term, bounded context from observations: detected objects,
person count, latest gesture, previous gesture, previous actions, recent
events, and timestamps. Context entries expire after a configurable TTL so
stale information does not influence decisions indefinitely.

Image data is never stored in memory; only lightweight summaries and
references are retained.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, timezone

from src.agents.observation import Observation, ObservationType


@dataclass
class ContextSnapshot:
    """Immutable snapshot of the current agent context."""

    detected_objects: list[str] = field(default_factory=list)
    person_count: Optional[int] = None
    latest_gesture: str = "UNKNOWN"
    previous_gesture: str = "UNKNOWN"
    previous_actions: list[str] = field(default_factory=list)
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ttl_seconds: float = 30.0

    def is_expired(self) -> bool:
        try:
            ts = datetime.fromisoformat(self.timestamp)
        except (ValueError, TypeError):
            return True
        now = datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds() > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_objects": self.detected_objects,
            "person_count": self.person_count,
            "latest_gesture": self.latest_gesture,
            "previous_gesture": self.previous_gesture,
            "previous_actions": list(self.previous_actions),
            "recent_events": list(self.recent_events),
            "timestamp": self.timestamp,
            "ttl_seconds": self.ttl_seconds,
            "expired": self.is_expired(),
        }


class ContextManager:
    def __init__(self, ttl_seconds: float = 30.0, max_events: int = 50) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_events = max_events
        self._detected_objects: list[str] = []
        self._person_count: Optional[int] = None
        self._latest_gesture: str = "UNKNOWN"
        self._previous_gesture: str = "UNKNOWN"
        self._previous_actions: list[str] = []
        self._recent_events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._last_update: Optional[str] = None

    def update(self, observation: Observation) -> None:
        from src.agents.observation import validate_observation

        errors = validate_observation(observation)
        if errors:
            self._record_event(
                "observation_rejected",
                {"errors": errors, "source": observation.source},
            )
            return

        self._last_update = observation.timestamp

        if observation.type == ObservationType.INFERENCE:
            label = observation.data.get("label")
            if label:
                self._detected_objects = [label]
                self._record_event(
                    "object_detected",
                    {"label": label, "confidence": observation.confidence},
                )
        elif observation.type == ObservationType.OBJECT_DETECTION:
            labels = observation.data.get("labels", [])
            if isinstance(labels, list):
                self._detected_objects = [str(l) for l in labels]
            self._record_event(
                "objects_detected",
                {"count": len(labels), "labels": labels},
            )
        elif observation.type == ObservationType.PERSON_COUNT:
            count = observation.data.get("count")
            if isinstance(count, (int, float)):
                self._person_count = int(count)
            self._record_event(
                "person_count_updated",
                {"count": self._person_count},
            )
        elif observation.type == ObservationType.GESTURE:
            gesture = observation.data.get("gesture", "UNKNOWN")
            if gesture and gesture != self._latest_gesture:
                self._previous_gesture = self._latest_gesture
                self._latest_gesture = gesture
            self._record_event(
                "gesture_recognized",
                {"gesture": gesture, "confidence": observation.confidence},
            )
        elif observation.type == ObservationType.SYSTEM_STATE:
            self._record_event(
                "system_state",
                observation.data,
            )

    def record_action(self, action_name: str) -> None:
        self._previous_actions.append(action_name)
        if len(self._previous_actions) > 20:
            self._previous_actions = self._previous_actions[-20:]
        self._record_event("action_executed", {"action": action_name})

    def _record_event(self, event_type: str, data: dict[str, Any]) -> None:
        self._recent_events.append(
            {
                "type": event_type,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def snapshot(self) -> ContextSnapshot:
        return ContextSnapshot(
            detected_objects=list(self._detected_objects),
            person_count=self._person_count,
            latest_gesture=self._latest_gesture,
            previous_gesture=self._previous_gesture,
            previous_actions=list(self._previous_actions),
            recent_events=list(self._recent_events),
            timestamp=self._last_update or datetime.now(timezone.utc).isoformat(),
            ttl_seconds=self.ttl_seconds,
        )

    def latest_gesture(self) -> str:
        return self._latest_gesture

    def previous_gesture(self) -> str:
        return self._previous_gesture

    def reset(self) -> None:
        self._detected_objects = []
        self._person_count = None
        self._latest_gesture = "UNKNOWN"
        self._previous_gesture = "UNKNOWN"
        self._previous_actions = []
        self._recent_events.clear()
        self._last_update = None

