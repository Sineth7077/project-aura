"""Deterministic, explainable decision engine for the autonomous agent.

Maps standardized observations plus agent context into concrete action
decisions using configurable, human-readable rules. Every decision carries a
unique ID, the input observation, the selected action, a confidence score, an
explanation, a timestamp, and permission/verification status.

The engine is intentionally rule-based (not an LLM) so that behavior is
auditable, deterministic, and safe. Rules are expressed as simple predicate
functions registered in a rule list; the first matching rule wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import uuid4
from datetime import datetime, timezone

from src.agents.context import ContextSnapshot
from src.agents.observation import Observation, ObservationType

# Canonical action names (allowlist-aligned with ActionRegistry)
ACTION_PAUSE = "pause"
ACTION_RESUME = "resume"
ACTION_DISPLAY_ALERT = "display_alert"
ACTION_SAVE_SCREENSHOT = "save_screenshot"
ACTION_RECORD_EVENT = "record_event"
ACTION_UPDATE_VISUALIZATION = "update_visualization"
ACTION_NOOP = "noop"
ACTION_REQUEST_VERIFICATION = "request_verification"

DEFAULT_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class Decision:
    """A single agent decision.

    Attributes:
        decision_id: Unique UUID4 identifier.
        timestamp: UTC ISO-8601 timestamp.
        observation_id: ID of the observation that triggered this decision.
        observation_type: Type of the triggering observation.
        selected_action: Canonical action name.
        confidence: Decision quality / confidence in [0, 1].
        explanation: Human-readable explanation of why this action was chosen.
        permission_status: "pending", "granted", or "denied".
        verification_status: "pending", "verified", or "failed".
        context_snapshot: Serialized context at decision time.
    """

    decision_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    observation_id: str = ""
    observation_type: str = ""
    selected_action: str = ACTION_NOOP
    confidence: float = 0.0
    explanation: str = ""
    permission_status: str = "pending"
    verification_status: str = "pending"
    context_snapshot: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "observation_id": self.observation_id,
            "observation_type": self.observation_type,
            "selected_action": self.selected_action,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "permission_status": self.permission_status,
            "verification_status": self.verification_status,
            "context_snapshot": self.context_snapshot,
        }

# Rule predicate type: (observation, context) -> (action, confidence, explanation) or None
RuleFunc = Callable[[Observation, ContextSnapshot], Optional[tuple[str, float, str]]]

class DecisionEngine:
    """Deterministic rule-based decision engine.

    Rules are evaluated in registration order. The first rule whose predicate
    returns a non-None result produces the decision. This makes behavior easy
    to audit: you can inspect the rule list and see exactly why an action was
    selected.

    Example built-in rules:
    - Reliable OPEN_PALM   -> pause
    - Reliable THUMBS_UP   -> resume (confirm permitted operation)
    - Reliable FIST        -> request_verification (stop request)
    - Reliable POINTING    -> update_visualization (select target)
    - Low-confidence / unknown gesture -> noop
    - Conflicting observations -> request_verification
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """Initialize the decision engine.

        Args:
            confidence_threshold: Minimum gesture/observation confidence
                required for a gesture-based action to be considered reliable.
        """
        self.confidence_threshold = confidence_threshold
        self._rules: list[RuleFunc] = []
        self._register_default_rules()



    # ------------------------------------------------------------------ #
    # Rule registration
    # ------------------------------------------------------------------ #
    def _register_default_rules(self) -> None:
        """Register the default, documented decision rules."""
        self.add_rule(self._rule_open_palm_pause)
        self.add_rule(self._rule_thumbs_up_resume)
        self.add_rule(self._rule_fist_stop)
        self.add_rule(self._rule_pointing_select)
        self.add_rule(self._rule_unknown_or_low_confidence)
        self.add_rule(self._rule_conflicting_observations)
        self.add_rule(self._rule_default_noop)

    def add_rule(self, rule: RuleFunc) -> None:
        """Register a decision rule.

        Rules are evaluated in order of registration; the first match wins.
        """
        self._rules.append(rule)

    def clear_rules(self) -> None:
        """Remove all rules (useful for testing with custom rule sets)."""
        self._rules.clear()

    # ------------------------------------------------------------------ #
    # Decision
    # ------------------------------------------------------------------ #
    def decide(
        self,
        observation: Observation,
        context: ContextSnapshot,
    ) -> Decision:
        """Produce a decision for the given observation and context.

        Args:
            observation: The incoming validated observation.
            context: Current agent context snapshot.

        Returns:
            A :class:`Decision` with selected action, confidence, explanation,
            and pending permission/verification status.
        """
        decision = Decision(
            observation_id=observation.observation_id,
            observation_type=observation.type.value,
            selected_action=ACTION_NOOP,
            confidence=0.0,
            explanation="No matching rule; defaulting to noop.",
            context_snapshot=context.to_dict(),
        )

        for rule in self._rules:
            result = rule(observation, context)
            if result is None:
                continue
            action, confidence, explanation = result
            decision.selected_action = action
            decision.confidence = float(confidence)
            decision.explanation = explanation
            break

        return decision

    # ------------------------------------------------------------------ #
    # Built-in rules
    # ------------------------------------------------------------------ #
    def _rule_open_palm_pause(
        self,
        observation: Observation,
        context: ContextSnapshot,
    ) -> Optional[tuple[str, float, str]]:
        if observation.type != ObservationType.GESTURE:
            return None
        gesture = observation.data.get("gesture", "")
        conf = observation.confidence or 0.0
        if gesture == "OPEN_PALM" and conf >= self.confidence_threshold:
            return (
                ACTION_PAUSE,
                conf,
                f"Reliable OPEN_PALM detected (confidence {conf:.2f}); requesting pause.",
            )
        return None

    def _rule_thumbs_up_resume(
        self,
        observation: Observation,
        context: ContextSnapshot,
    ) -> Optional[tuple[str, float, str]]:
        if observation.type != ObservationType.GESTURE:
            return None
        gesture = observation.data.get("gesture", "")
        conf = observation.confidence or 0.0
        if gesture == "THUMBS_UP" and conf >= self.confidence_threshold:
            return (
                ACTION_RESUME,
                conf,
                f"Reliable THUMBS_UP detected (confidence {conf:.2f}); confirming permitted operation.",
            )
        return None

    def _rule_fist_stop(
        self,
        observation: Observation,
        context: ContextSnapshot,
    ) -> Optional[tuple[str, float, str]]:
        if observation.type != ObservationType.GESTURE:
            return None
        gesture = observation.data.get("gesture", "")
        conf = observation.confidence or 0.0
        if gesture == "FIST" and conf >= self.confidence_threshold:
            return (
                ACTION_REQUEST_VERIFICATION,
                conf,
                f"Reliable FIST detected (confidence {conf:.2f}); requesting stop verification.",
            )
        return None

    def _rule_pointing_select(
        self,
        observation: Observation,
        context: ContextSnapshot,
    ) -> Optional[tuple[str, float, str]]:
        if observation.type != ObservationType.GESTURE:
            return None
        gesture = observation.data.get("gesture", "")
        conf = observation.confidence or 0.0
        if gesture == "POINTING" and conf >= self.confidence_threshold:
            return (
                ACTION_UPDATE_VISUALIZATION,
                conf,
                f"Reliable POINTING detected (confidence {conf:.2f}); selecting interface target.",
            )
        return None

    def _rule_unknown_or_low_confidence(
        self,
        observation: Observation,
        context: ContextSnapshot,
    ) -> Optional[tuple[str, float, str]]:
        if observation.type != ObservationType.GESTURE:
            return None
        gesture = observation.data.get("gesture", "")
        conf = observation.confidence or 0.0
        if gesture == "UNKNOWN" or conf < self.confidence_threshold:
            return (
                ACTION_NOOP,
                conf,
                f"Gesture '{gesture}' is unknown or low-confidence ({conf:.2f}); doing nothing.",
            )
        return None

    def _rule_conflicting_observations(
        self,
        observation: Observation,
        context: ContextSnapshot,
    ) -> Optional[tuple[str, float, str]]:
        if observation.type == ObservationType.GESTURE:
            gesture = observation.data.get("gesture", "")
            if (
                gesture != "UNKNOWN"
                and context.previous_gesture != "UNKNOWN"
                and gesture != context.previous_gesture
            ):
                return (
                    ACTION_REQUEST_VERIFICATION,
                    0.6,
                    f"Conflicting observations: current '{gesture}' vs previous "
                    f"'{context.previous_gesture}'; requesting verification.",
                )
        return None

    def _rule_default_noop(
        self,
        observation: Observation,
        context: ContextSnapshot,
    ) -> Optional[tuple[str, float, str]]:
        return (
            ACTION_NOOP,
            0.5,
            "No specific rule matched; defaulting to noop.",
        )
