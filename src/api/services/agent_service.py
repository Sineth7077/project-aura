"""Agent decision service for the API."""

from __future__ import annotations

from typing import Any


class AgentService:
    """Wraps the autonomous agent pipeline for API requests."""

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        from src.agents.action_registry import ActionRegistry
        from src.agents.context import ContextManager
        from src.agents.decision_engine import DecisionEngine
        from src.agents.permission_manager import PermissionManager
        from src.agents.executor import ActionExecutor
        from src.agents.verification import VerificationSystem

        self.registry = ActionRegistry()
        self.context = ContextManager()
        self.decision_engine = DecisionEngine(
            confidence_threshold=confidence_threshold
        )
        self.permission_manager = PermissionManager(self.registry)
        self.executor = ActionExecutor(self.registry, dry_run=True)
        self.verification = VerificationSystem()

    def decide(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Produce an agent decision from request data."""
        from src.agents.observation import Observation, ObservationType

        try:
            obs_type = ObservationType(request_data.get("observation_type", "custom"))
        except ValueError:
            obs_type = ObservationType.CUSTOM

        obs = Observation(
            source=request_data.get("observation_source", "api"),
            type=obs_type,
            data=request_data.get("data", {}),
            confidence=request_data.get("confidence"),
        )

        self.context.update(obs)
        context_snapshot = self.context.snapshot()
        decision = self.decision_engine.decide(obs, context_snapshot)

        return {
            "decision_id": decision.decision_id,
            "selected_action": decision.selected_action,
            "confidence": decision.confidence,
            "explanation": decision.explanation,
            "permission_status": decision.permission_status,
            "timestamp": decision.timestamp,
        }
