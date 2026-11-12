"""Agent decision service for the API.

Implements the complete agent pipeline:
Observation → Context → Decision → Permission → Execution → Verification → Event Logging
"""

from __future__ import annotations

from typing import Any

from src.agents.action_registry import ActionRegistry
from src.agents.context import ContextManager
from src.agents.decision_engine import DecisionEngine
from src.agents.permission_manager import PermissionManager, PermissionRequest
from src.agents.executor import ActionExecutor
from src.agents.verification import VerificationSystem
from src.agents.observation import Observation, ObservationType


class AgentService:
    """Wraps the autonomous agent pipeline for API requests.

    The complete pipeline:
    1. Receive and validate observation
    2. Update context
    3. Generate decision
    4. Check permission
    5. Execute permitted actions
    6. Verify result
    7. Return structured response
    """

    def __init__(self, confidence_threshold: float = 0.7, dry_run: bool = True) -> None:
        self.registry = ActionRegistry()
        self.context = ContextManager()
        self.decision_engine = DecisionEngine(
            confidence_threshold=confidence_threshold
        )
        self.permission_manager = PermissionManager(self.registry)
        self.executor = ActionExecutor(self.registry, dry_run=dry_run)
        self.verification = VerificationSystem()

    def decide(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the complete agent pipeline for a request.

        Args:
            request_data: Dictionary containing observation_source, observation_type,
                data, confidence, and optional context.

        Returns:
            Dictionary with decision_id, selected_action, confidence, explanation,
            permission_status, execution_status, verification_status, and timestamp.
        """
        # 1. Create observation
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

        # 2. Update context
        self.context.update(obs)
        context_snapshot = self.context.snapshot()

        # 3. Generate decision
        decision = self.decision_engine.decide(obs, context_snapshot)

        # 4. Check permission
        permission_request = PermissionRequest(
            action=decision.selected_action,
            requesting_component="agent_service",
            context={"observation_id": obs.observation_id, "decision_id": decision.decision_id},
        )
        permission_result = self.permission_manager.check_permission(permission_request)
        decision.permission_status = "granted" if permission_result.granted else "denied"

        # 5. Execute if permitted
        execution_result = None
        execution_status = "skipped"
        if permission_result.granted:
            execution_result = self.executor.execute(
                action=decision.selected_action,
                permission_result=permission_result,
                data=obs.data,
            )
            execution_status = "succeeded" if execution_result.succeeded else "failed"
        else:
            # Create a failed execution result for denied actions
            from src.agents.executor import ExecutionResult
            execution_result = ExecutionResult(
                action=decision.selected_action,
                succeeded=False,
                dry_run=self.executor.dry_run,
                message=f"Execution skipped: permission denied - {permission_result.reason}",
                error="permission_denied",
            )

        # 6. Verify if execution succeeded
        verification_status = "skipped"
        verification_record = None
        if execution_result and execution_result.succeeded:
            # Verify with independent state if available
            verification_record = self.verification.verify(execution_result)
            verification_status = verification_record.verification_status
            decision.verification_status = verification_status
        elif execution_result and not execution_result.succeeded:
            # Execution failed - verification status reflects this
            verification_status = "execution_failed"
            decision.verification_status = verification_status

        # 7. Return complete pipeline result
        return {
            "decision_id": decision.decision_id,
            "selected_action": decision.selected_action,
            "confidence": decision.confidence,
            "explanation": decision.explanation,
            "permission_status": decision.permission_status,
            "permission_reason": permission_result.reason,
            "execution_status": execution_status,
            "verification_status": verification_status,
            "timestamp": decision.timestamp,
            "execution_result": execution_result.to_dict() if execution_result else None,
            "verification_record": verification_record.to_dict() if verification_record else None,
        }
