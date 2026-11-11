"""Controlled action executor for the autonomous agent.

Executes allowlisted actions only after permission has been granted. Supports
dry-run mode (no side effects), structured result returns, exception handling,
and execution-status recording. This is the only component that performs
side effects; everything upstream is pure decision-making.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import uuid4
from datetime import datetime, timezone

from src.agents.action_registry import ActionRegistry
from src.agents.permission_manager import PermissionRequest, PermissionResult


@dataclass
class ExecutionResult:
    """Structured result of an action execution attempt.

    Attributes:
        execution_id: Unique UUID4 identifier.
        action: Canonical action name.
        succeeded: Whether the action completed successfully.
        dry_run: Whether this was a dry-run (no side effects).
        message: Human-readable result or error message.
        data: Optional data returned by the action handler.
        error: Optional exception message if execution failed.
        timestamp: UTC ISO-8601 timestamp.
        verified: Whether the result was independently verified (set by
            the verification system, not here).
    """

    execution_id: str = field(default_factory=lambda: str(uuid4()))
    action: str = ""
    succeeded: bool = False
    dry_run: bool = False
    message: str = ""
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "execution_id": self.execution_id,
            "action": self.action,
            "succeeded": self.succeeded,
            "dry_run": self.dry_run,
            "message": self.message,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp,
            "verified": self.verified,
        }


class ActionExecutor:
    """Executes registered actions with validation, permissions, and dry-run support.

    Responsibilities:
    - Validate that an action is allowlisted and executable.
    - Check permissions before execution.
    - Run the action handler with exception handling.
    - Return structured results (never raise unhandled exceptions).
    - Support dry-run mode for safe previews.
    """

    def __init__(
        self,
        action_registry: Optional[ActionRegistry] = None,
        dry_run: bool = False,
    ) -> None:
        """Initialize the executor.

        Args:
            action_registry: The allowlist to consult.
            dry_run: If True, all executions are simulated with no side effects.
        """
        self.registry = action_registry or ActionRegistry()
        self.dry_run = dry_run
        self._execution_history: list[ExecutionResult] = []

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def execute(
        self,
        action: str,
        permission_result: PermissionResult,
        data: Optional[dict[str, Any]] = None,
        dry_run: Optional[bool] = None,
    ) -> ExecutionResult:
        """Execute an allowlisted action.

        Args:
            action: Canonical action name.
            permission_result: A :class:`PermissionResult` with granted=True.
            data: Optional input data for the action handler.
            dry_run: Override the instance-level dry_run setting for this call.

        Returns:
            An :class:`ExecutionResult` with structured outcome information.
        """
        is_dry_run = self.dry_run if dry_run is None else dry_run
        execution_id = str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # Permission must be granted.
        if not permission_result.granted:
            result = ExecutionResult(
                action=action,
                succeeded=False,
                dry_run=is_dry_run,
                message="Execution rejected: permission not granted",
                error="permission_denied",
                timestamp=timestamp,
            )
            self._execution_history.append(result)
            return result

        # Action must be allowlisted and executable.
        if not self.registry.is_allowlisted(action):
            result = ExecutionResult(
                action=action,
                succeeded=False,
                dry_run=is_dry_run,
                message=f"Action '{action}' is not allowlisted",
                error="not_allowlisted",
                timestamp=timestamp,
            )
            self._execution_history.append(result)
            return result

        entry = self.registry.get(action)
        if entry is None:
            result = ExecutionResult(
                action=action,
                succeeded=False,
                dry_run=is_dry_run,
                message=f"Action '{action}' is not allowlisted",
                error="not_allowlisted",
                timestamp=timestamp,
            )
            self._execution_history.append(result)
            return result

        # Dry-run: simulate without invoking the handler.
        # For dry-run, handler presence is not required.
        if is_dry_run:
            if not self.registry.supports_dry_run(action):
                result = ExecutionResult(
                    action=action,
                    succeeded=False,
                    dry_run=True,
                    message=f"Action '{action}' does not support dry-run mode",
                    error="dry_run_not_supported",
                    timestamp=timestamp,
                )
                self._execution_history.append(result)
                return result
            result = ExecutionResult(
                action=action,
                succeeded=True,
                dry_run=True,
                message=f"Dry-run: would execute '{action}'",
                data=data or {},
                timestamp=timestamp,
            )
            self._execution_history.append(result)
            return result

        # Live execution with exception handling.
        if entry.handler is None:
            result = ExecutionResult(
                action=action,
                succeeded=False,
                dry_run=False,
                message=f"Action '{action}' has no executable handler",
                error="no_handler",
                timestamp=timestamp,
            )
            self._execution_history.append(result)
            return result

        try:
            handler = entry.handler
            handler_result = handler(data or {})
            handler_data = handler_result if isinstance(handler_result, dict) else {"result": handler_result}
            result = ExecutionResult(
                action=action,
                succeeded=True,
                dry_run=False,
                message=f"Action '{action}' executed successfully",
                data=handler_data,
                timestamp=timestamp,
            )
        except Exception as exc:
            result = ExecutionResult(
                action=action,
                succeeded=False,
                dry_run=False,
                message=f"Action '{action}' failed: {exc}",
                error=str(exc),
                timestamp=timestamp,
            )

        self._execution_history.append(result)
        return result

    # ------------------------------------------------------------------ #
    # History / query
    # ------------------------------------------------------------------ #
    def get_history(self) -> list[ExecutionResult]:
        """Return a copy of the execution history."""
        return list(self._execution_history)

    def get_last_result(self) -> Optional[ExecutionResult]:
        """Return the most recent execution result, or None."""
        if not self._execution_history:
            return None
        return self._execution_history[-1]

    def clear_history(self) -> None:
        """Clear execution history (mainly for testing)."""
        self._execution_history.clear()

    def set_dry_run(self, enabled: bool) -> None:
        """Enable or disable dry-run mode for all subsequent executions."""
        self.dry_run = enabled

    def get_execution_summary(self) -> dict[str, Any]:
        """Return summary statistics of the execution history."""
        total = len(self._execution_history)
        succeeded = sum(1 for r in self._execution_history if r.succeeded)
        failed = total - succeeded
        dry_runs = sum(1 for r in self._execution_history if r.dry_run)
        return {
            "total_executions": total,
            "succeeded": succeeded,
            "failed": failed,
            "dry_runs": dry_runs,
        }
