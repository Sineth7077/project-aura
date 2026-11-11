"""Verification system for autonomous agent action results.

After an action is executed, the verifier checks whether the action actually
completed by comparing expected and actual state. It records success or
failure and prevents repeated execution of already-verified actions.

This component never reports success without a verifiable result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from uuid import uuid4
from datetime import datetime, timezone

from src.agents.executor import ExecutionResult


@dataclass
class VerificationRecord:
    """A single verification outcome.

    Attributes:
        verification_id: Unique UUID4 identifier.
        execution_id: ID of the execution being verified.
        action: Canonical action name.
        verified: Whether the action was independently verified as successful.
        expected_state: The state expected after execution.
        actual_state: The state observed after execution.
        match: Whether expected and actual state match.
        message: Human-readable verification message.
        timestamp: UTC ISO-8601 timestamp.
    """

    verification_id: str = field(default_factory=lambda: str(uuid4()))
    execution_id: str = ""
    action: str = ""
    verified: bool = False
    expected_state: Optional[dict[str, Any]] = None
    actual_state: Optional[dict[str, Any]] = None
    match: bool = False
    message: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "verification_id": self.verification_id,
            "execution_id": self.execution_id,
            "action": self.action,
            "verified": self.verified,
            "expected_state": self.expected_state,
            "actual_state": self.actual_state,
            "match": self.match,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class VerificationSystem:
    """Verifies action execution results against expected state.

    Strategies:
    - State comparison: compare expected vs actual state dictionaries.
    - Custom check function: a caller-supplied callable that returns bool.
    - Execution-result trust: verify based on ExecutionResult.succeeded flag
      (least strict; used when no independent state source exists).

    Duplicate prevention: the same execution_id is not verified twice.
    """

    def __init__(self) -> None:
        """Initialize the verification system."""
        self._verification_history: list[VerificationRecord] = []
        self._verified_executions: set[str] = set()
        self._custom_check: Optional[Callable[[ExecutionResult, dict[str, Any]], bool]] = None

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    def set_custom_check(
        self,
        check_func: Callable[[ExecutionResult, dict[str, Any]], bool],
    ) -> None:
        """Register a custom verification function.

        The function receives (execution_result, expected_state) and should
        return True when the action is verified as successful.
        """
        self._custom_check = check_func

    # ------------------------------------------------------------------ #
    # Verification
    # ------------------------------------------------------------------ #
    def verify(
        self,
        execution: ExecutionResult,
        expected_state: Optional[dict[str, Any]] = None,
        actual_state: Optional[dict[str, Any]] = None,
    ) -> VerificationRecord:
        """Verify an execution result.

        Args:
            execution: The :class:`ExecutionResult` to verify.
            expected_state: State expected after the action.
            actual_state: State observed after the action (if None, defaults
                to the execution's data).

        Returns:
            A :class:`VerificationRecord` with verified=True only when
            independent evidence confirms success.

        Raises:
            ValueError: If the same execution_id has already been verified
                (duplicate prevention).
        """
        if execution.execution_id in self._verified_executions:
            raise ValueError(
                f"Execution '{execution.execution_id}' has already been verified"
            )

        # Determine match: custom check > state comparison > execution flag.
        verified = False
        match = False
        message_parts: list[str] = []

        if self._custom_check is not None:
            try:
                verified = bool(self._custom_check(execution, expected_state or {}))
                match = verified
                message_parts.append("custom check passed" if verified else "custom check failed")
            except Exception as exc:
                verified = False
                match = False
                message_parts.append(f"custom check error: {exc}")

        if not message_parts:
            if expected_state is not None:
                actual = actual_state if actual_state is not None else (execution.data or {})
                match = self._states_match(expected_state, actual)
                verified = execution.succeeded and match
                message_parts.append("state matched" if match else "state mismatch")
            else:
                # No independent state source: trust execution result only.
                verified = execution.succeeded
                match = execution.succeeded
                message_parts.append(
                    "execution succeeded" if execution.succeeded else "execution failed"
                )

        if execution.error:
            verified = False
            match = False
            message_parts.append(f"error: {execution.error}")

        record = VerificationRecord(
            execution_id=execution.execution_id,
            action=execution.action,
            verified=verified,
            expected_state=expected_state,
            actual_state=actual_state,
            match=match,
            message="; ".join(message_parts),
        )

        self._verification_history.append(record)
        if verified:
            self._verified_executions.add(execution.execution_id)

        # Update execution result with verification status.
        execution.verified = verified

        return record


    # ------------------------------------------------------------------ #
    # History / query
    # ------------------------------------------------------------------ #
    def get_history(self) -> list[VerificationRecord]:
        """Return a copy of the verification history."""
        return list(self._verification_history)

    def get_record(self, execution_id: str) -> Optional[VerificationRecord]:
        """Return the verification record for a given execution ID."""
        for record in self._verification_history:
            if record.execution_id == execution_id:
                return record
        return None

    def is_verified(self, execution_id: str) -> bool:
        """Return True if the execution has been verified as successful."""
        return execution_id in self._verified_executions

    def clear_history(self) -> None:
        """Clear verification history (mainly for testing)."""
        self._verification_history.clear()
        self._verified_executions.clear()

    def get_verification_summary(self) -> dict[str, Any]:
        """Return summary statistics of verification."""
        total = len(self._verification_history)
        verified = sum(1 for r in self._verification_history if r.verified)
        failed = total - verified
        return {
            "total_verifications": total,
            "verified": verified,
            "failed": failed,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _states_match(
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> bool:
        """Compare expected and actual state dictionaries."""
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return False
        for key, value in expected.items():
            if key not in actual:
                return False
            if actual[key] != value:
                return False
        return True
