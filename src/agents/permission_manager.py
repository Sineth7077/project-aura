"""Permission manager for the autonomous agent.

Enforces the action allowlist, validates permission requests, handles user
confirmation for sensitive actions, and maintains an audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4
from datetime import datetime, timezone

from src.agents.action_registry import ActionRegistry


@dataclass
class PermissionRequest:
    """A request to execute an action."""

    request_id: str = field(default_factory=lambda: str(uuid4()))
    action: str = ""
    requesting_component: str = "unknown"
    context: dict[str, Any] = field(default_factory=dict)
    confirmed_by_user: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class PermissionResult:
    """Outcome of a permission check."""

    request_id: str
    action: str
    granted: bool
    reason: str
    requires_confirmation: bool = False
    audit_entry: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "action": self.action,
            "granted": self.granted,
            "reason": self.reason,
            "requires_confirmation": self.requires_confirmation,
            "audit_entry": self.audit_entry,
        }


class PermissionManager:
    """Validates action permissions and maintains an audit trail."""

    def __init__(self, action_registry: Optional[ActionRegistry] = None) -> None:
        self.registry = action_registry or ActionRegistry()
        self._audit_log: list[dict[str, Any]] = []

    def check_permission(self, request: PermissionRequest) -> PermissionResult:
        audit_entry: dict[str, Any] = {
            "request_id": request.request_id,
            "action": request.action,
            "requesting_component": request.requesting_component,
            "timestamp": request.timestamp,
        }

        if not self.registry.is_allowlisted(request.action):
            reason = f"Action '{request.action}' is not allowlisted"
            result = PermissionResult(
                request_id=request.request_id,
                action=request.action,
                granted=False,
                reason=reason,
                requires_confirmation=False,
                audit_entry={**audit_entry, "decision": "denied", "reason": reason},
            )
            self._audit_log.append(result.audit_entry or {})
            return result

        if self.registry.is_sensitive(request.action):
            if not request.confirmed_by_user:
                reason = (
                    f"Action '{request.action}' is sensitive and requires "
                    "explicit user confirmation"
                )
                result = PermissionResult(
                    request_id=request.request_id,
                    action=request.action,
                    granted=False,
                    reason=reason,
                    requires_confirmation=True,
                    audit_entry={
                        **audit_entry,
                        "decision": "pending_confirmation",
                        "reason": reason,
                    },
                )
                self._audit_log.append(result.audit_entry or {})
                return result
            reason = f"Action '{request.action}' approved with user confirmation"
            result = PermissionResult(
                request_id=request.request_id,
                action=request.action,
                granted=True,
                reason=reason,
                requires_confirmation=True,
                audit_entry={
                    **audit_entry,
                    "decision": "granted",
                    "reason": reason,
                    "confirmed_by_user": True,
                },
            )
            self._audit_log.append(result.audit_entry or {})
            return result

        reason = f"Action '{request.action}' permitted"
        result = PermissionResult(
            request_id=request.request_id,
            action=request.action,
            granted=True,
            reason=reason,
            requires_confirmation=False,
            audit_entry={**audit_entry, "decision": "granted", "reason": reason},
        )
        self._audit_log.append(result.audit_entry or {})
        return result

    def confirm_action(self, request_id: str) -> bool:
        for entry in self._audit_log:
            if (
                entry.get("request_id") == request_id
                and entry.get("decision") == "pending_confirmation"
            ):
                entry["decision"] = "confirmed_by_user"
                entry["confirmed_by_user"] = True
                entry["confirmed_at"] = datetime.now(timezone.utc).isoformat()
                return True
        return False

    def get_audit_log(self) -> list[dict[str, Any]]:
        return list(self._audit_log)

    def clear_audit_log(self) -> None:
        self._audit_log.clear()

    def get_audit_summary(self) -> dict[str, Any]:
        total = len(self._audit_log)
        granted = sum(1 for e in self._audit_log if e.get("decision") == "granted")
        denied = sum(1 for e in self._audit_log if e.get("decision") == "denied")
        pending = sum(
            1
            for e in self._audit_log
            if e.get("decision") in ("pending_confirmation",)
        )
        return {
            "total_requests": total,
            "granted": granted,
            "denied": denied,
            "pending_confirmation": pending,
        }

