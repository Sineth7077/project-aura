"""Allowlisted action registry for the autonomous agent.

Defines the set of software actions the agent is permitted to execute.
Only actions explicitly registered here can be requested by the decision
engine and executed by the action executor. This is the core safety
mechanism: no action runs merely because a gesture or an LLM requested it.

Registered actions are purely software-level (pause/resume an internal
simulation, change dashboard state, display an alert, save a screenshot,
record an event, update visualization state). No physical hardware, shell
commands, arbitrary Python execution, or unrestricted external API calls are
supported at this stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Canonical action names (aligned with decision_engine.ACTION_*)
ACTION_PAUSE = "pause"
ACTION_RESUME = "resume"
ACTION_DISPLAY_ALERT = "display_alert"
ACTION_SAVE_SCREENSHOT = "save_screenshot"
ACTION_RECORD_EVENT = "record_event"
ACTION_UPDATE_VISUALIZATION = "update_visualization"
ACTION_NOOP = "noop"
ACTION_REQUEST_VERIFICATION = "request_verification"

ALL_ACTIONS = (
    ACTION_PAUSE,
    ACTION_RESUME,
    ACTION_DISPLAY_ALERT,
    ACTION_SAVE_SCREENSHOT,
    ACTION_RECORD_EVENT,
    ACTION_UPDATE_VISUALIZATION,
    ACTION_NOOP,
    ACTION_REQUEST_VERIFICATION,
)

# Actions that require explicit user confirmation before execution.
SENSITIVE_ACTIONS = frozenset({ACTION_SAVE_SCREENSHOT, ACTION_REQUEST_VERIFICATION})


@dataclass
class ActionEntry:
    """Metadata for a single registered action.

    Attributes:
        name: Canonical action name.
        handler: Optional callable that executes the action. When None,
            the action is registered but not executable (placeholder).
        description: Human-readable description.
        sensitive: Whether the action requires user confirmation.
        dry_run_supported: Whether the handler supports dry-run mode.
    """

    name: str
    handler: Optional[Callable[..., Any]] = None
    description: str = ""
    sensitive: bool = False
    dry_run_supported: bool = True

    def is_executable(self) -> bool:
        return self.handler is not None


class ActionRegistry:
    """Registry of allowlisted actions.

    Provides registration, lookup, and validation of actions. Actions are
    identified by canonical string names. Unknown actions are rejected.
    """

    def __init__(self) -> None:
        """Initialize the registry with the default allowlist."""
        self._actions: dict[str, ActionEntry] = {}
        self._register_default_actions()

    # ------------------------------------------------------------------ #
    # Default action registration
    # ------------------------------------------------------------------ #
    def _register_default_actions(self) -> None:
        """Register the built-in safe software actions.

        Non-sensitive actions get safe in-memory handlers that operate on
        a simple state dict stored in the registry. Sensitive actions
        (save_screenshot, request_verification) remain without handlers and
        require explicit user confirmation via the permission manager.

        Safety: No handler performs shell commands, file I/O (except the
        sensitive save_screenshot which requires confirmation), network
        calls, or physical hardware control. All actions are software-level.
        """

        # Internal state for pause/resume and visualization
        self._state: dict[str, Any] = {
            "paused": False,
            "visualization": {},
            "alerts": [],
            "events": [],
        }

        def _pause_handler(data: dict) -> dict:
            self._state["paused"] = True
            return {"paused": True, "state": "paused"}

        def _resume_handler(data: dict) -> dict:
            self._state["paused"] = False
            return {"resumed": True, "state": "running"}

        def _alert_handler(data: dict) -> dict:
            message = data.get("message", "Alert")
            self._state["alerts"].append({"message": message, "acknowledged": False})
            return {"alert_shown": True, "message": message}

        def _record_event_handler(data: dict) -> dict:
            self._state["events"].append(data)
            return {"recorded": True, "event_count": len(self._state["events"])}

        def _visualization_handler(data: dict) -> dict:
            self._state["visualization"].update(data)
            return {"visualization_updated": True, "current": self._state["visualization"]}

        def _noop_handler(data: dict) -> dict:
            return {"noop": True}

        defaults = [
            ActionEntry(
                name=ACTION_PAUSE,
                handler=_pause_handler,
                description="Pause an internal simulation or processing loop.",
                sensitive=False,
            ),
            ActionEntry(
                name=ACTION_RESUME,
                handler=_resume_handler,
                description="Resume a paused simulation or processing loop.",
                sensitive=False,
            ),
            ActionEntry(
                name=ACTION_DISPLAY_ALERT,
                handler=_alert_handler,
                description="Display a non-blocking alert to the user.",
                sensitive=False,
            ),
            ActionEntry(
                name=ACTION_SAVE_SCREENSHOT,
                handler=None,  # Sensitive: requires user confirmation + explicit handler
                description="Save an authorized screenshot to disk.",
                sensitive=True,
            ),
            ActionEntry(
                name=ACTION_RECORD_EVENT,
                handler=_record_event_handler,
                description="Record a software event to the audit log.",
                sensitive=False,
            ),
            ActionEntry(
                name=ACTION_UPDATE_VISUALIZATION,
                handler=_visualization_handler,
                description="Update a visualization or dashboard state.",
                sensitive=False,
            ),
            ActionEntry(
                name=ACTION_NOOP,
                handler=_noop_handler,
                description="No operation; placeholder for no action.",
                sensitive=False,
                dry_run_supported=True,
            ),
            ActionEntry(
                name=ACTION_REQUEST_VERIFICATION,
                handler=None,  # Sensitive: requires user confirmation
                description="Request manual verification before proceeding.",
                sensitive=True,
            ),
        ]
        for entry in defaults:
            self._actions[entry.name] = entry

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def register(
        self,
        name: str,
        handler: Optional[Callable[..., Any]] = None,
        description: str = "",
        sensitive: bool = False,
        dry_run_supported: bool = True,
    ) -> ActionEntry:
        """Register a new action or update an existing one.

        Args:
            name: Canonical action name.
            handler: Callable that executes the action.
            description: Human-readable description.
            sensitive: Whether user confirmation is required.
            dry_run_supported: Whether dry-run mode is supported.

        Returns:
            The registered :class:`ActionEntry`.
        """
        entry = ActionEntry(
            name=name,
            handler=handler,
            description=description,
            sensitive=sensitive,
            dry_run_supported=dry_run_supported,
        )
        self._actions[name] = entry
        return entry

    def get(self, name: str) -> Optional[ActionEntry]:
        """Return the :class:`ActionEntry` for a name, or None if unknown."""
        return self._actions.get(name)

    def is_allowlisted(self, name: str) -> bool:
        """Return True if the action is registered."""
        return name in self._actions

    def is_executable(self, name: str) -> bool:
        """Return True if the action is registered and has a handler."""
        entry = self._actions.get(name)
        return entry is not None and entry.is_executable()

    def is_sensitive(self, name: str) -> bool:
        """Return True if the action requires user confirmation."""
        entry = self._actions.get(name)
        return entry.sensitive if entry else False

    def supports_dry_run(self, name: str) -> bool:
        """Return True if the action supports dry-run execution."""
        entry = self._actions.get(name)
        return entry.dry_run_supported if entry else False

    def list_actions(self) -> list[ActionEntry]:
        """Return all registered actions."""
        return list(self._actions.values())

    def validate_action(self, name: str) -> list[str]:
        """Validate an action name and return a list of errors (empty = OK)."""
        errors: list[str] = []
        if not isinstance(name, str) or not name:
            errors.append("action name must be a non-empty string")
            return errors
        if name not in self._actions:
            errors.append(f"action '{name}' is not allowlisted")
        entry = self._actions.get(name)
        if entry and not entry.is_executable():
            errors.append(f"action '{name}' is registered but has no handler")
        return errors
