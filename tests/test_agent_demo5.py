"""Regression tests for the Demo 5 permission configuration fix.

Confirmed root cause: Demo 5 registered ``fail_action`` on the notebook's own
``action_registry``, but ``agent_service.decide()`` uses
``agent_service.registry`` (a different :class:`ActionRegistry` instance) for
both permission checking and execution. As a result:

* ``is_allowlisted("fail_action")`` was ``False`` on the engine's registry ->
  permission was denied before the handler could run.
* ``agent_service.executor`` runs in dry-run mode by default, which would
  simulate rather than genuinely execute the handler.

These tests lock in the correct behaviour:

1. An action registered on one registry is *not* allowlisted on another
   (registration does not leak across registry instances).
2. An explicitly allowlisted, non-sensitive ``fail_action`` receives
   permission from the permission manager.
3. The failing handler genuinely executes (dry run off) and produces a real
   execution failure from :class:`ActionExecutor`.
4. An unknown action remains denied.
5. Default decision rules still work after a temporary action is registered
   and cleaned up.

The permission manager is never bypassed, mocked, or weakened.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from src.agents.decision_engine import DecisionEngine, ACTION_PAUSE
from src.agents.context import ContextManager
from src.agents.observation import Observation, ObservationType
from src.agents.action_registry import ActionRegistry
from src.agents.permission_manager import PermissionManager, PermissionRequest
from src.agents.executor import ActionExecutor


def _make_gesture_observation(gesture, confidence=0.9):
    return Observation(
        source="regression_test",
        type=ObservationType.GESTURE,
        data={"gesture": gesture},
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# 1. Registered on one registry is not allowlisted on another (no leakage)
# ---------------------------------------------------------------------------
def test_registered_on_other_registry_is_not_allowlisted():
    """Action registered on a separate registry is not allowlisted here."""
    reg_a = ActionRegistry()
    reg_b = ActionRegistry()

    def handler(data):
        return data

    reg_a.register("some_action", handler=handler, sensitive=False)
    assert reg_a.is_allowlisted("some_action") is True
    assert reg_b.is_allowlisted("some_action") is False

    pm = PermissionManager(reg_b)
    result = pm.check_permission(PermissionRequest(
        action="some_action", requesting_component="test",
    ))
    assert result.granted is False
    assert "not allowlisted" in result.reason


# ---------------------------------------------------------------------------
# 2. Explicitly allowlisted non-sensitive fail_action receives permission
# ---------------------------------------------------------------------------
def test_allowlisted_fail_action_receives_permission():
    """Allowlisted non-sensitive fail_action is granted by the permission manager."""
    reg = ActionRegistry()

    def fail_handler(data):
        raise RuntimeError("Simulated action failure")

    reg.register("fail_action", handler=fail_handler,
                 description="Simulated failing action", sensitive=False)
    assert reg.is_allowlisted("fail_action") is True

    pm = PermissionManager(reg)
    result = pm.check_permission(PermissionRequest(
        action="fail_action", requesting_component="test",
    ))
    assert result.granted is True
    assert result.requires_confirmation is False


# ---------------------------------------------------------------------------
# 3. Failing handler genuinely executes and produces an execution failure
# ---------------------------------------------------------------------------
def test_fail_action_handler_executes_and_fails():
    """Dry-run off: the handler genuinely executes and ActionExecutor fails it."""
    reg = ActionRegistry()

    def fail_handler(data):
        raise RuntimeError("Simulated action failure")

    reg.register("fail_action", handler=fail_handler,
                 description="Simulated failing action", sensitive=False)

    pm = PermissionManager(reg)
    perm = pm.check_permission(PermissionRequest(
        action="fail_action", requesting_component="test",
    ))
    assert perm.granted is True

    executor = ActionExecutor(reg, dry_run=False)  # genuine execution
    result = executor.execute(action="fail_action", permission_result=perm, data={})

    assert result.succeeded is False
    assert result.error == "Simulated action failure"


# ---------------------------------------------------------------------------
# 4. Unknown action remains denied
# ---------------------------------------------------------------------------
def test_unknown_action_remains_denied():
    """An action never registered is denied by the permission manager."""
    reg = ActionRegistry()
    pm = PermissionManager(reg)
    result = pm.check_permission(PermissionRequest(
        action="totally_unknown_action", requesting_component="test",
    ))
    assert result.granted is False
    assert "not allowlisted" in result.reason


# ---------------------------------------------------------------------------
# 5. Default rules and denials still work after a temporary action/cleanup
# ---------------------------------------------------------------------------
def test_default_rules_and_denials_still_work_after_cleanup():
    """Temporary rule registration + restore leaves default behaviour intact."""
    engine = DecisionEngine(confidence_threshold=0.7)
    context = ContextManager()

    # Temporarily add a custom rule (mirrors Demo 5), then restore defaults.
    engine.clear_rules()
    engine.add_rule(lambda o, c: ("fail_action", 0.9, "temp"))
    try:
        obs = _make_gesture_observation("OPEN_PALM", 0.9)
        decision = engine.decide(obs, context.snapshot())
        assert decision.selected_action == "fail_action"
    finally:
        engine.clear_rules()
        engine._register_default_rules()

    # Default rules restored: reliable OPEN_PALM -> pause.
    obs = _make_gesture_observation("OPEN_PALM", 0.9)
    decision = engine.decide(obs, context.snapshot())
    assert decision.selected_action == ACTION_PAUSE

    # Unknown action still denied by the permission manager.
    reg = ActionRegistry()
    pm = PermissionManager(reg)
    result = pm.check_permission(PermissionRequest(
        action="unknown_action", requesting_component="test",
    ))
    assert result.granted is False
