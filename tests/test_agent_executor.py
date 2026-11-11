"""Tests for the agent action executor module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from src.agents.action_registry import ActionRegistry
from src.agents.permission_manager import PermissionManager, PermissionRequest
from src.agents.executor import ActionExecutor, ExecutionResult


def _make_handler(data):
    return {"executed": True, "data": data}


def test_executor_creation():
    """Test ActionExecutor instantiation."""
    registry = ActionRegistry()
    executor = ActionExecutor(registry)
    assert executor.registry is registry
    assert executor.dry_run is False


def test_executor_dry_run_mode():
    """Test dry-run mode executes without side effects."""
    registry = ActionRegistry()
    executor = ActionExecutor(registry, dry_run=True)
    pm = PermissionManager(registry)
    request = PermissionRequest(action="pause", requesting_component="test")
    perm_result = pm.check_permission(request)
    assert perm_result.granted is True

    result = executor.execute("pause", perm_result, data={"key": "val"})
    assert result.dry_run is True
    assert result.succeeded is True
    assert "Dry-run" in result.message


def test_executor_denied_permission():
    """Test that execution is rejected without permission."""
    registry = ActionRegistry()
    executor = ActionExecutor(registry)
    perm_result = type("PermResult", (), {"granted": False})()
    result = executor.execute("pause", perm_result)
    assert result.succeeded is False
    assert result.error == "permission_denied"


def test_executor_unknown_action():
    """Test that unknown actions are rejected."""
    registry = ActionRegistry()
    executor = ActionExecutor(registry)
    pm = PermissionManager(registry)
    request = PermissionRequest(action="unknown_action", requesting_component="test")
    perm_result = pm.check_permission(request)
    assert perm_result.granted is False
    result = executor.execute("unknown_action", perm_result)
    assert result.succeeded is False


def test_executor_execution_history():
    """Test that executions are recorded in history."""
    registry = ActionRegistry()
    registry.register("pause", handler=lambda d: {"paused": True})
    executor = ActionExecutor(registry)
    pm = PermissionManager(registry)
    request = PermissionRequest(action="pause", requesting_component="test")
    perm_result = pm.check_permission(request)

    executor.execute("pause", perm_result)
    history = executor.get_history()
    assert len(history) == 1
    assert history[0].action == "pause"


def test_executor_execution_summary():
    """Test execution summary statistics."""
    registry = ActionRegistry()
    registry.register("pause", handler=lambda d: {"paused": True})
    executor = ActionExecutor(registry)
    pm = PermissionManager(registry)

    # Successful execution
    request = PermissionRequest(action="pause", requesting_component="test")
    perm_result = pm.check_permission(request)
    executor.execute("pause", perm_result)

    summary = executor.get_execution_summary()
    assert summary["total_executions"] == 1
    assert summary["succeeded"] == 1


def test_executor_set_dry_run():
    """Test toggling dry-run mode."""
    registry = ActionRegistry()
    executor = ActionExecutor(registry, dry_run=False)
    assert executor.dry_run is False
    executor.set_dry_run(True)
    assert executor.dry_run is True
