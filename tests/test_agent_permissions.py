"""Tests for the agent permission manager module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from src.agents.action_registry import ActionRegistry
from src.agents.permission_manager import PermissionManager, PermissionRequest


def test_permission_manager_creation():
    """Test PermissionManager instantiation."""
    registry = ActionRegistry()
    pm = PermissionManager(registry)
    assert pm.registry is registry


def test_permission_granted_for_allowlisted_action():
    """Test that allowlisted non-sensitive actions are granted."""
    registry = ActionRegistry()
    pm = PermissionManager(registry)
    request = PermissionRequest(action="pause", requesting_component="test")
    result = pm.check_permission(request)
    assert result.granted is True
    assert result.requires_confirmation is False


def test_permission_denied_for_unknown_action():
    """Test that unknown actions are denied."""
    registry = ActionRegistry()
    pm = PermissionManager(registry)
    request = PermissionRequest(action="unknown_action", requesting_component="test")
    result = pm.check_permission(request)
    assert result.granted is False
    assert "not allowlisted" in result.reason


def test_permission_sensitive_requires_confirmation():
    """Test that sensitive actions require user confirmation."""
    registry = ActionRegistry()
    pm = PermissionManager(registry)
    request = PermissionRequest(
        action="save_screenshot", requesting_component="test", confirmed_by_user=False
    )
    result = pm.check_permission(request)
    assert result.granted is False
    assert result.requires_confirmation is True


def test_permission_sensitive_granted_with_confirmation():
    """Test that sensitive actions are granted with user confirmation."""
    registry = ActionRegistry()
    pm = PermissionManager(registry)
    request = PermissionRequest(
        action="save_screenshot",
        requesting_component="test",
        confirmed_by_user=True,
    )
    result = pm.check_permission(request)
    assert result.granted is True


def test_permission_audit_log():
    """Test that permission decisions are recorded in audit log."""
    registry = ActionRegistry()
    pm = PermissionManager(registry)
    request = PermissionRequest(action="pause", requesting_component="test")
    pm.check_permission(request)
    log = pm.get_audit_log()
    assert len(log) == 1
    assert log[0]["action"] == "pause"


def test_permission_audit_summary():
    """Test audit summary statistics."""
    registry = ActionRegistry()
    pm = PermissionManager(registry)
    pm.check_permission(PermissionRequest(action="pause", requesting_component="test"))
    pm.check_permission(
        PermissionRequest(action="unknown_action", requesting_component="test")
    )
    summary = pm.get_audit_summary()
    assert summary["total_requests"] == 2
    assert summary["granted"] == 1
    assert summary["denied"] == 1


def test_confirm_action():
    """Test confirming a pending sensitive action."""
    registry = ActionRegistry()
    pm = PermissionManager(registry)
    request = PermissionRequest(
        action="save_screenshot", requesting_component="test", confirmed_by_user=False
    )
    result = pm.check_permission(request)
    assert result.granted is False
    assert result.requires_confirmation is True

    confirmed = pm.confirm_action(result.request_id)
    assert confirmed is True

    # Re-check should now be granted
    request2 = PermissionRequest(
        action="save_screenshot",
        requesting_component="test",
        confirmed_by_user=True,
    )
    result2 = pm.check_permission(request2)
    assert result2.granted is True
