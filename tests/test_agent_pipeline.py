"""Integration tests for the complete agent pipeline.

Tests the full flow: Observation → Context → Decision → Permission → Execution → Verification
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.api.services.agent_service import AgentService
from src.agents.action_registry import ACTION_PAUSE, ACTION_RESUME, ACTION_SAVE_SCREENSHOT


def test_allowed_action_executed():
    """Test that an allowed action is executed successfully."""
    service = AgentService(dry_run=False)
    
    # OPEN_PALM gesture should trigger pause action (safe, allowed)
    result = service.decide({
        "observation_source": "test",
        "observation_type": "gesture",
        "data": {"gesture": "OPEN_PALM"},
        "confidence": 0.9,
    })
    
    assert result["selected_action"] == "pause"
    assert result["permission_status"] == "granted"
    assert result["execution_status"] == "succeeded"
    # Without independent verification, status is verification_unavailable
    assert result["verification_status"] == "verification_unavailable"
    assert result["execution_result"] is not None
    assert result["execution_result"]["succeeded"] is True


def test_denied_action_not_executed():
    """Test that a denied action is not executed."""
    service = AgentService(dry_run=False)
    
    # Try to execute an unknown action (not in registry)
    # We'll test this by modifying the decision engine temporarily
    service.decision_engine.clear_rules()
    
    def unknown_rule(obs, ctx):
        return ("unknown_action", 0.9, "Test unknown action")
    
    service.decision_engine.add_rule(unknown_rule)
    
    result = service.decide({
        "observation_source": "test",
        "observation_type": "custom",
        "data": {},
        "confidence": 0.9,
    })
    
    assert result["selected_action"] == "unknown_action"
    assert result["permission_status"] == "denied"
    assert result["execution_status"] == "skipped"
    # When execution is skipped (permission denied), verification is execution_failed
    assert result["verification_status"] == "execution_failed"
    assert result["execution_result"] is not None
    assert result["execution_result"]["succeeded"] is False
    # Permission is denied before allowlist check, so error is permission_denied
    assert result["execution_result"]["error"] == "permission_denied"


def test_sensitive_action_requires_confirmation():
    """Test that sensitive actions require user confirmation."""
    service = AgentService(dry_run=False)
    
    # Modify decision to trigger save_screenshot (sensitive action)
    service.decision_engine.clear_rules()
    
    def sensitive_rule(obs, ctx):
        return (ACTION_SAVE_SCREENSHOT, 0.9, "Test sensitive action")
    
    service.decision_engine.add_rule(sensitive_rule)
    
    result = service.decide({
        "observation_source": "test",
        "observation_type": "custom",
        "data": {},
        "confidence": 0.9,
    })
    
    assert result["selected_action"] == ACTION_SAVE_SCREENSHOT
    assert result["permission_status"] == "denied"
    assert "requires explicit user confirmation" in result["permission_reason"]
    assert result["execution_status"] == "skipped"


def test_failed_execution_reported():
    """Test that failed execution is reported accurately."""
    service = AgentService(dry_run=False)
    
    # Register an action that will fail
    def failing_handler(data):
        raise RuntimeError("Simulated failure")
    
    service.registry.register(
        name="failing_action",
        handler=failing_handler,
        description="Action that always fails",
        sensitive=False,
    )
    
    service.decision_engine.clear_rules()
    
    def failing_rule(obs, ctx):
        return ("failing_action", 0.9, "Test failing action")
    
    service.decision_engine.add_rule(failing_rule)
    
    result = service.decide({
        "observation_source": "test",
        "observation_type": "custom",
        "data": {},
        "confidence": 0.9,
    })
    
    assert result["selected_action"] == "failing_action"
    assert result["permission_status"] == "granted"
    assert result["execution_status"] == "failed"
    # When execution fails, verification status is execution_failed
    assert result["verification_status"] == "execution_failed"
    assert result["execution_result"] is not None
    assert result["execution_result"]["succeeded"] is False
    assert result["execution_result"]["error"] is not None


def test_dry_run_mode():
    """Test that dry-run mode simulates execution without side effects."""
    service = AgentService(dry_run=True)
    
    result = service.decide({
        "observation_source": "test",
        "observation_type": "gesture",
        "data": {"gesture": "OPEN_PALM"},
        "confidence": 0.9,
    })
    
    assert result["selected_action"] == "pause"
    assert result["permission_status"] == "granted"
    assert result["execution_status"] == "succeeded"
    assert result["execution_result"] is not None
    assert result["execution_result"]["dry_run"] is True
    assert "Dry-run" in result["execution_result"]["message"]


def test_low_confidence_gesture_noop():
    """Test that low-confidence gestures result in noop."""
    service = AgentService(dry_run=False)
    
    result = service.decide({
        "observation_source": "test",
        "observation_type": "gesture",
        "data": {"gesture": "OPEN_PALM"},
        "confidence": 0.5,  # Below threshold
    })
    
    assert result["selected_action"] == "noop"
    assert result["permission_status"] == "granted"  # noop is always allowed
    assert result["execution_status"] == "succeeded"


def test_pipeline_context_updates():
    """Test that context is updated with each observation."""
    service = AgentService(dry_run=False)
    
    # First observation
    result1 = service.decide({
        "observation_source": "test",
        "observation_type": "gesture",
        "data": {"gesture": "OPEN_PALM"},
        "confidence": 0.9,
    })
    
    # Context should have latest_gesture as OPEN_PALM
    assert service.context.snapshot().latest_gesture == "OPEN_PALM"
    
    # Second observation
    result2 = service.decide({
        "observation_source": "test",
        "observation_type": "gesture",
        "data": {"gesture": "THUMBS_UP"},
        "confidence": 0.9,
    })
    
    # Context should have updated
    assert result1["decision_id"] != result2["decision_id"]
    # After second gesture, latest should be THUMBS_UP, previous should be OPEN_PALM
    assert service.context.snapshot().latest_gesture == "THUMBS_UP"
    assert service.context.snapshot().previous_gesture == "OPEN_PALM"


def test_verification_failure_representation():
    """Test that verification failure is represented correctly."""
    service = AgentService(dry_run=False)
    
    # Register a custom verification check that fails
    def failing_check(execution_result, expected_state):
        return False  # Always fail verification
    
    service.verification.set_custom_check(failing_check)
    
    result = service.decide({
        "observation_source": "test",
        "observation_type": "gesture",
        "data": {"gesture": "OPEN_PALM"},
        "confidence": 0.9,
    })
    
    assert result["execution_status"] == "succeeded"
    # With custom check that fails, status is verification_failed
    assert result["verification_status"] == "verification_failed"
    assert result["verification_record"] is not None
    assert result["verification_record"]["verified"] is False


def test_rest_and_websocket_consistency():
    """Test that REST and WebSocket would use the same agent service."""
    service = AgentService(dry_run=False)
    
    # Simulate REST request
    rest_result = service.decide({
        "observation_source": "api",
        "observation_type": "gesture",
        "data": {"gesture": "OPEN_PALM"},
        "confidence": 0.9,
    })
    
    # Simulate WebSocket request
    ws_result = service.decide({
        "observation_source": "websocket",
        "observation_type": "gesture",
        "data": {"gesture": "OPEN_PALM"},
        "confidence": 0.9,
    })
    
    # Both should have the same pipeline behavior
    assert rest_result["selected_action"] == ws_result["selected_action"]
    assert rest_result["permission_status"] == ws_result["permission_status"]
    assert rest_result["execution_status"] == ws_result["execution_status"]
    assert rest_result["verification_status"] == ws_result["verification_status"]
