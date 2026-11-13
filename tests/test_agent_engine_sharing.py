"""Regression tests for the DecisionEngine / AgentService instance sharing fix.

Covers the confirmed root cause in notebook 09: the notebook previously created
two different DecisionEngine instances (a global one and
``agent_service.decision_engine``). ``run_agent_demo()`` calls
``agent_service.decide(...)``, which uses ``agent_service.decision_engine``,
so rule changes made only on the global engine never affected the pipeline.

These tests verify that:
1. AgentService uses the exact DecisionEngine instance it is given.
2. Rule modifications on that shared engine drive ``agent_service.decide()``.
3. The permission manager genuinely rejects an unregistered action
   (no mocking, no weakened permission checks).
4. Default rules are restored on the same engine even if an assertion fails
   (try/finally).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from src.agents.decision_engine import DecisionEngine, ACTION_NOOP
from src.agents.context import ContextManager
from src.agents.observation import Observation, ObservationType
from src.api.services.agent_service import AgentService


def _make_gesture_observation(gesture, confidence=0.9):
    return Observation(
        source="regression_test",
        type=ObservationType.GESTURE,
        data={"gesture": gesture},
        confidence=confidence,
    )


def test_agent_service_uses_shared_decision_engine_instance():
    """AgentService.decision_engine must be the exact instance supplied."""
    engine = DecisionEngine(confidence_threshold=0.7)
    service = AgentService(confidence_threshold=0.7, dry_run=True)
    service.decision_engine = engine
    assert service.decision_engine is engine


def test_shared_engine_rule_affects_decide_and_unknown_action_denied():
    """A rule added to the shared engine drives decide(); unknown action denied.

    The permission manager genuinely denies 'dangerous_action' because it is
    not allowlisted; execution is therefore skipped.
    """
    engine = DecisionEngine(confidence_threshold=0.7)
    service = AgentService(confidence_threshold=0.7, dry_run=True)
    service.decision_engine = engine
    assert service.decision_engine is engine

    engine.clear_rules()

    def unknown_action_rule(obs, ctx):
        return ("dangerous_action", 0.9, "regression: unknown action request")

    # Custom rule first, fallback only after it.
    engine.add_rule(unknown_action_rule)
    engine.add_rule(lambda o, c: ("noop", 0.5, "fallback"))

    try:
        result = service.decide({
            "observation_source": "regression_test",
            "observation_type": "gesture",
            "data": {"gesture": "OPEN_PALM", "hand": "Right"},
            "confidence": 0.9,
        })

        # The custom rule on the shared engine wins.
        assert result["selected_action"] == "dangerous_action"
        # Permission manager genuinely rejects the unregistered action.
        assert result["permission_status"] == "denied"
        assert result["execution_status"] == "skipped"
    finally:
        # Restore default rules on the same engine even if an assertion fails.
        engine.clear_rules()
        engine._register_default_rules()


def test_other_demos_still_use_default_rules_after_restore():
    """After restoring defaults, the shared engine produces default decisions."""
    engine = DecisionEngine(confidence_threshold=0.7)
    service = AgentService(confidence_threshold=0.7, dry_run=True)
    service.decision_engine = engine

    # Simulate a demo that temporarily replaces rules, then restores them.
    engine.clear_rules()
    engine.add_rule(lambda o, c: ("dangerous_action", 0.9, "temporary"))
    try:
        result = service.decide({
            "observation_source": "regression_test",
            "observation_type": "gesture",
            "data": {"gesture": "OPEN_PALM", "hand": "Right"},
            "confidence": 0.9,
        })
        assert result["selected_action"] == "dangerous_action"
    finally:
        engine.clear_rules()
        engine._register_default_rules()

    # Default rules restored: OPEN_PALM at high confidence -> pause.
    context = ContextManager()
    obs = _make_gesture_observation("OPEN_PALM", 0.95)
    decision = engine.decide(obs, context.snapshot())
    assert decision.selected_action == "pause"
    assert decision.selected_action != ACTION_NOOP
