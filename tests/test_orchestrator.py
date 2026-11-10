"""Tests for agent orchestrator module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.agents.orchestrator import AgentOrchestrator, AgentTask


def test_orchestrator_creation():
    """Test orchestrator can be created."""
    orch = AgentOrchestrator("test-orc")
    assert orch.orchestrator_name == "test-orc"
    assert len(orch.list_agents()) == 0


def test_register_agent():
    """Test agent registration."""
    orch = AgentOrchestrator()
    meta = orch.register_agent("agent-1", "vision", capabilities=["detect", "track"])
    assert meta.name == "agent-1"
    assert meta.agent_type == "vision"
    assert "detect" in meta.capabilities
    assert len(orch.list_agents()) == 1


def test_submit_and_execute_task():
    """Test task submission and execution."""
    orch = AgentOrchestrator()
    orch.register_agent("agent-1", "vision")

    def handler(payload):
        return {"result": f"processed {payload.get('input', 'none')}"}

    orch.register_task_handler("inference", handler)
    task = orch.submit_task("agent-1", "inference", {"input": "test-image.jpg"})
    assert task.status == "pending"

    result = orch.execute_task(task.task_id)
    assert result.status == "completed"
    assert result.result == {"result": "processed test-image.jpg"}


def test_submit_task_unregistered_agent():
    """Test submitting task to unregistered agent raises."""
    orch = AgentOrchestrator()
    try:
        orch.submit_task("unknown", "inference")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_execute_unknown_task():
    """Test executing unknown task raises."""
    orch = AgentOrchestrator()
    try:
        orch.execute_task("non-existent-id")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_task_to_dict():
    """Test task serialization."""
    task = AgentTask(agent_name="a1", task_type="test", payload={"k": "v"})
    d = task.to_dict()
    assert d["agent_name"] == "a1"
    assert d["task_type"] == "test"
    assert d["payload"] == {"k": "v"}
    assert d["status"] == "pending"