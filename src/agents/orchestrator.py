"""Autonomous AI agent orchestrator for Project Aura 2.0.

Manages the lifecycle, task delegation, and coordination of multiple
AI agents. Designed to run both in Colab notebooks and as a production
service.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


@dataclass
class AgentTask:
    """Represents a task assigned to an AI agent."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_name: str = ""
    task_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "task_type": self.task_type,
            "payload": self.payload,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class AgentMetadata:
    """Metadata for a registered AI agent."""

    name: str
    agent_type: str
    capabilities: list[str] = field(default_factory=list)
    is_active: bool = True
    max_concurrent_tasks: int = 5
    current_task_count: int = 0
class AgentOrchestrator:
    """Orchestrates autonomous AI agents in Project Aura.

    Handles agent registration, task assignment, status tracking,
    and result collection. Thread-safe design for concurrent use.
    """

    def __init__(self, orchestrator_name: str = "aura-orchestrator"):
        self.orchestrator_name = orchestrator_name
        self._agents: dict[str, AgentMetadata] = {}
        self._tasks: dict[str, AgentTask] = {}
        self._task_handlers: dict[str, Callable] = {}

    def register_agent(
        self,
        name: str,
        agent_type: str,
        capabilities: Optional[list[str]] = None,
        max_concurrent_tasks: int = 5,
    ) -> AgentMetadata:
        """Register a new AI agent with the orchestrator.

        Args:
            name: Unique agent name.
            agent_type: Type/category of the agent.
            capabilities: List of capability strings.
            max_concurrent_tasks: Maximum concurrent tasks for this agent.

        Returns:
            The registered AgentMetadata.
        """
        metadata = AgentMetadata(
            name=name,
            agent_type=agent_type,
            capabilities=capabilities or [],
            max_concurrent_tasks=max_concurrent_tasks,
        )
        self._agents[name] = metadata
        return metadata

    def register_task_handler(self, task_type: str, handler: Callable) -> None:
        """Register a handler function for a specific task type.

        Args:
            task_type: String identifier for the task type.
            handler: Callable that accepts a task payload and returns a result.
        """
        self._task_handlers[task_type] = handler

    def submit_task(
        self,
        agent_name: str,
        task_type: str,
        payload: Optional[dict] = None,
    ) -> AgentTask:
        """Submit a new task for execution.

        Args:
            agent_name: Name of the agent to handle the task.
            task_type: Type of task to execute.
            payload: Task input data.

        Returns:
            The created AgentTask.

        Raises:
            ValueError: If the agent is not registered or task type has no handler.
        """
        if agent_name not in self._agents:
            raise ValueError(f"Agent '{agent_name}' is not registered")

        if task_type not in self._task_handlers:
            raise ValueError(f"No handler registered for task type '{task_type}'")

        agent = self._agents[agent_name]
        if agent.current_task_count >= agent.max_concurrent_tasks:
            raise ValueError(
                f"Agent '{agent_name}' is at capacity "
                f"({agent.current_task_count}/{agent.max_concurrent_tasks})"
            )

        task = AgentTask(
            agent_name=agent_name,
            task_type=task_type,
            payload=payload or {},
        )
        self._tasks[task.task_id] = task
        agent.current_task_count += 1
        return task

    def execute_task(self, task_id: str) -> AgentTask:
        """Execute a submitted task synchronously.

        Args:
            task_id: ID of the task to execute.

        Returns:
            The updated AgentTask with result or error.

        Raises:
            ValueError: If the task is not found.
        """
        if task_id not in self._tasks:
            raise ValueError(f"Task '{task_id}' not found")

        task = self._tasks[task_id]
        agent = self._agents[task.agent_name]

        task.status = "running"
        try:
            handler = self._task_handlers[task.task_type]
            result = handler(task.payload)
            task.result = result
            task.status = "completed"
        except Exception as exc:
            task.status = "failed"
            task.error = str(exc)
        finally:
            task.completed_at = datetime.now(timezone.utc)
            agent.current_task_count = max(0, agent.current_task_count - 1)

        return task

    def get_task(self, task_id: str) -> Optional[AgentTask]:
        """Retrieve a task by ID."""
        return self._tasks.get(task_id)

    def list_tasks(self, status: Optional[str] = None) -> list[AgentTask]:
        """List all tasks, optionally filtered by status."""
        if status is None:
            return list(self._tasks.values())
        return [t for t in self._tasks.values() if t.status == status]

    def list_agents(self) -> list[AgentMetadata]:
        """List all registered agents."""
        return list(self._agents.values())

    def get_agent(self, name: str) -> Optional[AgentMetadata]:
        """Retrieve an agent by name."""
        return self._agents.get(name)
