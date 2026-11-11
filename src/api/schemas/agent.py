"""Pydantic schemas for agent decision endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentDecideRequest(BaseModel):
    """Request for POST /agent/decide."""

    observation_source: str = Field(..., description="Source component")
    observation_type: str = Field(..., description="Observation type")
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = Field(default=None)
    context: Optional[dict[str, Any]] = Field(default=None)


class AgentDecideResponse(BaseModel):
    """Response for POST /agent/decide."""

    decision_id: str
    selected_action: str
    confidence: float
    explanation: str
    permission_status: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
