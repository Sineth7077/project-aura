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
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score in [0, 1]")
    context: Optional[dict[str, Any]] = Field(default=None)


class AgentDecideResponse(BaseModel):
    """Response for POST /agent/decide with complete pipeline status."""

    decision_id: str
    selected_action: str
    confidence: float
    explanation: str
    permission_status: str
    permission_reason: Optional[str] = None
    execution_status: str
    verification_status: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    execution_result: Optional[dict[str, Any]] = None
    verification_record: Optional[dict[str, Any]] = None
