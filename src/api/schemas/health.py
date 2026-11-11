"""Pydantic schemas for health and readiness endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "healthy"
    service: str = "project-aura"
    version: str = "2.0.0"
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ReadinessResponse(BaseModel):
    """Response for GET /ready."""

    ready: bool
    components: dict[str, bool] = {}
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ModelInfoResponse(BaseModel):
    """Response for GET /model/info."""

    model_name: str
    num_classes: int
    input_size: list[int]
    loaded: bool
    device: str = "cpu"
