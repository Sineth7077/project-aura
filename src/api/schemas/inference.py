"""Pydantic schemas for inference endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Request for POST /inference."""

    image_base64: str = Field(..., description="Base64-encoded image data")
    format: str = Field(default="jpeg", description="Image format (jpeg, png)")
    model_name: Optional[str] = Field(default=None, description="Model to use")


class InferenceResult(BaseModel):
    """Single inference result entry."""

    predicted_class: int
    confidence: float
    class_name: Optional[str] = None
    probabilities: Optional[list[float]] = None


class InferenceResponse(BaseModel):
    """Response for POST /inference."""

    results: list[InferenceResult]
    model_name: str
    processing_time_ms: float
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
