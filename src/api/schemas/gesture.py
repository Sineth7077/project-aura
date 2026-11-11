"""Pydantic schemas for gesture recognition endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class GestureRequest(BaseModel):
    """Request for POST /gesture."""

    image_base64: str = Field(..., description="Base64-encoded image data")
    format: str = Field(default="jpeg", description="Image format (jpeg, png)")


class GestureResult(BaseModel):
    """Single gesture recognition result."""

    gesture: str
    confidence: float
    handedness: str = "Unknown"
    bounding_box: Optional[list[int]] = None


class GestureResponse(BaseModel):
    """Response for POST /gesture."""

    results: list[GestureResult]
    num_hands: int
    available: bool
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
