"""Health and readiness routes."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas.health import (
    HealthResponse,
    ReadinessResponse,
    ModelInfoResponse,
)
from src.api.services.model_service import ModelService

router = APIRouter(tags=["health"])

# Set by main.py during app setup; defaults to unloaded.
model_service: ModelService = ModelService()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Basic health check."""
    return HealthResponse()


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """Check whether required components are initialized."""
    components = {
        "model_loaded": model_service.is_loaded(),
    }
    return ReadinessResponse(
        ready=model_service.is_loaded(),
        components=components,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return available model information."""
    info = model_service.get_model_info()
    return ModelInfoResponse(
        model_name=info["model_name"],
        num_classes=info["num_classes"],
        input_size=[224, 224],
        loaded=info["loaded"],
        device=info["device"],
    )
