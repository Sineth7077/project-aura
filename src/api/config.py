"""API configuration for the Project Aura deployment.

Centralizes FastAPI application settings: host, port, CORS, model paths,
confidence thresholds, and request limits. Uses environment variables with
safe defaults. No credentials are stored here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _default_cors_origins() -> list[str]:
    """Default CORS origins from environment variable."""
    env_value = os.getenv("API_CORS_ORIGINS", "http://localhost:8000,http://localhost:3000")
    if not env_value:
        return []
    
    origins = [o.strip() for o in env_value.split(",") if o.strip()]
    
    # In production, reject wildcard
    if os.getenv("ENVIRONMENT", "development") == "production":
        origins = [o for o in origins if o != "*"]
        if not origins:
            return []
    
    return origins


@dataclass
class APIConfig:
    """FastAPI application configuration.

    Attributes:
        host: Host to bind the server.
        port: Port to bind the server.
        workers: Number of uvicorn workers.
        cors_origins: Allowed CORS origins.
        max_image_size_mb: Maximum allowed image upload size in MB.
        model_checkpoint: Path to the model checkpoint (optional).
        model_name: Name of the vision model.
        input_size: Model input dimensions (H, W).
        confidence_threshold: Gesture confidence threshold.
        debug: Enable debug mode.
        environment: "development" or "production".
    """

    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "1")))
    cors_origins: list[str] = field(default_factory=_default_cors_origins)
    max_image_size_mb: float = field(
        default_factory=lambda: float(
            os.getenv("API_MAX_IMAGE_SIZE_MB", "10")
        )
    )
    model_checkpoint: str = field(
        default_factory=lambda: os.getenv("MODEL_CHECKPOINT", "")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "VISION_MODEL_NAME", "efficientnet_b0"
        )
    )
    input_size: tuple[int, int] = field(
        default_factory=lambda: (
            int(os.getenv("VISION_INPUT_SIZE", "224")),
            int(os.getenv("VISION_INPUT_SIZE", "224")),
        )
    )
    confidence_threshold: float = field(
        default_factory=lambda: float(
            os.getenv("GESTURE_CONFIDENCE_THRESHOLD", "0.7")
        )
    )
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )

    @property
    def max_image_size_bytes(self) -> int:
        """Max image size in bytes."""
        return int(self.max_image_size_mb * 1024 * 1024)

    def is_production(self) -> bool:
        """Return True if running in production mode."""
        return self.environment == "production"


def load_api_config() -> APIConfig:
    """Load API configuration from environment variables."""
    return APIConfig()
