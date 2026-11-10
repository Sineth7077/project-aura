"""Central configuration management for Project Aura 2.0.

Loads environment variables, defines configuration dataclasses, and
provides typed access to all runtime settings used by notebooks and
production code alike. Designed to work both in Google Colab (where
env vars are injected at runtime) and in local/production environments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ColabConfig:
    """Google Colab-specific configuration."""
    runtime_type: str = field(default_factory=lambda: os.getenv("COLAB_RUNTIME", "local"))
    gpu_enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_GPU", "false").lower() == "true")
    tpu_enabled: bool = field(default_factory=lambda: os.getenv("ENABLE_TPU", "false").lower() == "true")
    mount_drive: bool = field(default_factory=lambda: os.getenv("MOUNT_GDRIVE", "false").lower() == "true")
    gdrive_mount_path: str = "/content/drive"


@dataclass
class VisionConfig:
    """Computer vision model configuration."""
    model_name: str = "efficientnet_b0"
    input_size: tuple[int, int] = (224, 224)
    num_classes: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 10
    pretrained: bool = True
    checkpoint_dir: str = "checkpoints"


@dataclass
class GestureConfig:
    """Gesture recognition configuration."""
    model_name: str = "mediapipe"
    confidence_threshold: float = 0.7
    max_num_hands: int = 2
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class AgentConfig:
    """Autonomous AI agent configuration."""
    agent_name: str = "aura-agent"
    max_iterations: int = 10
    temperature: float = 0.7
    max_tokens: int = 2048
    enable_tools: bool = True


@dataclass
class CloudConfig:
    """Cloud infrastructure configuration."""
    provider: str = field(default_factory=lambda: os.getenv("CLOUD_PROVIDER", "gcp"))
    project_id: Optional[str] = field(default_factory=lambda: os.getenv("GCP_PROJECT_ID"))
    region: str = field(default_factory=lambda: os.getenv("GCP_REGION", "us-central1"))
    bucket_name: Optional[str] = field(default_factory=lambda: os.getenv("GCS_BUCKET_NAME"))
    service_account_path: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))


@dataclass
class AuraConfig:
    """Master configuration aggregating all sub-configurations."""
    colab: ColabConfig = field(default_factory=ColabConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @classmethod
    def from_env(cls) -> "AuraConfig":
        """Create configuration from environment variables."""
        return cls()


def load_config() -> AuraConfig:
    """Load and return the application configuration.

    Returns:
        AuraConfig: Fully populated configuration instance.
    """
    return AuraConfig.from_env()