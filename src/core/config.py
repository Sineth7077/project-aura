"""Central configuration management for Project Aura 2.0.

Loads environment variables, defines configuration dataclasses, and
provides typed access to all runtime settings used by notebooks and
production code alike. Designed to work both in Google Colab (where
env vars are injected at runtime) and in local/production environments.

Auto-detection of Colab runtime and GPU availability is performed
lazily to avoid importing heavy frameworks when they are not installed.
Environment variables can override auto-detected values but are not
required for detection to work.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional


def _detect_colab() -> bool:
    """Detect if running inside Google Colab.

    Checks for the google.colab module and the COLAB_GPU / COLAB_RELEASE_TAG
    environment variables without importing heavy frameworks at module load time.
    """
    if "google.colab" in sys.modules:  # type: ignore[name-defined]
        return True
    colab_indicators = (
        os.getenv("COLAB_GPU", "").lower() in ("true", "1"),
        os.getenv("COLAB_RELEASE_TAG", "") != "",
    )
    return any(colab_indicators)


def _detect_torch_gpu() -> bool:
    """Detect GPU availability via PyTorch (lazy import).

    Returns False if PyTorch is not installed or no CUDA GPU is found.
    """
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _detect_tensorflow_gpu() -> bool:
    """Detect GPU availability via TensorFlow (lazy import).

    Returns False if TensorFlow is not installed or no GPU is found.
    """
    try:
        import tensorflow as tf  # type: ignore

        return len(tf.config.list_physical_devices("GPU")) > 0  # type: ignore[attr-defined]
    except Exception:
        return False


def detect_gpu() -> bool:
    """Detect GPU availability using PyTorch and/or TensorFlow.

    Tries PyTorch first, then TensorFlow. Returns True if either
    framework reports a GPU. Safe to call even when neither framework
    is installed.
    """
    if _detect_torch_gpu():
        return True
    return _detect_tensorflow_gpu()


def detect_tpu() -> bool:
    """Detect TPU availability.

    Checks for COLAB_TPU_ADDR environment variable and attempts a
    lazy TensorFlow TPU detection. Returns False if unavailable.
    """
    if os.getenv("COLAB_TPU_ADDR", ""):
        return True
    try:
        import tensorflow as tf  # type: ignore

        try:
            tf.distribute.cluster_resolver.TPUClusterResolver()  # type: ignore[attr-defined]
            return True
        except Exception:
            return False
    except Exception:
        return False



@dataclass
class ColabConfig:
    """Google Colab-specific configuration.

    Runtime type, GPU, and TPU are auto-detected when not explicitly
    set via environment variables.
    """

    runtime_type: str = field(
        default_factory=lambda: "colab" if _detect_colab() else os.getenv("COLAB_RUNTIME", "local")
    )
    gpu_enabled: bool = field(
        default_factory=lambda: (
            os.getenv("ENABLE_GPU", "").lower() in ("true", "1")
            if os.getenv("ENABLE_GPU") is not None
            else detect_gpu()
        )
    )
    tpu_enabled: bool = field(
        default_factory=lambda: (
            os.getenv("ENABLE_TPU", "").lower() in ("true", "1")
            if os.getenv("ENABLE_TPU") is not None
            else detect_tpu()
        )
    )
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