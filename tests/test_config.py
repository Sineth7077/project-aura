"""Tests for core configuration module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.core.config import AuraConfig, ColabConfig, load_config
from src.core.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_INPUT_SIZE,
    DEFAULT_LEARNING_RATE,
    PROJECT_NAME,
    PROJECT_VERSION,
    SUPPORTED_CLOUD_PROVIDERS,
    SUPPORTED_VISION_MODELS,
)


def test_project_constants():
    """Verify project constants are defined correctly."""
    assert PROJECT_NAME == "project-aura"
    assert PROJECT_VERSION == "2.0.0"
    assert DEFAULT_BATCH_SIZE == 32
    assert DEFAULT_EPOCHS == 10
    assert DEFAULT_LEARNING_RATE == 1e-4
    assert DEFAULT_INPUT_SIZE == (224, 224)
    assert "gcp" in SUPPORTED_CLOUD_PROVIDERS
    assert "efficientnet_b0" in SUPPORTED_VISION_MODELS


def test_aura_config_defaults():
    """Verify AuraConfig loads with sensible defaults."""
    config = load_config()
    assert isinstance(config, AuraConfig)
    assert isinstance(config.colab, ColabConfig)
    assert config.vision.batch_size == 32
    assert config.vision.epochs == 10
    assert config.vision.learning_rate == 1e-4
    assert config.debug is False
    assert config.log_level == "INFO"


def test_config_from_env(monkeypatch):
    """Verify configuration reads from environment variables."""
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ENABLE_GPU", "true")

    config = load_config()
    assert config.debug is True
    assert config.log_level == "DEBUG"
    assert config.colab.gpu_enabled is True


def test_colab_config_defaults():
    """Verify ColabConfig defaults."""
    config = ColabConfig()
    assert config.runtime_type == "local"
    assert config.gpu_enabled is False
    assert config.tpu_enabled is False
    assert config.mount_drive is False
    assert config.gdrive_mount_path == "/content/drive"