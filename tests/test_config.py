"""Tests for core configuration module."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.core.config import (
    AuraConfig,
    ColabConfig,
    _detect_colab,
    detect_gpu,
    detect_tpu,
    load_config,
)
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
    assert config.tpu_enabled is False
    assert config.mount_drive is False
    assert config.gdrive_mount_path == "/content/drive"


# --- Auto-detection tests ---


def test_detect_colab_returns_false_locally():
    """_detect_colab should return False when not running in Colab."""
    # Ensure google.colab is not in sys.modules
    sys.modules.pop("google.colab", None)
    # Ensure COLAB_GPU is not set
    old_val = os.environ.pop("COLAB_GPU", None)
    try:
        assert _detect_colab() is False
    finally:
        if old_val is not None:
            os.environ["COLAB_GPU"] = old_val


def test_detect_colab_returns_true_when_module_present(monkeypatch):
    """_detect_colab should return True when google.colab is in sys.modules."""
    monkeypatch.setitem(sys.modules, "google.colab", object())
    assert _detect_colab() is True


def test_detect_colab_returns_true_when_env_set(monkeypatch):
    """_detect_colab should return True when COLAB_GPU=true."""
    monkeypatch.setenv("COLAB_GPU", "true")
    sys.modules.pop("google.colab", None)
    assert _detect_colab() is True


def test_detect_colab_returns_true_when_colab_release_tag_set(monkeypatch):
    """_detect_colab should return True when COLAB_RELEASE_TAG is set."""
    monkeypatch.setenv("COLAB_RELEASE_TAG", "2024.1")
    sys.modules.pop("google.colab", None)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    assert _detect_colab() is True


def test_detect_gpu_returns_false_when_no_frameworks(monkeypatch):
    """detect_gpu should return False when neither torch nor tf has GPU."""
    # Force both detection functions to return False
    monkeypatch.setattr("src.core.config._detect_torch_gpu", lambda: False)
    monkeypatch.setattr("src.core.config._detect_tensorflow_gpu", lambda: False)
    assert detect_gpu() is False


def test_detect_gpu_returns_true_when_torch_gpu(monkeypatch):
    """detect_gpu should return True when PyTorch reports GPU."""
    monkeypatch.setattr("src.core.config._detect_torch_gpu", lambda: True)
    monkeypatch.setattr("src.core.config._detect_tensorflow_gpu", lambda: False)
    assert detect_gpu() is True


def test_detect_gpu_returns_true_when_tf_gpu(monkeypatch):
    """detect_gpu should return True when TensorFlow reports GPU."""
    monkeypatch.setattr("src.core.config._detect_torch_gpu", lambda: False)
    monkeypatch.setattr("src.core.config._detect_tensorflow_gpu", lambda: True)
    assert detect_gpu() is True


def test_detect_tpu_returns_false_locally(monkeypatch):
    """detect_tpu should return False when no TPU is available."""
    monkeypatch.delenv("COLAB_TPU_ADDR", raising=False)
    assert detect_tpu() is False


def test_colab_config_gpu_env_override(monkeypatch):
    """ENABLE_GPU env var should override auto-detection."""
    monkeypatch.setenv("ENABLE_GPU", "true")
    # Force auto-detection to return False
    monkeypatch.setattr("src.core.config.detect_gpu", lambda: False)
    config = ColabConfig()
    assert config.gpu_enabled is True


def test_colab_config_gpu_env_override_false(monkeypatch):
    """ENABLE_GPU=false should override auto-detection to False."""
    monkeypatch.setenv("ENABLE_GPU", "false")
    # Force auto-detection to return True
    monkeypatch.setattr("src.core.config.detect_gpu", lambda: True)
    config = ColabConfig()
    assert config.gpu_enabled is False


def test_colab_config_runtime_type_colab(monkeypatch):
    """When running in Colab, runtime_type should be 'colab'."""
    monkeypatch.setitem(sys.modules, "google.colab", object())
    monkeypatch.delenv("COLAB_RUNTIME", raising=False)
    config = ColabConfig()
    assert config.runtime_type == "colab"


def test_colab_config_runtime_type_local(monkeypatch):
    """When not in Colab and COLAB_RUNTIME not set, runtime_type should be 'local'."""
    sys.modules.pop("google.colab", None)
    monkeypatch.delenv("COLAB_RUNTIME", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    config = ColabConfig()
    assert config.runtime_type == "local"


def test_load_config_returns_aura_config():
    """load_config should return an AuraConfig instance."""
    config = load_config()
    assert isinstance(config, AuraConfig)
    assert isinstance(config.colab, ColabConfig)