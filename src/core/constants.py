"""Project-wide constants for Project Aura 2.0."""

# Project metadata
PROJECT_NAME = "project-aura"
PROJECT_VERSION = "2.0.0"
PROJECT_DESCRIPTION = "A fully cloud-based AI Vision Control Center with computer vision, gesture recognition and autonomous AI agents."

# Directory names
NOTEBOOKS_DIR = "notebooks"
SRC_DIR = "src"
CONFIGS_DIR = "configs"
TESTS_DIR = "tests"
DOCS_DIR = "docs"
SCRIPTS_DIR = "scripts"

# File extensions
NOTEBOOK_EXT = ".ipynb"
PYTHON_EXT = ".py"
ENV_EXT = ".env"

# Default values
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 10
DEFAULT_INPUT_SIZE = (224, 224)
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

# Supported cloud providers
SUPPORTED_CLOUD_PROVIDERS = ("gcp", "aws", "azure")

# Supported model architectures
SUPPORTED_VISION_MODELS = ("efficientnet_b0", "resnet50", "mobilenet_v3", "vit_b_16")

# MIME types
MIME_TYPE_JSON = "application/json"
MIME_TYPE_IMAGE = "image/jpeg"
MIME_TYPE_NOTEBOOK = "application/x-ipynb+json"