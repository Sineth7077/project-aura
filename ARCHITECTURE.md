# Project Aura 2.0 - Technical Architecture

## Overview

Project Aura 2.0 is a fully cloud-based AI Vision Control Center that combines
computer vision, gesture recognition, and autonomous AI agents. The system is
designed for **cloud-first development using Google Colab** as the primary
development environment, with a clear path to production deployment on cloud
infrastructure (GCP, AWS, or Azure).

## Design Principles

1. **Cloud-First**: Primary development happens in Google Colab notebooks.
   Production code lives in `src/` and is importable by notebooks.
2. **Modular**: Clear separation between configuration, vision, agents, API,
   and utility layers. Each module is independently testable.
3. **No Lock-in**: Cloud provider is configurable via environment variables.
   Default target is GCP but abstraction allows AWS/Azure migration.
4. **Credential Security**: Zero secrets in version control. `.env.example`
   provides a template; `.env` is gitignored.
5. **Graceful Degradation**: All ML components work with and without heavy
   dependencies (PyTorch, MediaPipe, google-cloud-storage).
6. **Executable Notebooks**: All `.ipynb` files are valid, runnable Jupyter
   notebooks designed for Google Colab.
## Repository Structure

```
project-aura/
+-- .env.example              # Environment variable template (no secrets)
+-- .gitignore                # Comprehensive ignore rules
+-- ARCHITECTURE.md           # This document
+-- DEVELOPMENT_ROADMAP.md    # Phased implementation plan
+-- PROJECT_STATUS.md         # Current project status tracking
+-- README.md                 # Project overview and quickstart
+-- LICENSE                   # Project license
+-- notebooks/                # Google Colab notebooks (.ipynb)
¦   +-- 01_environment_setup.ipynb
¦   +-- 02_data_ingestion.ipynb
¦   +-- 03_model_training.ipynb
¦   +-- 04_inference_engine.ipynb
¦   +-- 05_gesture_recognition.ipynb
¦   +-- 06_ai_agents.ipynb
¦   +-- 07_deployment_prep.ipynb
¦   +-- 08_monitoring_analytics.ipynb
+-- src/                      # Production source code (importable by notebooks)
¦   +-- __init__.py
¦   +-- api/
¦   ¦   +-- __init__.py
¦   ¦   +-- routes.py            # REST API route definitions
¦   +-- core/
¦   ¦   +-- __init__.py
¦   ¦   +-- config.py            # Central configuration management
¦   ¦   +-- constants.py         # Project-wide constants
¦   ¦   +-- exceptions.py        # Custom exception hierarchy
¦   +-- utils/
¦   ¦   +-- __init__.py
¦   ¦   +-- gcs.py               # Google Cloud Storage client wrapper
¦   ¦   +-- logging.py           # Structured logging (JSON + text)
¦   +-- vision/
¦   ¦   +-- __init__.py
¦   ¦   +-- inference.py         # Inference engine (load/preprocess/predict)
¦   ¦   +-- models.py            # Vision model wrappers (EfficientNet + fallback)
¦   ¦   +-- preprocessing.py     # Image preprocessing pipeline
¦   +-- agents/
¦       +-- __init__.py
¦       +-- gesture.py           # Gesture recognition (MediaPipe + fallback)
¦       +-- orchestrator.py      # AI agent orchestration
+-- configs/                    # Configuration files (YAML, JSON)
+-- docs/                       # Additional documentation
+-- scripts/                    # Deployment and utility scripts
+-- tests/                      # Test suite (pytest-compatible)
    +-- __init__.py
    +-- test_config.py
    +-- test_exceptions.py
    +-- test_gesture.py
    +-- test_inference.py
    +-- test_orchestrator.py
    +-- test_preprocessing.py
```

## Module Architecture

### Core Layer (`src/core/`)
- **`config.py`**: Dataclass-based configuration (`AuraConfig`, `ColabConfig`,
  `VisionConfig`, `GestureConfig`, `AgentConfig`, `CloudConfig`).
  Loaded from environment variables via `load_config()`.
- **`constants.py`**: Project metadata, supported models, cloud providers,
  default hyperparameters.
- **`exceptions.py`**: Exception hierarchy (`AuraException` base +
  `ConfigurationError`, `ModelNotFoundError`, `InferenceError`,
  `DataIngestionError`, `CloudConnectionError`, `GestureRecognitionError`,
  `AgentOrchestrationError`).

### Vision Layer (`src/vision/`)
- **`preprocessing.py`**: Image resize (aspect-ratio preserving), normalization
  (ImageNet stats), batch preparation (NCHW), denormalization for display.
- **`models.py`**: `VisionModelBase` abstract base, `EfficientNetModel`
  (PyTorch/torchvision with `_LightweightVisionModel` fallback), `create_model()`
  factory.
- **`inference.py`**: `InferenceEngine` — model lifecycle, preprocessing,
  single/batch prediction, result dictionaries.

### Agents Layer (`src/agents/`)
- **`orchestrator.py`**: `AgentOrchestrator` — agent registration, task
  submission/execution, status tracking. `AgentTask` and `AgentMetadata`
  dataclasses.
- **`gesture.py`**: `GestureRecognizer` — MediaPipe Hands integration with
  lightweight motion-based fallback. `GestureResult` dataclass.

### API Layer (`src/api/`)
- **`routes.py`**: `Router` class with `get`/`post`/`put`/`delete` methods.
  Default routes: `/health`, `/vision/predict`, `/gesture/detect`,
  `/agents/status`.

### Utils Layer (`src/utils/`)
- **`logging.py`**: `get_logger()` / `setup_logging()` with optional JSON
  formatting for cloud observability (GCP Cloud Logging, etc.).
- **`gcs.py`**: `GCSClient` — lazy-initialized Google Cloud Storage wrapper
  for upload/download/list operations. Gracefully degrades when
  `google-cloud-storage` is not installed.

## Google Colab Integration

All eight notebooks in `notebooks/` are designed for Google Colab:

1. **Environment Setup**: GPU/TPU detection, dependency installation,
   `.env` loading from Colab secrets, project module import path setup.
2. **Data Ingestion**: Dataset loading from local files, GCS, or public URLs.
   Preprocessing via `src.vision.preprocessing`.
3. **Model Training**: Transfer learning with `src.vision.models`, training
   loop, checkpoint saving to local disk or GCS.
4. **Inference Engine**: Load trained models, run inference on images/video,
   visualize results via `src.vision.inference`.
5. **Gesture Recognition**: Real-time gesture detection using
   `src.agents.gesture` (MediaPipe or fallback).
6. **AI Agents**: Autonomous agent orchestration using
   `src.agents.orchestrator`. Register agents, submit tasks, track results.
7. **Deployment Prep**: Model export (ONNX, TorchScript), Dockerfile
   generation, cloud deployment configuration.
8. **Monitoring & Analytics**: Performance metrics, inference latency tracking,
   logging configuration via `src.utils.logging`.

Notebooks import from `src/` by adding the project root to `sys.path`:

```python
import sys
sys.path.append('/content/project-aura')  # Colab path after git clone
from src.core.config import load_config
from src.vision.inference import InferenceEngine
```

## Cloud Deployment Path

### Development (Google Colab)
- Free tier GPU/TPU for training
- No infrastructure costs
- Notebooks as primary interface
- `.env` via Colab Secrets

### Staging (GCP Cloud Run / AWS App Runner)
- Containerized inference API (`src/api/routes.py`)
- GCS for model artifacts and datasets
- Cloud Logging via JSON formatter (`src.utils.logging`)

### Production (GKE / EKS / AKS)
- Scalable deployment with autoscaling
- CI/CD pipeline (GitHub Actions)
- Monitoring dashboards (notebook 08)

## Security Considerations

- **No credentials in git**: `.env` is gitignored; `.env.example` is the
  template. Secrets injected via Colab Secrets or cloud Secret Manager.
- **Service accounts**: `GOOGLE_APPLICATION_CREDENTIALS` points to a key file
  that must never be committed.
- **Input validation**: All inference inputs validated for shape and type.
- **Error handling**: Custom exception hierarchy prevents information leakage.

## Future Extensibility

- Add new vision models by extending `VisionModelBase` and updating
  `create_model()` factory.
- Add new agent types by registering with `AgentOrchestrator.register_agent()`.
- Add new API routes via `Router` in `src/api/routes.py`.
- Add new cloud providers by extending `GCSClient` or creating new storage
  adapter classes.
