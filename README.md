# Project Aura 2.0

A fully cloud-based AI Vision Control Center with computer vision, gesture
recognition, and autonomous AI agents. Built for Google Colab development with
a clear path to production deployment on cloud infrastructure.

## Overview

Project Aura 2.0 is a modular, cloud-first AI vision platform that combines:

- **Computer Vision** — Image classification and inference with transfer learning
- **Gesture Recognition** — Hand gesture detection (MediaPipe + fallback)
- **Autonomous AI Agents** — Task orchestration and agent coordination
- **Cloud Integration** — Google Cloud Storage, configurable provider (GCP/AWS/Azure)

## Architecture

The project follows a **cloud-first** development model:

- **Primary development**: Google Colab notebooks (`notebooks/`)
- **Production code**: Modular Python packages (`src/`)
- **Configuration**: Environment variables (`.env.example` template)
- **Testing**: Pytest-compatible test suite (`tests/`)

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full technical architecture.

## Repository Structure

```
project-aura/
+-- notebooks/          # 8 Google Colab notebooks
+-- src/                # Production source code
+-- tests/              # Test suite
+-- configs/            # Configuration files
+-- docs/               # Documentation
+-- scripts/            # Deployment scripts
+-- .env.example        # Environment variable template
+-- .gitignore          # Git ignore rules
+-- ARCHITECTURE.md     # Technical architecture
+-- DEVELOPMENT_ROADMAP.md  # Implementation roadmap
+-- PROJECT_STATUS.md   # Project status tracking
+-- README.md           # This file
```

## Quick Start

### Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Clone this repository:
   ```bash
   !git clone https://github.com/Sineth7077/project-aura.git
   %cd project-aura
   ```
3. Add the project to Python path:
   ```python
   import sys
   sys.path.append('/content/project-aura')
   ```
4. Open any notebook from `notebooks/` and run it.
5. Add your environment variables to Colab Secrets (keys match `.env.example`).

### Local Development

```bash
# Clone the repository
git clone https://github.com/Sineth7077/project-aura.git
cd project-aura

# Copy environment template
cp .env.example .env
# Edit .env with your values

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# Run tests
pytest tests/
```

## Source Modules

| Module | Purpose |
|--------|---------|
| `src/core/config.py` | Central configuration management |
| `src/core/constants.py` | Project-wide constants |
| `src/core/exceptions.py` | Custom exception hierarchy |
| `src/vision/preprocessing.py` | Image preprocessing pipeline |
| `src/vision/models.py` | Vision model wrappers (EfficientNet + fallback) |
| `src/vision/inference.py` | Inference engine |
| `src/agents/orchestrator.py` | AI agent orchestration |
| `src/agents/gesture.py` | Gesture recognition |
| `src/api/routes.py` | REST API route definitions |
| `src/utils/logging.py` | Structured logging |
| `src/utils/gcs.py` | Google Cloud Storage client |

## Google Colab Notebooks

| Notebook | Description |
|----------|-------------|
| `01_environment_setup.ipynb` | Colab env setup, GPU/TPU, dependencies |
| `02_data_ingestion.ipynb` | Dataset loading, preprocessing, cloud storage |
| `03_model_training.ipynb` | CV model training with transfer learning |
| `04_inference_engine.ipynb` | Model inference, real-time processing |
| `05_gesture_recognition.ipynb` | Hand/body gesture recognition |
| `06_ai_agents.ipynb` | Autonomous AI agent orchestration |
| `07_deployment_prep.ipynb` | Model export, containerization, deploy prep |
| `08_monitoring_analytics.ipynb` | Performance monitoring, metrics, dashboards |

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Never commit `.env` — it is gitignored. Use Colab Secrets for notebook execution.

## Testing

```bash
pytest tests/ -v
```

## Roadmap

See [`DEVELOPMENT_ROADMAP.md`](DEVELOPMENT_ROADMAP.md) for the phased
implementation plan.

## Status

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for current project status.

## License

See [`LICENSE`](LICENSE) for license information.
