# Project Aura

AI Vision Control Center — an extensible computer vision and autonomous
software-agent platform that observes visual input, detects objects and people,
recognizes hand gestures, interprets observations, makes controlled decisions,
executes authorized software actions, verifies outcomes, and monitors system
performance.

**Current stage: software-first.** The system perceives its environment,
interprets human gestures, makes deterministic and explainable decisions, and
executes permitted software actions. No physical robot control, no external
cloud deployment, and no paid services are part of this stage.

## Overview

Project Aura combines:

- **Computer Vision** — image classification/inference (EfficientNet-B0,
  CIFAR-10), optional YOLO object detection, CPU-compatible inference engine
- **Gesture Recognition** — hand landmark tracking, 7-class geometric gesture
  classification (OPEN_PALM, FIST, THUMBS_UP, POINTING, VICTORY, PINCH,
  UNKNOWN), temporal smoothing, structured gesture events
- **Autonomous AI Agents** — observation → context → decision → permission
  check → action execution → verification → event logging; all actions are
  permission-controlled and allowlisted
- **API** — FastAPI backend (`/health`, `/ready`, `/model/info`,
  `/inference`, `/gesture`, `/agent/decide`, WebSocket `/ws/gesture`)
- **Monitoring & Analytics** — structured logging, real metrics, resource
  monitoring, analytics engine, Matplotlib visualizations, JSON/CSV/PNG/TXT
  export, optional Streamlit dashboard

## Architecture

The system follows a modular, layered architecture:

```
IMAGE / VIDEO / CAMERA
        v
OBJECT DETECTION
        v
HAND TRACKING
        v
GESTURE RECOGNITION
        v
STRUCTURED OBSERVATIONS
        v
CONTEXT MANAGER
        v
DECISION ENGINE (deterministic, rule-based)
        v
PERMISSION MANAGER (allowlist, audit trail)
        v
ACTION EXECUTOR (dry-run supported)
        v
VERIFICATION (expected vs actual state)
        v
EVENT LOGGING / METRICS
        v
FASTAPI API
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full technical architecture,
including module-by-module documentation and interface contracts.

## Repository Structure

```
project-aura/
+-- notebooks/          # 8 Google Colab notebooks (01-08)
+-- src/                # Production source code (core, vision, agents, api,
+   |                   # monitoring, dashboard)
+-- tests/              # Pytest test suite (157 tests, all passing)
+-- deploy/             # CPU-compatible Dockerfile + requirements.txt
+-- archive/            # Archived historical executed-notebook copies
+-- scripts/            # Deployment and utility scripts
+-- .env.example        # Environment variable template (no secrets)
+-- .gitignore
+-- README.md
+-- ARCHITECTURE.md
+-- DEVELOPMENT_ROADMAP.md
+-- PROJECT_STATUS.md
+-- agents.md           # Engineering rules
```

## Quick Start

### Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Clone this repository:
   ```bash
   !git clone https://github.com/Sineth7077/project-aura.git
   %cd project-aura
   ```
3. Add the project to the Python path:
   ```python
   import sys
   sys.path.append('/content/project-aura')
   ```
4. Open any notebook from `notebooks/` and run it.
   Notebooks should be executed in order: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08.
   Each notebook runs independently from top to bottom.
5. Add environment variables to Colab Secrets (keys match `.env.example`).

### Local Development

```bash
# Clone the repository
git clone https://github.com/Sineth7077/project-aura.git
cd project-aura

# Copy environment template (do not commit .env)
cp .env.example .env
# Edit .env with your values as needed

# Create virtual environment (Python 3.11 preferred)
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install dependencies (CPU-compatible path)
pip install -r deploy/requirements.txt
# For EfficientNet inference, also install: pip install torch torchvision

# Run tests
pytest tests/

# Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Start the optional dashboard (requires Streamlit)
pip install streamlit
streamlit run src/dashboard/app.py
```

## Notebook Execution Order

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_environment_setup.ipynb` | Environment detection, dependencies, CPU/GPU |
| 02 | `02_data_ingestion.ipynb` | Data ingestion, preprocessing, validation |
| 03 | `03_model_training.ipynb` | EfficientNet-B0 transfer learning (CIFAR-10) |
| 04 | `04_inference_engine.ipynb` | Model reconstruction, inference, evaluation, exports |
| 05 | `05_gesture_recognition.ipynb` | Hand tracking, gesture classification, smoothing, events |
| 06 | `06_ai_agents.ipynb` | Autonomous agent pipeline, decision trace |
| 07 | `07_deployment_prep.ipynb` | FastAPI backend, schemas, Docker prep |
| 08 | `08_monitoring_analytics.ipynb` | Logging, metrics, analytics, visualizations, export |

Notebooks 05–08 have been executed end-to-end in a supported environment with
zero error outputs. Notebooks 01–04 were previously executed. All notebooks are
statically valid (JSON + code-cell syntax) and run independently.

## Source Modules

| Module | Purpose |
|--------|---------|
| `src.core.config` | Central configuration (AuraConfig, env-driven) |
| `src.core.constants` | Project-wide constants |
| `src.core.exceptions` | Custom exception hierarchy |
| `src.vision.preprocessing` | Image resize / normalize / batch |
| `src.vision.models` | Vision model wrappers (EfficientNet + fallback) |
| `src.vision.inference` | Inference engine |
| `src.vision.hand_tracker` | MediaPipe hand tracking (graceful fallback) |
| `src.vision.gesture_recognizer` | Rule-based 7-class gesture classifier |
| `src.vision.gesture_smoother` | Temporal gesture smoothing |
| `src.vision.gesture_events` | Canonical gesture event schema |
| `src.vision.object_detector` | Optional YOLO object detection |
| `src.agents.observation` | Observation schema + validation |
| `src.agents.context` | Short-term bounded context manager |
| `src.agents.decision_engine` | Deterministic rule-based decisions |
| `src.agents.action_registry` | Allowlist of safe software actions |
| `src.agents.permission_manager` | Permission validation + audit |
| `src.agents.executor` | Controlled action executor (dry-run) |
| `src.agents.verification` | Action result verification |
| `src.agents.orchestrator` | Multi-agent task management |
| `src.api.main` | FastAPI application |
| `src.api.config` | API configuration |
| `src.api.routes.*` | Health, inference, gesture, agent routes |
| `src.api.schemas.*` | Pydantic request/response schemas |
| `src.api.services.*` | Model, gesture, agent services |
| `src.monitoring.logger` | Structured JSON logging |
| `src.monitoring.metrics` | Inference / gesture / agent metrics |
| `src.monitoring.analytics` | Analytics engine |
| `src.monitoring.resource_monitor` | CPU / RAM / disk / GPU monitoring |
| `src.dashboard.app` | Optional Streamlit dashboard |

## Testing

```bash
pytest tests/
```

All 157 tests pass. Tests use deterministic offline fixtures and mocks — no
internet access or pretrained-weight downloads are required during testing.

## Deployment Preparation

Project Aura is prepared for deployment but has **not been deployed** to any
external hosting service.

- `deploy/Dockerfile` — CPU-compatible image (`python:3.11-slim`), dependency
  installation, app startup, port 8000, health check
- `deploy/requirements.txt` — CPU-compatible dependencies; GPU packages
  (torch/torchvision) are commented out
- `GET /health` and `GET /ready` endpoints for health checks
- GPU deployment should be documented separately

See [`DEVELOPMENT_ROADMAP.md`](DEVELOPMENT_ROADMAP.md) for phased planning and
[`PROJECT_STATUS.md`](PROJECT_STATUS.md) for current status.

## Configuration

Copy `.env.example` to `.env` and adjust as needed. Never commit `.env` — it is
gitignored. Use Colab Secrets for notebook execution. No credentials or secrets
are stored in version control.

## Known Limitations

- MediaPipe `mp.solutions.hands` is unavailable with the installed mediapipe
  1.2.2; `HandTracker` degrades gracefully and reports itself as unavailable.
  Geometric gesture recognition works offline with synthetic/provided landmarks.
  Live hand tracking requires a compatible MediaPipe version.
- No webcam, GPU, or Google Drive is assumed or required; all are optional.
- Dashboard requires Streamlit (optional dependency); notebooks and API work
  without it.
- Not deployed to any external hosting service; Docker configuration is
  preparation only.

## License

See [`LICENSE`](LICENSE) for license information.
