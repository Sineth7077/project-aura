# Project Aura - Technical Architecture

## Overview

Project Aura is an AI Vision Control Center — an extensible computer vision and
autonomous software-agent platform that observes visual input, detects objects
and people, recognizes hand gestures, interprets observations, makes controlled
decisions, executes authorized software actions, verifies outcomes, and monitors
system performance.

Primary development is in Google Colab notebooks (`notebooks/`); production code
lives in `src/` and is importable by notebooks. The current stage is
**software-first**: no physical robot control, no external cloud deployment, no
paid services.

## Design Principles

1. **Modular**: Clear separation between core config, vision, agents, API,
   monitoring, and dashboard layers. Each module is independently testable.
2. **Software-first, safe by default**: All agent actions are permission
   controlled. No arbitrary code execution, no shell access, no unrestricted
   filesystem access, no physical robot control.
3. **Deterministic and explainable**: The decision engine is rule-based (not an
   LLM). Every decision records observation, selected action, explanation,
   permission result, execution result, and verification result.
4. **Graceful degradation**: ML components work with and without heavy
   dependencies (PyTorch, MediaPipe, psutil, Streamlit). CPU is always
   supported; GPU is optional.
5. **No secrets**: `.env` is gitignored; `.env.example` is the template.
6. **Executable notebooks**: All eight `.ipynb` files are valid Jupyter
   notebooks designed for Google Colab and runnable locally.

## Repository Structure

```
project-aura/
+-- archive/                    # Archived historical executed-notebook copies
+-- deploy/                     # CPU-compatible Dockerfile + requirements.txt
+-- notebooks/                  # 8 Google Colab notebooks (01-08)
+   +-- 01_environment_setup.ipynb
+   +-- 02_data_ingestion.ipynb
+   +-- 03_model_training.ipynb
+   +-- 04_inference_engine.ipynb
+   +-- 05_gesture_recognition.ipynb
+   +-- 06_ai_agents.ipynb
+   +-- 07_deployment_prep.ipynb
+   +-- 08_monitoring_analytics.ipynb
+-- scripts/                    # Deployment and utility scripts
+-- src/                        # Production source code (importable by notebooks)
+   +-- core/
+   |   +-- config.py           # Central configuration (AuraConfig, load_config)
+   |   +-- constants.py        # Project-wide constants
+   |   +-- exceptions.py       # Custom exception hierarchy (AuraException base)
+   +-- vision/
+   |   +-- preprocessing.py    # Image resize / normalize / batch / denormalize
+   |   +-- models.py           # VisionModelBase, EfficientNetModel, create_model()
+   |   +-- inference.py        # InferenceEngine (load/preprocess/predict/batch)
+   |   +-- hand_tracker.py     # MediaPipe Hands wrapper (graceful fallback)
+   |   +-- landmark_processor.py # Geometry: distances, angles, finger states
+   |   +-- gesture_recognizer.py # Rule-based 7-class gesture classifier
+   |   +-- gesture_smoother.py   # Temporal smoothing (majority vote, cooldown)
+   |   +-- gesture_events.py     # Canonical GestureEvent schema
+   |   +-- object_detector.py    # YOLO object detection (optional dependency)
+   +-- agents/
+   |   +-- observation.py      # Observation schema + validation
+   |   +-- context.py          # ContextManager (short-term bounded context)
+   |   +-- decision_engine.py  # Deterministic rule-based decision engine
+   |   +-- action_registry.py  # Allowlist of safe software actions
+   |   +-- permission_manager.py # Permission validation + audit trail
+   |   +-- executor.py         # Controlled action executor (dry-run support)
+   |   +-- verification.py     # Action result verification
+   |   +-- orchestrator.py     # AgentOrchestrator (multi-agent task mgmt)
+   |   +-- gesture.py          # Legacy gesture module (retained)
+   +-- api/
+   |   +-- main.py             # FastAPI app (lifespan, CORS, routes)
+   |   +-- config.py           # APIConfig (host, port, limits, CORS)
+   |   +-- routes/
+   |   |   +-- health.py       # GET /health, /ready, /model/info
+   |   |   +-- inference.py    # POST /inference
+   |   |   +-- gesture.py      # POST /gesture
+   |   |   +-- agent.py        # POST /agent/decide, WebSocket /ws/gesture
+   |   +-- schemas/
+   |   |   +-- health.py       # HealthResponse, ReadinessResponse, ModelInfoResponse
+   |   |   +-- inference.py    # InferenceRequest/Response/Result
+   |   |   +-- gesture.py      # GestureRequest/Response/Result
+   |   |   +-- agent.py        # AgentDecideRequest/Response
+   |   +-- services/
+   |       +-- model_service.py  # Lazy model loading (ModelService)
+   |       +-- gesture_service.py  # GestureService
+   |       +-- agent_service.py  # AgentService (full pipeline)
+   +-- monitoring/
+   |   +-- logger.py           # Structured logging (AuraLogger, JSON)
+   |   +-- metrics.py          # InferenceMetrics, GestureMetrics, AgentMetrics
+   |   +-- analytics.py        # AnalyticsEngine (aggregation, degradation)
+   |   +-- resource_monitor.py # ResourceMonitor (CPU/RAM/disk/GPU via psutil)
+   +-- dashboard/
+   |   +-- app.py              # Optional Streamlit analytics dashboard
+   +-- __init__.py
+-- tests/                      # Pytest test suite (157 tests)
+-- .env.example                # Environment variable template (no secrets)
+-- .gitignore
+-- README.md
+-- ARCHITECTURE.md
+-- DEVELOPMENT_ROADMAP.md
+-- PROJECT_STATUS.md
+-- agents.md                   # Engineering rules
```

## Data Flow (cross-component integration)

```
IMAGE / VIDEO / CAMERA
        |
        v
OBJECT DETECTION   (src.vision.object_detector / src.vision.inference)
        |
        v
HAND TRACKING      (src.vision.hand_tracker -> HandTrackingResult)
        |
        v
GESTURE RECOGNITION (src.vision.gesture_recognizer -> GesturePrediction)
        |
        v
TEMPORAL SMOOTHING  (src.vision.gesture_smoother -> SmoothedGesture)
        |
        v
GESTURE EVENTS      (src.vision.gesture_events -> GestureEvent)
        |
        v
STRUCTURED OBSERVATIONS (src.agents.observation -> Observation)
        |
        v
CONTEXT MANAGER     (src.agents.context -> ContextManager / ContextSnapshot)
        |
        v
DECISION ENGINE     (src.agents.decision_engine -> Decision)
        |
        v
PERMISSION MANAGER  (src.agents.permission_manager -> PermissionResult)
        |
        v
ACTION EXECUTOR     (src.agents.executor -> ExecutionResult)
        |
        v
VERIFICATION        (src.agents.verification -> VerificationRecord)
        |
        v
EVENT LOGGING / METRICS (src.monitoring.*)
        |
        v
FASTAPI             (src.api.main -> /health, /ready, /model/info, /inference,
                     /gesture, /agent/decide, /ws/gesture)
```

### Key interfaces

- **Gesture event schema** (`GestureEvent`): `event_id`, `timestamp`, `gesture`,
  `hand`, `confidence`, `landmarks`, `bbox`, `prev_gesture`, `curr_gesture`,
  `metadata`. Produced by Notebook 05, consumed verbatim by Notebook 06.
- **Observation schema** (`Observation`): `observation_id`, `timestamp`, `source`,
  `type` (ObservationType), `data`, `confidence`, `metadata`. Produced by
  Notebooks 04/05, consumed by Notebook 06.
- **Decision** (`Decision`): `decision_id`, `timestamp`, `observation_id`,
  `selected_action`, `confidence`, `explanation`, `permission_status`,
  `verification_status`, `context_snapshot`.
- **API**: Pydantic schemas; consistent JSON responses; `GET /health`, `/ready`,
  `/model/info`; `POST /inference`, `/gesture`, `/agent/decide`;
  WebSocket `/ws/gesture`.

## Module Layers

### Core Layer (`src/core/`)
Configuration (`AuraConfig` with `ColabConfig`, `VisionConfig`, `GestureConfig`,
`AgentConfig`, `CloudConfig`), constants, and the exception hierarchy
(`AuraException` base). Auto-detects Colab runtime, GPU, and TPU lazily.

### Vision Layer (`src/vision/`)
- `preprocessing.py`: resize (aspect-ratio preserving), ImageNet normalization,
  batch preparation (NCHW), denormalization for display. Requires OpenCV.
- `models.py`: `VisionModelBase` ABC, `EfficientNetModel` (torchvision with a
  lightweight statistical fallback when PyTorch is unavailable), `create_model()`
  factory.
- `inference.py`: `InferenceEngine` — model lifecycle, preprocessing,
  single/batch prediction.
- `hand_tracker.py`: `HandTracker` — MediaPipe Hands wrapper (up to 2 hands,
  21 landmarks, handedness, bounding boxes). Degrades gracefully when MediaPipe
  or `mp.solutions` is unavailable.
- `landmark_processor.py`: pure-geometry utilities (distances, angles,
  finger extension, handedness normalization). Testable with synthetic data.
- `gesture_recognizer.py`: rule-based 7-class classifier
  (OPEN_PALM, FIST, THUMBS_UP, POINTING, VICTORY, PINCH, UNKNOWN) with
  confidence. Not a neural network.
- `gesture_smoother.py`: `GestureSmoother` — rolling history, majority vote,
  minimum stable frames, cooldown, transition detection.
- `gesture_events.py`: `GestureEvent` + `GestureEventGenerator` (canonical
  event schema shared between Notebooks 05 and 06).
- `object_detector.py`: YOLO object detection wrapper (optional `ultralytics`
  dependency) with `Detection`/`DetectionResult` and conversion to `Observation`.

### Agents Layer (`src/agents/`)
- `observation.py`: `Observation` dataclass, `ObservationType` enum,
  `validate_observation()`.
- `context.py`: `ContextManager` — bounded short-term context (objects, person
  count, latest/previous gesture, previous actions, recent events, TTL expiry).
  Image data is never stored.
- `decision_engine.py`: `DecisionEngine` — deterministic rule-based decisions
  (first matching rule wins). Built-in rules map gestures to actions; rules are
  configurable and explainable.
- `action_registry.py`: `ActionRegistry` — allowlist of safe software actions
  (pause, resume, display_alert, save_screenshot, record_event,
  update_visualization, noop, request_verification). Sensitive actions require
  user confirmation.
- `permission_manager.py`: `PermissionManager` — allowlist validation, sensitive
  action confirmation, audit trail, safe defaults. An AI decision never bypasses
  permissions.
- `executor.py`: `ActionExecutor` — validation, permission checks, controlled
  execution, dry-run mode, exception handling, structured `ExecutionResult`.
- `verification.py`: `VerificationSystem` — expected vs actual state comparison,
  custom checks, duplicate prevention, `VerificationRecord`. Never reports
  success without evidence.
- `orchestrator.py`: `AgentOrchestrator` — multi-agent registration and task
  lifecycle management.

### API Layer (`src/api/`)
- `main.py`: FastAPI application with lifespan (model pre-load on startup,
  cleanup on shutdown), CORS, request logging, global error handling.
- `config.py`: `APIConfig` — host, port, workers, CORS, image size limits,
  model paths, thresholds. Environment-variable driven with safe defaults.
- `routes/`: `GET /health`, `GET /ready`, `GET /model/info`,
  `POST /inference`, `POST /gesture`, `POST /agent/decide`,
  WebSocket `/ws/gesture`.
- `schemas/`: Pydantic request/response schemas with validation (e.g. agent
  decision confidence constrained to [0, 1]).
- `services/`: `ModelService` (lazy, cached, thread-safe model loading),
  `GestureService`, `AgentService` (full observation→verification pipeline).

### Monitoring Layer (`src/monitoring/`)
- `logger.py`: `AuraLogger` — structured JSON logging with component, event
  type, session ID, severity (DEBUG/INFO/WARNING/ERROR). No secrets logged.
- `metrics.py`: `InferenceMetrics`, `GestureMetrics`, `AgentMetrics` — real
  measurements (latency percentiles, distributions, counts, success/failure).
- `analytics.py`: `AnalyticsEngine` — aggregation, time filtering, component
  grouping, session comparison, degradation detection.
- `resource_monitor.py`: `ResourceMonitor` — CPU, RAM, process memory, disk,
  optional GPU (psutil + PyTorch). Unavailable metrics reported as None, never
  invented.

### Dashboard Layer (`src/dashboard/`)
- `app.py`: optional Streamlit analytics dashboard. The notebooks and API work
  without Streamlit.

## Google Colab Integration

All eight notebooks import from `src/` by adding the project root to
`sys.path` (e.g. `/content/project-aura` after a git clone). Each notebook runs
independently from top to bottom, uses CPU fallback where possible, and does not
assume GPU, webcam, or Google Drive availability.

## Cloud Deployment Path (prepared, not deployed)

- **Development**: Google Colab notebooks; local CPU execution.
- **Staging**: Containerized FastAPI inference API (`deploy/Dockerfile`),
  CPU-compatible image (`python:3.11-slim`), health checks.
- **Production**: Not provisioned. No cloud accounts, no paid services, no
  external deployment.

## Security Considerations

- No credentials in git (`.env` gitignored, `.env.example` template)
- Action allowlist; sensitive actions require explicit user confirmation
- No arbitrary code/shell/filesystem execution by the agent
- Image upload size limits; input validation on all API endpoints
- Structured logging without sensitive data

## Future Extensibility

- Add vision models by extending `VisionModelBase` and updating `create_model()`.
- Add agent decision rules by registering predicates in `DecisionEngine`.
- Add API routes via `APIRouter` in `src/api/routes/`.
- Optional LLM adapter must remain disabled by default and pass every proposed
  decision through the `PermissionManager`.
