# Project Aura - Development Roadmap

## Vision

Build a foundation for an agentic robot-control system where an AI system can
perceive its environment, interpret human gestures, make decisions, and
eventually control permitted external systems. Current stage: **software-first**.
No physical robot control, no external cloud deployment, no paid services.

## Roadmap Phases

### Phase 1: Repository Initialization & Architecture — COMPLETE

**Status:** Complete  
**Target:** Complete repository foundation with architecture, documentation,
source modules, and Colab notebooks.

**Objectives (all done):**
- [x] Repair and initialize Git repository
- [x] Create modular folder structure (`src/`, `notebooks/`, `tests/`, `deploy/`,
  `scripts/`, `archive/`)
- [x] Create core configuration, constants, and exceptions
- [x] Create vision module (preprocessing, models, inference, hand tracking,
  gesture recognition, smoothing, events, object detection)
- [x] Create agents module (observation, context, decision engine, action
  registry, permission manager, executor, verification, orchestrator)
- [x] Create API module (FastAPI app, config, routes, schemas, services)
- [x] Create monitoring module (logger, metrics, analytics, resource monitor)
- [x] Create optional Streamlit dashboard
- [x] Create 8 Google Colab notebooks (`.ipynb`)
- [x] Create `.env.example` and update `.gitignore`
- [x] Create `ARCHITECTURE.md`, `DEVELOPMENT_ROADMAP.md`, `PROJECT_STATUS.md`,
  `README.md`, `agents.md`
- [x] Validate all notebooks as proper Jupyter notebooks (JSON + syntax)
- [x] Run test suite and verify all tests pass (157 passing)

### Phase 2: Notebook Implementation & Validation — COMPLETE

**Status:** Complete  
**Target:** Flesh out all 8 Colab notebooks with real, executable code.

**Objectives (all done):**
- [x] Notebook 01: Environment setup — dependency checks, CPU/GPU detection,
  path config
- [x] Notebook 02: Data ingestion — load datasets, preprocess, synthetic pipeline
- [x] Notebook 03: Model training — EfficientNet-B0 transfer learning, CIFAR-10
  (85.71% test accuracy), checkpoints
- [x] Notebook 04: Inference engine — model reconstruction, checkpoint loading,
  CPU-compatible inference, batch, evaluation, exports
- [x] Notebook 05: Gesture recognition — MediaPipe hand tracking (graceful
  fallback), 7-class geometric classifier, temporal smoothing, structured
  gesture events, visualization, tests, export
- [x] Notebook 06: AI agents — observation schema, context manager, deterministic
  decision engine, action registry, permission manager, executor, verification,
  complete decision trace, NB05/NB04 integration, tests
- [x] Notebook 07: Deployment prep — centralized config, FastAPI backend, Pydantic
  schemas, health/readiness/model info, inference/gesture/agent routes, WebSocket
  prep, lazy model loading, image limits, Docker prep, API tests
- [x] Notebook 08: Monitoring — structured logging, inference/gesture/agent
  metrics, resource monitoring, analytics engine, Matplotlib visualizations,
  optional Streamlit dashboard, JSON/CSV/PNG/TXT export, tests

**Success Criteria (met):**
- [x] All notebooks execute end-to-end (05–08 executed with 0 error outputs;
  01–04 previously executed)
- [x] Notebooks import and use `src/` modules
- [x] No fake or placeholder functionality; synthetic data clearly labeled
- [x] All notebooks statically valid (JSON + code-cell syntax)

### Phase 3: Testing & Quality Assurance — PARTIAL

**Status:** In Progress  
**Target:** Comprehensive test coverage and code quality.

**Objectives:**
- [x] Test suite exists and passes (157 tests)
- [x] Unit tests for gesture, smoothing, agent observation/decision/permissions/
  executor, monitoring, API, inference, preprocessing, model service,
  object detector, orchestrator, config, exceptions
- [x] Integration tests for agent pipeline (test_agent_pipeline.py)
- [ ] Achieve 80%+ test coverage on `src/` modules (not measured)
- [ ] Add linting (ruff/flake8) and formatting (black)
- [ ] Add type checking (mypy)
- [ ] Create CI workflow (GitHub Actions)
- [ ] Performance benchmarks for inference and training

### Phase 4: Cloud Deployment Preparation — COMPLETE (prepared, not deployed)

**Status:** Complete (infrastructure prepared only)  
**Target:** Prepare for production deployment on cloud infrastructure.

**Objectives:**
- [x] Dockerfile for containerized inference API (`deploy/Dockerfile`, CPU image)
- [x] `deploy/requirements.txt` (CPU-compatible; GPU packages optional/commented)
- [x] Health checks where appropriate (`GET /health`, `/ready`)
- [x] Cloud Build / CI/CD pipeline — NOT created (planned)
- [x] GCS bucket configuration — NOT provisioned (no cloud accounts)
- [x] Cloud Run / App Runner deployment configuration — NOT deployed
- [x] IAM and service account setup — NOT done (no secrets, no cloud)
- [x] Secret management — NOT done (`.env` template only, no secrets committed)

**Constraints (honored):**
- [x] No paid infrastructure provisioned during development
- [x] No external cloud accounts created
- [x] Nothing deployed to production

### Phase 5: Production Deployment — NOT STARTED

**Status:** Planned (not started)  
**Target:** Deploy to production cloud infrastructure.

**Objectives (not started):**
- [ ] Deploy inference API to Cloud Run / App Runner
- [ ] Configure monitoring and alerting
- [ ] Automated model retraining pipeline
- [ ] Production runbook
- [ ] Security review

**Note:** Deployment only after all previous phases complete and tested. Nothing
has been deployed.

## Milestone Timeline

| Milestone | Target | Status |
|-----------|--------|--------|
| M1: Repo initialization & architecture | Week 1 | Complete |
| M2: Notebook implementation | Week 2-3 | Complete |
| M3: Testing & QA | Week 4 | In Progress |
| M4: Cloud deployment prep | Week 5-6 | Complete (prepared only) |
| M5: Production deployment | Week 7+ | Not started |

## Success Metrics

- [x] All 8 notebooks execute successfully
- [ ] 80%+ test coverage on `src/` modules (not yet measured)
- [x] Zero credentials or secrets in version control
- [x] Modular architecture supporting future extensions
- [x] Clear path from Colab development to production deployment (prepared only)

## Known Limitations (honest)

- MediaPipe `mp.solutions.hands` is unavailable with the installed mediapipe
  1.2.2; `HandTracker` degrades gracefully. Geometric gesture recognition works
  offline with synthetic/provided landmarks.
- No webcam, GPU, or Google Drive is assumed or required.
- Dashboard requires Streamlit (optional dependency).
- Not deployed to any external hosting service; Docker config is preparation
  only.
