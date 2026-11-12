# Project Aura Phase 2 - Integration Map

This document maps each Phase 2 feature to the existing Project Aura modules that will be reused.

| Phase 2 Section | Feature | Existing Module | Class/Function | Reuse Status |
|---|---|---|---|---|
| 1 | Environment initialization | `src.core.config` | `load_config()`, `AuraConfig` | Direct reuse |
| 1 | Colab/GPU detection | `src.core.config` | `_detect_colab()`, `detect_gpu()` | Direct reuse |
| 2 | Webcam capture | (new in notebook) | JavaScript bridge + OpenCV | New |
| 3 | YOLO object detection | `src.vision.object_detector` | `ObjectDetector` | Direct reuse |
| 3 | YOLO model loading | `src.vision.object_detector` | `ObjectDetector.load_model()` | Direct reuse |
| 4 | Person detection/counting | `src.vision.object_detector` | `ObjectDetector.detect()` (person class) | Direct reuse |
| 5 | Hand tracking | `src.vision.hand_tracker` | `HandTracker` (graceful fallback) | Reuse + MediaPipe Tasks fallback |
| 5 | Hand landmarks | `src.vision.hand_tracker` | `HandLandmarks` (21 landmarks) | Direct reuse |
| 6 | Gesture recognition | `src.vision.gesture_recognizer` | `GestureRecognizer` | Direct reuse |
| 6 | Gesture classes | `src.vision.gesture_recognizer` | `OPEN_PALM`, `FIST`, `THUMBS_UP`, `POINTING`, `VICTORY`, `PINCH`, `UNKNOWN` | Direct reuse |
| 6 | Temporal smoothing | `src.vision.gesture_smoother` | `GestureSmoother` | Direct reuse |
| 6 | Gesture events | `src.vision.gesture_events` | `GestureEvent`, `GestureEventGenerator` | Direct reuse |
| 6 | Landmark geometry | `src.vision.landmark_processor` | `validate_landmarks`, `distance`, `extended_fingers`, etc. | Direct reuse |
| 7 | Finger air drawing | (new in notebook) | Canvas state management | New in notebook |
| 8 | Agent observation | `src.agents.observation` | `Observation`, `ObservationType`, `validate_observation` | Direct reuse |
| 8 | Agent context | `src.agents.context` | `ContextManager`, `ContextSnapshot` | Direct reuse |
| 8 | Agent decision | `src.agents.decision_engine` | `DecisionEngine`, `Decision` | Direct reuse |
| 8 | Action registry | `src.agents.action_registry` | `ActionRegistry`, `ActionEntry` | Direct reuse |
| 8 | Permission manager | `src.agents.permission_manager` | `PermissionManager`, `PermissionRequest`, `PermissionResult` | Direct reuse |
| 8 | Action executor | `src.agents.executor` | `ActionExecutor`, `ExecutionResult` | Direct reuse |
| 8 | Verification | `src.agents.verification` | `VerificationSystem`, `VerificationRecord` | Direct reuse |
| 8 | Agent service | `src.api.services.agent_service` | `AgentService` (full pipeline) | Direct reuse |
| 9 | Inference metrics | `src.monitoring.metrics` | `InferenceMetrics` | Direct reuse |
| 9 | Gesture metrics | `src.monitoring.metrics` | `GestureMetrics` | Direct reuse |
| 9 | Agent metrics | `src.monitoring.metrics` | `AgentMetrics` | Direct reuse |
| 9 | Resource monitoring | `src.monitoring.resource_monitor` | `ResourceMonitor` | Direct reuse |
| 9 | Structured logging | `src.monitoring.logger` | `get_aura_logger`, `AuraLogger` | Direct reuse |
| 10 | Integrated pipeline | All above | Combined in notebook | Integration |
| 11 | Report generation | `json`, `csv`, `matplotlib` | Standard library + matplotlib | New in notebook |

**Modules NOT modified:** All existing `src/` modules are reused without modification.

**New code:** Only in `notebooks/09_live_vision_testing.ipynb` - no new source modules created.
