"""Performance metrics collection for Project Aura 2.0.

Provides metric trackers for inference, gesture recognition, and agent
decisions. All functions compute real measurements from collected data.
Percentile calculations use numpy when available with a pure-Python fallback.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, Optional

import numpy as np


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    return float(np.percentile(np.array(data, dtype=float), pct))


def _median(data: list[float]) -> float:
    if not data:
        return 0.0
    return float(np.median(np.array(data, dtype=float)))


def _mean(data: list[float]) -> float:
    if not data:
        return 0.0
    return float(np.mean(np.array(data, dtype=float)))


class InferenceMetrics:
    """Tracks inference performance metrics."""

    def __init__(self) -> None:
        self.latencies_ms: list[float] = []
        self.success_count = 0
        self.failure_count = 0
        self.total_requests = 0
        self._start_time = __import__("time").time()

    def record(self, latency_ms: float, success: bool = True) -> None:
        self.total_requests += 1
        self.latencies_ms.append(float(latency_ms))
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def summary(self) -> dict[str, Any]:
        elapsed = max(__import__("time").time() - self._start_time, 1e-6)
        throughput = self.total_requests / elapsed if elapsed > 0 else 0.0
        return {
            "total_requests": self.total_requests,
            "successful": self.success_count,
            "failed": self.failure_count,
            "error_rate": self.failure_count / max(self.total_requests, 1),
            "avg_latency_ms": _mean(self.latencies_ms),
            "median_latency_ms": _median(self.latencies_ms),
            "p95_latency_ms": _percentile(self.latencies_ms, 95),
            "p99_latency_ms": _percentile(self.latencies_ms, 99),
            "throughput_per_sec": throughput,
        }

    def clear(self) -> None:
        self.latencies_ms.clear()
        self.success_count = 0
        self.failure_count = 0
        self.total_requests = 0
        self._start_time = __import__("time").time()


class GestureMetrics:
    """Tracks gesture recognition analytics."""

    def __init__(self) -> None:
        self.gesture_counter: Counter[str] = Counter()
        self.confidences: list[float] = []
        self.transitions = []
        self.total_gestures = 0
        self.unknown_count = 0
        self.cooldown_suppressions = 0
        self._previous_gesture = "UNKNOWN"

    def record(self, gesture: str, confidence: float, stable: bool = True) -> None:
        self.total_gestures += 1
        self.gesture_counter[gesture] += 1
        self.confidences.append(float(confidence))
        if gesture == "UNKNOWN":
            self.unknown_count += 1
        if gesture != self._previous_gesture:
            self.transitions.append((self._previous_gesture, gesture))
            self._previous_gesture = gesture

    def record_cooldown_suppression(self) -> None:
        self.cooldown_suppressions += 1

    def summary(self) -> dict[str, Any]:
        total = max(self.total_gestures, 1)
        return {
            "total_gestures": self.total_gestures,
            "distribution": dict(self.gesture_counter),
            "unknown_count": self.unknown_count,
            "unknown_rate": self.unknown_count / total,
            "avg_confidence": _mean(self.confidences),
            "median_confidence": _median(self.confidences),
            "num_transitions": len(self.transitions),
            "transitions": self.transitions[:20],
            "cooldown_suppressions": self.cooldown_suppressions,
        }

    def clear(self) -> None:
        self.gesture_counter.clear()
        self.confidences.clear()
        self.transitions.clear()
        self.total_gestures = 0
        self.unknown_count = 0
        self.cooldown_suppressions = 0
        self._previous_gesture = "UNKNOWN"


class AgentMetrics:
    """Tracks agent decision-making analytics."""

    def __init__(self) -> None:
        self.total_observations = 0
        self.decisions_by_type: Counter[str] = Counter()
        self.decision_latencies_ms: list[float] = []
        self.actions_requested = 0
        self.actions_approved = 0
        self.actions_rejected = 0
        self.executions_succeeded = 0
        self.executions_failed = 0
        self.verification_failures = 0

    def record_observation(self) -> None:
        self.total_observations += 1

    def record_decision(self, action: str, latency_ms: float) -> None:
        self.decisions_by_type[action] += 1
        self.decision_latencies_ms.append(float(latency_ms))

    def record_action_request(self, approved: bool) -> None:
        self.actions_requested += 1
        if approved:
            self.actions_approved += 1
        else:
            self.actions_rejected += 1

    def record_execution(self, success: bool) -> None:
        if success:
            self.executions_succeeded += 1
        else:
            self.executions_failed += 1

    def record_verification_failure(self) -> None:
        self.verification_failures += 1

    def summary(self) -> dict[str, Any]:
        return {
            "total_observations": self.total_observations,
            "decisions_by_type": dict(self.decisions_by_type),
            "avg_decision_latency_ms": _mean(self.decision_latencies_ms),
            "median_decision_latency_ms": _median(self.decision_latencies_ms),
            "p95_decision_latency_ms": _percentile(self.decision_latencies_ms, 95),
            "actions_requested": self.actions_requested,
            "actions_approved": self.actions_approved,
            "actions_rejected": self.actions_rejected,
            "executions_succeeded": self.executions_succeeded,
            "executions_failed": self.executions_failed,
            "execution_success_rate": (
                self.executions_succeeded
                / max(self.executions_succeeded + self.executions_failed, 1)
            ),
            "verification_failures": self.verification_failures,
        }

    def clear(self) -> None:
        self.total_observations = 0
        self.decisions_by_type.clear()
        self.decision_latencies_ms.clear()
        self.actions_requested = 0
        self.actions_approved = 0
        self.actions_rejected = 0
        self.executions_succeeded = 0
        self.executions_failed = 0
        self.verification_failures = 0

