"""Tests for the monitoring module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np

from src.monitoring.metrics import (
    InferenceMetrics,
    GestureMetrics,
    AgentMetrics,
)
from src.monitoring.analytics import AnalyticsEngine
from src.monitoring.logger import get_aura_logger, AuraLogger
from src.monitoring.resource_monitor import ResourceMonitor


def test_inference_metrics_empty():
    """Test InferenceMetrics with no data."""
    metrics = InferenceMetrics()
    summary = metrics.summary()
    assert summary["total_requests"] == 0
    assert summary["error_rate"] == 0.0


def test_inference_metrics_record():
    """Test recording inference requests."""
    metrics = InferenceMetrics()
    metrics.record(50.0, success=True)
    metrics.record(30.0, success=True)
    metrics.record(100.0, success=False)
    summary = metrics.summary()
    assert summary["total_requests"] == 3
    assert summary["successful"] == 2
    assert summary["failed"] == 1
    assert summary["avg_latency_ms"] == pytest.approx(60.0)


def test_inference_metrics_percentiles():
    """Test percentile calculations."""
    metrics = InferenceMetrics()
    for latency in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        metrics.record(float(latency), success=True)
    summary = metrics.summary()
    assert summary["median_latency_ms"] == pytest.approx(55.0, abs=5.0)
    assert summary["p95_latency_ms"] == pytest.approx(95.5, abs=5.0)


def test_gesture_metrics():
    """Test GestureMetrics recording and summary."""
    metrics = GestureMetrics()
    metrics.record("OPEN_PALM", 0.9)
    metrics.record("FIST", 0.85)
    metrics.record("OPEN_PALM", 0.92)
    metrics.record("UNKNOWN", 0.0)
    summary = metrics.summary()
    assert summary["total_gestures"] == 4
    assert summary["distribution"]["OPEN_PALM"] == 2
    assert summary["unknown_count"] == 1
    assert summary["unknown_rate"] == pytest.approx(0.25)


def test_agent_metrics():
    """Test AgentMetrics recording and summary."""
    metrics = AgentMetrics()
    metrics.record_observation()
    metrics.record_decision("pause", 5.0)
    metrics.record_action_request(approved=True)
    metrics.record_execution(success=True)
    metrics.record_verification_failure()
    summary = metrics.summary()
    assert summary["total_observations"] == 1
    assert summary["decisions_by_type"]["pause"] == 1
    assert summary["actions_approved"] == 1
    assert summary["execution_success_rate"] == 1.0
    assert summary["verification_failures"] == 1


def test_analytics_engine():
    """Test AnalyticsEngine aggregation."""
    engine = AnalyticsEngine()
    engine.add_event({"component": "inference", "latency_ms": 50.0, "timestamp": "2024-01-01T00:00:00+00:00"})
    engine.add_event({"component": "inference", "latency_ms": 30.0, "timestamp": "2024-01-01T00:01:00+00:00"})
    engine.add_event({"component": "gesture", "latency_ms": 20.0, "timestamp": "2024-01-01T00:02:00+00:00"})

    assert engine.event_count == 3
    assert len(engine.filter_by_component("inference")) == 2
    grouped = engine.group_by_component()
    assert "inference" in grouped
    assert "gesture" in grouped
    stats = engine.aggregate_latencies()
    assert stats["count"] == 3
    assert stats["avg"] == pytest.approx(100.0 / 3)


def test_analytics_engine_empty():
    """Test AnalyticsEngine with no data."""
    engine = AnalyticsEngine()
    stats = engine.aggregate_latencies()
    assert stats["count"] == 0
    degradation = engine.detect_degradation()
    assert degradation["degraded"] is False


def test_aura_logger_creation():
    """Test AuraLogger creation and logging."""
    logger = get_aura_logger("test", session_id="sess-001")
    assert logger.component == "test"
    assert logger.session_id == "sess-001"
    # Should not raise
    logger.info("Test message", event_type="test")
    logger.warning("Warning message", event_type="warning")
    logger.error("Error message", event_type="error")


def test_resource_monitor():
    """Test ResourceMonitor snapshot."""
    monitor = ResourceMonitor(check_gpu=False)
    snapshot = monitor.get_snapshot()
    d = snapshot.to_dict()
    assert "cpu_percent" in d
    assert d["gpu_available"] is False  # No GPU check enabled


def test_resource_monitor_history():
    """Test ResourceMonitor history tracking."""
    monitor = ResourceMonitor(check_gpu=False)
    monitor.get_snapshot()
    monitor.get_snapshot()
    history = monitor.get_history()
    assert len(history) == 2
    monitor.clear_history()
    assert len(monitor.get_history()) == 0
