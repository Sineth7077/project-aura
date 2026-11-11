"""Analytics engine for Project Aura 2.0.

Provides reusable functions for aggregating metrics, filtering by time range,
grouping by component, comparing sessions, calculating performance
statistics, and detecting errors and performance degradation.
"""

from __future__ import annotations

from typing import Any, Optional
from datetime import datetime, timezone, timedelta

import numpy as np


class AnalyticsEngine:
    """Reusable analytics engine for aggregating monitoring data."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def add_event(self, event: dict[str, Any]) -> None:
        """Add a timestamped event to the store."""
        if "timestamp" not in event:
            event["timestamp"] = datetime.now(timezone.utc).isoformat()
        self._events.append(event)

    def filter_by_time_range(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Filter events within a time range."""
        result = self._events
        if start:
            result = [e for e in result if e.get("timestamp", "") >= start]
        if end:
            result = [e for e in result if e.get("timestamp", "") <= end]
        return list(result)

    def filter_by_component(self, component: str) -> list[dict[str, Any]]:
        """Filter events for a specific component."""
        return [e for e in self._events if e.get("component") == component]

    def group_by_component(self) -> dict[str, list[dict[str, Any]]]:
        """Group all events by component name."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for event in self._events:
            comp = event.get("component", "unknown")
            grouped.setdefault(comp, []).append(event)
        return grouped

    def aggregate_latencies(self, events: list[dict[str, Any]] | None = None) -> dict[str, float]:
        """Aggregate latency statistics from events."""
        latencies = []
        for e in (events or self._events):
            if "latency_ms" in e and isinstance(e["latency_ms"], (int, float)):
                latencies.append(float(e["latency_ms"]))
        if not latencies:
            return {"count": 0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(latencies, dtype=float)
        return {
            "count": len(latencies),
            "avg": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
            "min": float(np.min(arr)),
        }

    def detect_errors(self, events: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Return events with error indicators."""
        return [
            e
            for e in (events or self._events)
            if e.get("level") in ("ERROR", "CRITICAL") or e.get("error")
        ]

    def detect_degradation(
        self,
        events: list[dict[str, Any]] | None = None,
        threshold_ms: float = 200.0,
    ) -> dict[str, Any]:
        """Detect performance degradation by comparing latency percentiles.

        Splits events into first half vs second half and checks if the
        p95 latency has increased significantly.
        """
        evts = events or self._events
        latencies = [
            float(e["latency_ms"])
            for e in evts
            if "latency_ms" in e and isinstance(e["latency_ms"], (int, float))
        ]
        if len(latencies) < 4:
            return {"degraded": False, "reason": "insufficient data"}

        half = len(latencies) // 2
        first_half = np.array(latencies[:half])
        second_half = np.array(latencies[half:])

        first_p95 = float(np.percentile(first_half, 95))
        second_p95 = float(np.percentile(second_half, 95))
        degradation = (second_p95 - first_p95) / max(first_p95, 1e-6)

        return {
            "degraded": degradation > 0.2 and second_p95 > threshold_ms,
            "first_half_p95_ms": first_p95,
            "second_half_p95_ms": second_p95,
            "degradation_ratio": degradation,
            "reason": "latency increase" if degradation > 0.2 else "normal",
        }

    def compare_sessions(
        self,
        session_a: list[dict[str, Any]],
        session_b: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compare latency stats between two sessions."""
        a_stats = self.aggregate_latencies(session_a)
        b_stats = self.aggregate_latencies(session_b)
        return {
            "session_a": a_stats,
            "session_b": b_stats,
            "p95_change_pct": (
                (b_stats["p95"] - a_stats["p95"]) / max(a_stats["p95"], 1e-6) * 100
                if a_stats["p95"] > 0
                else 0.0
            ),
        }

    def clear(self) -> None:
        """Clear all stored events."""
        self._events.clear()

    @property
    def event_count(self) -> int:
        return len(self._events)
