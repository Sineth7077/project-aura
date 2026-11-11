"""Structured logging for Project Aura 2.0.

Provides a rich logger that emits structured log entries with component,
event type, session ID, and error context. Supports INFO, WARNING, ERROR,
and DEBUG levels. Never logs secrets or unnecessary personal data.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Optional


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "event_type": getattr(record, "event_type", "log"),
            "message": record.getMessage(),
        }
        if getattr(record, "session_id", None):
            log_entry["session_id"] = record.session_id
        if record.exc_info:
            log_entry["error"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class AuraLogger:
    """Structured logger for Project Aura components.

    Wraps Python's logging module with structured context: component name,
    event type, and optional session ID. All log calls return structured
    entries suitable for cloud ingestion.
    """

    def __init__(
        self,
        component: str,
        session_id: Optional[str] = None,
    ) -> None:
        """Initialize the structured logger.

        Args:
            component: Component name (e.g. "inference_engine", "gesture").
            session_id: Optional session identifier for trace correlation.
        """
        self.component = component
        self.session_id = session_id
        self._logger = logging.getLogger(f"project-aura.{component}")

        if not self._logger.handlers:
            self._logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(StructuredFormatter())
            self._logger.addHandler(handler)
            self._logger.propagate = False

    # ------------------------------------------------------------------ #
    # Log methods
    # ------------------------------------------------------------------ #
    def _log(
        self,
        level: int,
        message: str,
        event_type: str = "log",
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        log_extra: dict[str, Any] = {
            "component": self.component,
            "event_type": event_type,
        }
        if self.session_id:
            log_extra["session_id"] = self.session_id
        if extra:
            log_extra.update(extra)
        self._logger.log(level, message, extra=log_extra)

    def debug(self, message: str, event_type: str = "log", **kwargs) -> None:
        self._log(logging.DEBUG, message, event_type, kwargs or None)

    def info(self, message: str, event_type: str = "log", **kwargs) -> None:
        self._log(logging.INFO, message, event_type, kwargs or None)

    def warning(self, message: str, event_type: str = "log", **kwargs) -> None:
        self._log(logging.WARNING, message, event_type, kwargs or None)

    def error(self, message: str, event_type: str = "log", **kwargs) -> None:
        self._log(logging.ERROR, message, event_type, kwargs or None)

    def exception(self, message: str, event_type: str = "error", **kwargs) -> None:
        self._logger.exception(message, extra={"component": self.component, "event_type": event_type})


def get_aura_logger(
    component: str = "project-aura",
    session_id: Optional[str] = None,
) -> AuraLogger:
    """Factory function for structured loggers."""
    return AuraLogger(component=component, session_id=session_id)
