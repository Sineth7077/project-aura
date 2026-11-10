"""Structured logging utilities for Project Aura 2.0.

Provides consistent logging configuration across notebooks and
production code. Supports JSON-formatted logs for cloud observability.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Optional


class JsonFormatter(logging.Formatter):
    """JSON log formatter for cloud-native observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra fields if present
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "getMessage", "exc_info",
                "exc_text", "stack_info", "created",
            ):
                log_entry[key] = value

        return json.dumps(log_entry)


def get_logger(
    name: str = "project-aura",
    level: Optional[str] = None,
    json_format: bool = False,
) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: If True, output JSON-formatted logs for cloud ingestion.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured
        return logger

    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)
    logger.setLevel(log_level)

    handler = logging.StreamHandler(sys.stdout)

    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s"
        )
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False
    return logger


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
) -> None:
    """Setup root logging for the application.

    Call this once at application startup (or Colab notebook entry point).
    """
    get_logger("project-aura", level=level, json_format=json_format)