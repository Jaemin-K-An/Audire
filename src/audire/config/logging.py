"""Structured logging.

JSON output by default so that experiment logs are machine-checkable artifacts;
human-readable console output when ``AUDIRE_LOG_FORMAT=console``.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog

_CONFIGURED = False


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Configure structlog once per process. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = (level or os.environ.get("AUDIRE_LOG_LEVEL", "INFO")).upper()
    fmt = fmt or os.environ.get("AUDIRE_LOG_FORMAT", "json")

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=getattr(logging, level, 20))

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    processors.append(
        structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
        if fmt == "console"
        else structlog.processors.JSONRenderer(ensure_ascii=False)
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level, 20)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structured logger, configuring logging on first use."""
    configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]
