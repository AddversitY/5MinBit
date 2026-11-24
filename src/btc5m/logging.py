"""Structured logging (JSON lines to stderr) via structlog."""
from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def setup_logging(level: str = "INFO", json: bool = True) -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(stream=sys.stderr, level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(),
    ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str):
    setup_logging()
    return structlog.get_logger(name)
