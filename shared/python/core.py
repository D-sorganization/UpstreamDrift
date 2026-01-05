"""Core lightweight utilities for the Golf Modeling Suite.

This module contains base exceptions and logging setup that do not require
heavy dependencies like numpy, pandas, or matplotlib.

OBS-001: Added structured logging with structlog for production-ready observability.
"""

import logging
import sys
from typing import Any, cast

import structlog

from .exceptions import DataFormatError, EngineNotFoundError, GolfModelingError

__all__ = [
    "GolfModelingError",
    "EngineNotFoundError",
    "DataFormatError",
    "setup_logging",
    "setup_structured_logging",
    "get_logger",
]


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up consistent logging across all engines (legacy).

    Args:
        name: Logger name (typically __name__)
        level: Logging level

    Returns:
        Configured logger instance

    Note:
        This is the legacy logging setup. For new code, consider using
        setup_structured_logging() and get_logger() for better observability.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Global flag to track if structured logging has been configured
_structured_logging_configured = False


def setup_structured_logging(
    level: int = logging.INFO,
    json_output: bool = False,
    dev_mode: bool = True,
) -> None:
    """Configure structured logging with structlog.

    OBS-001: Production-ready structured logging configuration.

    This function configures structlog with:
    - Timestamp processing
    - Log level filtering
    - Stack info extraction
    - Exception formatting
    - Optional JSON output for production
    - Console-friendly output for development

    Args:
        level: Minimum log level (default: logging.INFO)
        json_output: If True, output JSON; if False, use human-readable format
        dev_mode: If True, enable development-friendly features (colors, pretty printing)

    Example:
        >>> from shared.python.core import setup_structured_logging, get_logger
        >>> setup_structured_logging(level=logging.DEBUG, dev_mode=True)
        >>> logger = get_logger(__name__)
        >>> logger.info("simulation_started", engine="mujoco", duration=2.5)
    """
    global _structured_logging_configured
    import threading

    # Thread-safe initialization lock
    _logging_lock = threading.Lock()

    if _structured_logging_configured:
        # Fast path check
        return

    with _logging_lock:
        if _structured_logging_configured:
            # Double check inside lock
            return

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    # Build processor chain
    processors: list[Any] = [
        # Add log level to event dict
        structlog.stdlib.add_log_level,
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),
        # Add caller information (file, function, line)
        structlog.processors.CallsiteParameterAdder(
            parameters={
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        # Format exceptions
        structlog.processors.format_exc_info,
        # Stack info extraction
        structlog.processors.StackInfoRenderer(),
    ]

    # Add dev-mode or production processors
    if dev_mode and not json_output:
        # Development mode: human-readable console output with colors
        processors.extend(
            [
                structlog.dev.ConsoleRenderer(
                    colors=True,
                    exception_formatter=structlog.dev.plain_traceback,
                )
            ]
        )
    elif json_output:
        # Production mode: JSON output for log aggregation systems
        processors.extend(
            [
                # Ensure all values are JSON-serializable
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ]
        )
    else:
        # Fallback: key-value pairs
        processors.append(structlog.processors.KeyValueRenderer())

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _structured_logging_configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance.

    OBS-001: Structured logger with rich contextual information.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Structured logger that supports:
        - Keyword arguments for structured data
        - Method chaining with bind()
        - Automatic context propagation

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("processing_file", filename="data.csv", size_mb=12.5)
        >>> # Bind persistent context
        >>> request_logger = logger.bind(request_id="abc-123")
        >>> request_logger.info("request_started")
        >>> request_logger.info("request_completed", duration_ms=250)
    """
    # Ensure structured logging is configured with defaults
    if not _structured_logging_configured:
        setup_structured_logging()

    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
