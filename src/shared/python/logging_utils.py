"""Centralized logging utilities for the Golf Modeling Suite.

This module provides:
- Standardized logging setup with environment variable support
- JSON formatter for structured logging
- Seed management for reproducible computations
- Performance timing utilities

Refactored from duplicate logger_utils.py files to follow DRY principle.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

# Check for optional dependencies
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore[assignment]

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

# Default constants
DEFAULT_SEED: int = 42
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL: int = logging.INFO


class JSONFormatter(logging.Formatter):
    """Formatter to output logs in JSON format for structured observability."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log string.
        """
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": record.filename,
            "line": record.lineno,
        }

        # Include structured data from the 'extra' dict if present
        if hasattr(record, "props"):
            log_data.update(record.props)  # type: ignore[attr-defined]

        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(
    level: int | None = None,
    format_string: str | None = None,
    use_json: bool | None = None,
) -> None:
    """Set up logging configuration for the application.

    Respects environment variables:
    - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - LOG_FORMAT: Output format (TEXT or JSON)

    Args:
        level: Logging level. If None, uses LOG_LEVEL env var or INFO.
        format_string: Log message format string. If None, uses default.
        use_json: If True, use JSON formatter. If None, checks LOG_FORMAT env var.
    """
    # Determine log level
    if level is None:
        log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, log_level_str, logging.INFO)

    # Determine format
    if format_string is None:
        format_string = LOG_FORMAT

    # Determine if JSON format
    if use_json is None:
        use_json = os.environ.get("LOG_FORMAT", "TEXT").upper() == "JSON"

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    if use_json:
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(format_string)

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Respects environment variables:
    - LOG_LEVEL: Logging level (default: INFO)
    - LOG_FORMAT: Output format (default: TEXT, can be JSON)

    Args:
        name: The name for the logger (typically __name__).

    Returns:
        A configured logger instance.
    """
    logger_instance = logging.getLogger(name)

    # Check environment variable for log level
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level_str, logging.INFO)
    logger_instance.setLevel(level)

    # Only add handler if it doesn't already have one and root logger has none
    if not logger_instance.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()

        log_format = os.environ.get("LOG_FORMAT", "TEXT").upper()
        if log_format == "JSON":
            formatter: logging.Formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(LOG_FORMAT)

        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)

    return logger_instance


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducible computations.

    Sets seeds for Python's random module, NumPy's random generator,
    and PyTorch if available.

    Args:
        seed: Random seed value (default: 42).

    Raises:
        ValueError: If seed is out of valid range for numpy.
    """
    # Validate seed range if numpy available
    if NUMPY_AVAILABLE and np is not None:
        if not (0 <= seed <= np.iinfo(np.uint32).max):
            raise ValueError(
                f"Seed must be between 0 and {np.iinfo(np.uint32).max}"
            ) from None

    # Set Python random seed
    random.seed(seed)

    # Set NumPy seed if available
    if NUMPY_AVAILABLE and np is not None:
        np.random.seed(seed)
        # Also use the new Generator API
        np.random.default_rng(seed)

    # Set PyTorch seeds if PyTorch is available
    if TORCH_AVAILABLE and torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)


@contextmanager
def log_execution_time(
    operation_name: str, logger_obj: logging.Logger | None = None
) -> Generator[None, None, None]:
    """Context manager to log the duration of an operation.

    Args:
        operation_name: Logical name of the operation.
        logger_obj: Specific logger to use. If None, uses module logger.

    Yields:
        None - Use as context manager.

    Example:
        >>> with log_execution_time("data_processing"):
        ...     process_data()
    """
    logr = logger_obj or logging.getLogger(__name__)
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        # Log with structured 'props' for JSON formatter usage
        logr.info(
            "Telemetry: %s took %.4f seconds",
            operation_name,
            duration,
            extra={
                "props": {
                    "event": "performance_metric",
                    "operation": operation_name,
                    "duration_seconds": duration,
                }
            },
        )
