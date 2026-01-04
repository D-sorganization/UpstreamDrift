import json
import logging
import os
import random
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np


class JSONFormatter(logging.Formatter):
    """Formatter to output logs in JSON format for structured observability."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
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
            log_data.update(record.props)  # type: ignore

        # Include exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Respects the LOG_LEVEL environment variable (default: INFO).
    respects LOG_FORMAT environment variable (default: TEXT, can be JSON).

    Args:
        name: The name for the logger (usually __name__)

    Returns:
        A configured logger instance
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
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)

    return logger_instance


logger = get_logger(__name__)


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across numpy and random modules.

    Args:
        seed: Integer seed value for deterministic random number generation

    Returns:
        None

    Note:
        Sets seeds for both numpy.random and Python's random module to ensure
        complete reproducibility in scientific computations.
    """
    if not (0 <= seed <= np.iinfo(np.uint32).max):
        raise ValueError(f"Seed must be between 0 and {np.iinfo(np.uint32).max}")
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Seeds set to %d", seed, extra={"props": {"seed": seed}})


@contextmanager
def log_execution_time(
    operation_name: str, logger_obj: logging.Logger | None = None
) -> Generator[None, None, None]:
    """Context manager to log the duration of an operation.

    Args:
        operation_name: logical name of the operation
        logger_obj: specific logger to use, or default logger if None
    """
    logr = logger_obj or logger
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
