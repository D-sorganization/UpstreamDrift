"""Logging utilities for scientific computing with reproducibility.

This module provides logging setup and seed management for deterministic
scientific computations.

Note: Logging setup now delegates to the centralized logging_config module.
"""

import logging
import random
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np

from src.shared.python.logging_config import (
    DEFAULT_LOG_FORMAT,
)
from src.shared.python.logging_config import (
    get_logger as _get_logger,
)
from src.shared.python.logging_config import (
    setup_logging as _setup_logging,
)

# Reproducibility constants
DEFAULT_SEED: int = 42  # Answer to everything
LOG_FORMAT: str = DEFAULT_LOG_FORMAT
LOG_LEVEL: int = logging.INFO

logger = _get_logger(__name__)


def setup_logging(level: int = LOG_LEVEL, format_string: str = LOG_FORMAT) -> None:
    """Set up logging configuration for the application.

    This function delegates to the centralized logging_config module.

    Args:
        level: Logging level (default: INFO)
        format_string: Log message format string
    """
    _setup_logging(level=level, format_string=format_string)
    logger.info("Logging configured with level %s", logging.getLevelName(level))


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Delegates to the centralized logging_config module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return _get_logger(name)


def set_seeds(seed: int = DEFAULT_SEED) -> None:
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
    logger.info("Seeds set to %d", seed)


def log_execution_time(
    operation_name: str, logger_obj: logging.Logger | None = None
) -> Generator[None, None, None]:
    """Context manager to log the duration of an operation.

    Args:
        operation_name: logical name of the operation
        logger_obj: specific logger to use, or default logger if None

    Yields:
        None
    """
    import time

    logr = logger_obj or logger
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        logr.info(
            "Telemetry: %s took %.4f seconds",
            operation_name,
            duration,
        )


# Apply decorator for proper context manager behavior
log_execution_time = contextmanager(log_execution_time)
