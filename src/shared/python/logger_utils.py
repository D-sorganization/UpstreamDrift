"""Logging utilities with fallback support for standalone usage.

This module provides logging utilities that work both when used as part of
the full Golf Modeling Suite and when engines are run in standalone mode.

Follows DRY principles from The Pragmatic Programmer - all fallback logic
is centralized here instead of being duplicated in each engine module.

Usage:
    from src.shared.python.logger_utils import (
        get_logger,
        setup_logging,
        set_seeds,
        log_execution_time,
        DEFAULT_SEED,
        LOG_FORMAT,
    )
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Generator
from contextlib import contextmanager

# Try to import from full implementations, with fallback for standalone use
_USING_FULL_IMPLEMENTATION = False

try:
    from src.shared.python.logging_config import DEFAULT_LOG_FORMAT as LOG_FORMAT
    from src.shared.python.logging_config import get_logger as _get_logger
    from src.shared.python.logging_config import setup_logging as _setup_logging
    from src.shared.python.reproducibility import (
        DEFAULT_SEED,
        log_execution_time,  # Context manager - re-export directly
    )
    from src.shared.python.reproducibility import set_seeds as _set_seeds

    _USING_FULL_IMPLEMENTATION = True

    # Re-export from full implementations
    get_logger = _get_logger
    set_seeds = _set_seeds

    def setup_logging(level: int = logging.INFO) -> None:
        """Setup logging configuration."""
        _setup_logging(level=level)

except ImportError:
    # Fallback implementations for standalone use (no numpy/torch dependencies)
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DEFAULT_SEED = 42

    def get_logger(name: str | None = None) -> logging.Logger:
        """Get a logger instance with basic configuration."""
        logger = logging.getLogger(name or __name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(LOG_FORMAT))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def setup_logging(level: int = logging.INFO) -> None:
        """Setup basic logging configuration."""
        logging.basicConfig(level=level, format=LOG_FORMAT)

    def set_seeds(seed: int = DEFAULT_SEED, *, validate: bool = True) -> None:
        """Set random seeds for reproducibility."""
        if validate and seed < 0:
            raise ValueError("expected non-negative integer for seed")
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass

    @contextmanager
    def log_execution_time(
        operation_name: str,
        logger_obj: logging.Logger | None = None,
    ) -> Generator[None, None, None]:
        """Context manager to log the duration of an operation.

        Args:
            operation_name: Logical name of the operation being timed
            logger_obj: Specific logger to use, or module logger if None

        Yields:
            None
        """
        logr = logger_obj or get_logger(__name__)
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


# For backwards compatibility
LOG_LEVEL: int = logging.INFO

__all__ = [
    "DEFAULT_SEED",
    "LOG_FORMAT",
    "LOG_LEVEL",
    "get_logger",
    "log_execution_time",
    "set_seeds",
    "setup_logging",
]
