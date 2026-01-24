"""Logging utilities - thin wrapper delegating to centralized modules.

This module follows DRY principles from The Pragmatic Programmer.
All functionality is delegated to src.shared.python modules when available.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeVar

    F = TypeVar("F", bound=Callable[..., object])

# Try to import from centralized modules, with fallback for standalone use
try:
    from src.shared.python.logging_config import (
        DEFAULT_LOG_FORMAT as LOG_FORMAT,
    )
    from src.shared.python.logging_config import (
        get_logger,
        setup_logging,
    )
    from src.shared.python.reproducibility import (
        DEFAULT_SEED,
        log_execution_time,
        set_seeds,
    )
except ImportError:
    # Fallback implementations for standalone use
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

    def setup_logging(level: int = logging.INFO) -> None:  # type: ignore[misc]
        """Setup basic logging configuration."""
        logging.basicConfig(level=level, format=LOG_FORMAT)

    def set_seeds(seed: int = DEFAULT_SEED) -> None:  # type: ignore[misc]
        """Set random seeds for reproducibility."""
        if seed < 0:
            raise ValueError("expected non-negative integer for seed")
        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass

    def log_execution_time(func: F) -> F:  # type: ignore[misc]
        """Decorator to log function execution time."""
        import functools
        import time

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            get_logger(__name__).info(f"{func.__name__} took {elapsed:.3f}s")
            return result

        return wrapper  # type: ignore[return-value]


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
