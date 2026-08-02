"""Small logging utility helpers with optional reproducibility support."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Generator
from contextlib import contextmanager

from .logging_config import DEFAULT_LOG_FORMAT as LOG_FORMAT
from .logging_config import get_logger, setup_logging

DEFAULT_SEED = 42
LOG_LEVEL = logging.INFO


def set_seeds(seed: int = DEFAULT_SEED, *, validate: bool = True) -> None:
    """Seed Python and NumPy random generators when available."""
    if validate and seed < 0:
        raise ValueError("expected non-negative integer for seed")
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)


@contextmanager
def log_execution_time(
    operation_name: str,
    logger_obj: logging.Logger | None = None,
) -> Generator[None, None, None]:
    """Log elapsed wall time for a named operation."""
    if not operation_name:
        raise ValueError("operation_name must be a non-empty string")
    logger = logger_obj or get_logger(__name__)
    start_time = time.perf_counter()
    try:
        yield
    finally:
        logger.info(
            "Telemetry: %s took %.4f seconds",
            operation_name,
            time.perf_counter() - start_time,
        )


__all__ = [
    "DEFAULT_SEED",
    "LOG_FORMAT",
    "LOG_LEVEL",
    "get_logger",
    "log_execution_time",
    "set_seeds",
    "setup_logging",
]
