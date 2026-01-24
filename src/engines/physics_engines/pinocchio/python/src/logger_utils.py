"""Logging utilities for Pinocchio physics engine.

This module re-exports the shared logging utilities for backward compatibility.
All logging functionality has been consolidated in src/shared/python/logging_utils.py.

Refactored to use shared module (DRY principle).
"""

from src.shared.python.logging_utils import (
    DEFAULT_SEED,
    LOG_FORMAT,
    LOG_LEVEL,
    JSONFormatter,
    get_logger,
    log_execution_time,
    set_seeds,
    setup_logging,
)

__all__ = [
    "DEFAULT_SEED",
    "LOG_FORMAT",
    "LOG_LEVEL",
    "JSONFormatter",
    "get_logger",
    "log_execution_time",
    "set_seeds",
    "setup_logging",
]

# Module-level logger for backward compatibility
logger = get_logger(__name__)
