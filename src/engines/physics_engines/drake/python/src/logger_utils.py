"""Logging utilities for Drake physics engine.

Re-exports from the centralized logger_utils module,
following DRY principles from The Pragmatic Programmer.
"""

from __future__ import annotations

# Re-export from centralized module (includes fallback handling)
from src.shared.python.logger_utils import (
    DEFAULT_SEED,
    LOG_FORMAT,
    LOG_LEVEL,
    get_logger,
    log_execution_time,
    set_seeds,
    setup_logging,
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
