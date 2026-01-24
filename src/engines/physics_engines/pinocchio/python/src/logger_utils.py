"""Logging utilities - thin wrapper delegating to centralized modules.

This module follows DRY principles from The Pragmatic Programmer.
All functionality is delegated to src.shared.python modules.
"""

from __future__ import annotations

import logging

# Re-export from centralized logging module
from src.shared.python.logging_config import (
    DEFAULT_LOG_FORMAT as LOG_FORMAT,
)
from src.shared.python.logging_config import (
    get_logger,
    setup_logging,
)

# Re-export from centralized reproducibility module
from src.shared.python.reproducibility import (
    DEFAULT_SEED,
    log_execution_time,
    set_seeds,
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
