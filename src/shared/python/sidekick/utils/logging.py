"""Logging utilities for upstream_drift_tools.

Fully delegates to the canonical implementation in utils.logging_utils.
This module exists for backward-compatibility; new code should import
directly from utils.logging_utils.
"""

from __future__ import annotations

# Re-export all public symbols from the canonical module
from utils.logging_utils import (
    DEFAULT_FORMAT,
    DEFAULT_SEED,
    SIMPLE_FORMAT,
    JsonFormatter,
    LogExecutionTime,
    get_logger,
    init_default_logging,
    log_execution_time,
    set_seeds,
    setup_logging,
)

__all__ = [
    "DEFAULT_FORMAT",
    "DEFAULT_SEED",
    "JsonFormatter",
    "LogExecutionTime",
    "SIMPLE_FORMAT",
    "get_logger",
    "init_default_logging",
    "log_execution_time",
    "set_seeds",
    "setup_logging",
]
