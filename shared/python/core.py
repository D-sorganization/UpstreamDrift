"""Core lightweight utilities for the Golf Modeling Suite.

This module contains base exceptions and logging setup that do not require
heavy dependencies like numpy, pandas, or matplotlib.
"""

import logging
import sys


class GolfModelingError(Exception):
    """Base exception for golf modeling suite."""

    pass


class EngineNotFoundError(GolfModelingError):
    """Raised when a physics engine is not found or not properly installed."""

    pass


class DataFormatError(GolfModelingError):
    """Raised when data format is invalid or unsupported."""

    pass


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up consistent logging across all engines.

    Args:
        name: Logger name (typically __name__)
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
