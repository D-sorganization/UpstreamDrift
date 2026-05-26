"""Core lightweight utilities for the Golf Modeling Suite.

This module contains base exceptions and logging setup that do not require
heavy dependencies like numpy, pandas, or matplotlib.

Logging strategy: stdlib ``logging`` is the canonical logging mechanism for
this codebase.  Structured logging configuration (including optional structlog
processors) lives in ``logging_pkg.logging_config`` and is applied at
application start-up via ``setup_logging()`` / ``setup_structured_logging()``.
Callers should obtain loggers with ``get_logger(__name__)`` rather than calling
``logging.getLogger(__name__)`` directly so that any future pipeline changes
propagate automatically.

See GitHub issue #2061 for the rationale behind removing the per-module
structlog configuration that previously lived here.
"""

import logging
import sys

from .exceptions import DataFormatError, EngineNotFoundError, GolfModelingError

__all__ = [
    "GolfModelingError",
    "EngineNotFoundError",
    "DataFormatError",
    "setup_logging",
    "setup_structured_logging",
    "get_logger",
]


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up consistent logging across all engines.

    Args:
        name: Logger name (typically __name__)
        level: Logging level

    Returns:
        Configured logger instance
    """
    if not (name is not None):
        raise ValueError("name must be provided")
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


def setup_structured_logging(
    level: int = logging.INFO,
    json_output: bool = False,
    dev_mode: bool = True,
) -> None:
    """Configure logging for the Golf Modeling Suite.

    Delegates to the canonical ``logging_pkg.logging_config.setup_logging``
    so that all configuration (including optional structlog processors and
    sensitive-data redaction) is applied in one place.

    Args:
        level: Minimum log level (default: logging.INFO)
        json_output: If True, request JSON output (passed through to
            ``logging_config.setup_logging`` when structlog is available).
        dev_mode: If True, enable development-friendly features.
    """
    try:
        from src.shared.python.logging_pkg.logging_config import (
            setup_logging as _canonical_setup,
        )

        _canonical_setup(
            level=level,
            json_output=json_output,
            dev_mode=dev_mode,
        )
    except ImportError:
        # Fallback when running in a minimal environment without the full package
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stdout,
            level=level,
        )


def get_logger(name: str) -> logging.Logger:
    """Get a stdlib logger instance.

    This is the standard way to obtain a logger throughout the codebase.
    All modules should use this function rather than calling
    ``logging.getLogger()`` directly, so that any future changes to the
    logging pipeline (e.g. adding handlers or processors) can be made in
    one place.

    Args:
        name: Logger name (typically ``__name__``)

    Returns:
        :class:`logging.Logger` instance.

    Example:
        >>> from src.shared.python.core import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("simulation_started")
    """
    from src.shared.python.logging_pkg.logging_config import get_logger as _get_logger

    return _get_logger(name)
