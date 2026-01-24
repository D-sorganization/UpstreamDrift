"""Centralized logging configuration for the Golf Modeling Suite.

This module consolidates the common pattern of logging.basicConfig calls
across the codebase, addressing DRY violations identified in Pragmatic
Programmer reviews.

Usage:
    from src.shared.python.logging_config import (
        setup_logging,
        get_logger,
        LogLevel,
    )

    # Basic setup with default configuration
    setup_logging()

    # Setup with custom level
    setup_logging(level=LogLevel.DEBUG)

    # Get a named logger
    logger = get_logger(__name__)
    logger.info("Starting application...")

    # Setup for GUI applications (with Qt handler)
    setup_logging(use_qt_handler=True)
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from collections.abc import Sequence

# Standard log format used across the suite
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_LOG_FORMAT = "%(levelname)s: %(message)s"
DETAILED_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(message)s"
)


class LogLevel(Enum):
    """Log level enumeration for type-safe configuration."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def setup_logging(
    *,
    level: LogLevel | int = LogLevel.INFO,
    format_string: str | None = None,
    stream: TextIO | None = None,
    filename: str | Path | None = None,
    filemode: str = "a",
    datefmt: str | None = None,
    force: bool = False,
    use_simple_format: bool = False,
    use_detailed_format: bool = False,
    use_qt_handler: bool = False,
    quiet_libraries: Sequence[str] | None = None,
) -> logging.Logger:
    """Setup logging configuration for the Golf Modeling Suite.

    This function provides a standardized way to configure logging across
    all modules in the suite.

    Args:
        level: The logging level (LogLevel enum or int).
        format_string: Custom format string. Overrides use_simple_format
            and use_detailed_format.
        stream: Stream to write logs to (default: sys.stderr).
        filename: If specified, logs will be written to this file.
        filemode: Mode for opening the log file ('a' for append, 'w' for write).
        datefmt: Date format string for the formatter.
        force: If True, remove and close existing handlers before setup.
        use_simple_format: Use simplified format (level: message).
        use_detailed_format: Use detailed format with file/line info.
        use_qt_handler: Setup for Qt applications (suppresses matplotlib).
        quiet_libraries: List of library names to set to WARNING level.

    Returns:
        The root logger instance.

    Example:
        # Basic setup
        setup_logging()

        # Debug mode with detailed format
        setup_logging(level=LogLevel.DEBUG, use_detailed_format=True)

        # Write to file
        setup_logging(filename="app.log")

        # Quiet noisy libraries
        setup_logging(quiet_libraries=["matplotlib", "PIL"])
    """
    # Determine log level
    log_level = level.value if isinstance(level, LogLevel) else level

    # Determine format string
    if format_string:
        fmt = format_string
    elif use_detailed_format:
        fmt = DETAILED_LOG_FORMAT
    elif use_simple_format:
        fmt = SIMPLE_LOG_FORMAT
    else:
        fmt = DEFAULT_LOG_FORMAT

    # Build basicConfig kwargs
    config_kwargs: dict = {
        "level": log_level,
        "format": fmt,
    }

    if stream is not None:
        config_kwargs["stream"] = stream
    elif filename is None:
        config_kwargs["stream"] = sys.stderr

    if filename is not None:
        config_kwargs["filename"] = str(filename)
        config_kwargs["filemode"] = filemode

    if datefmt is not None:
        config_kwargs["datefmt"] = datefmt

    if force:
        config_kwargs["force"] = True

    # Configure logging
    logging.basicConfig(**config_kwargs)

    # Get root logger
    root_logger = logging.getLogger()

    # Quiet noisy libraries
    default_quiet = []
    if use_qt_handler:
        default_quiet.extend(["matplotlib", "matplotlib.font_manager", "PIL"])

    libraries_to_quiet = list(quiet_libraries or []) + default_quiet
    for lib_name in libraries_to_quiet:
        logging.getLogger(lib_name).setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance with the given name.

    This is a convenience wrapper around logging.getLogger that ensures
    consistent logger naming across the suite.

    Args:
        name: Logger name (typically __name__). If None, returns root logger.

    Returns:
        Logger instance.

    Example:
        logger = get_logger(__name__)
        logger.info("Starting process...")
    """
    return logging.getLogger(name)


def configure_test_logging(
    *,
    level: LogLevel | int = LogLevel.WARNING,
    capture_warnings: bool = True,
) -> logging.Logger:
    """Configure logging for test environments.

    This provides sensible defaults for pytest and unittest:
    - WARNING level by default (less noise)
    - Captures Python warnings
    - Simple format

    Args:
        level: The logging level.
        capture_warnings: Whether to capture Python warnings in logging.

    Returns:
        The root logger instance.

    Example:
        # In conftest.py or test setup
        from src.shared.python.logging_config import configure_test_logging
        configure_test_logging()
    """
    if capture_warnings:
        logging.captureWarnings(True)

    return setup_logging(
        level=level,
        use_simple_format=True,
        force=True,
        quiet_libraries=["matplotlib", "PIL", "urllib3", "asyncio"],
    )


def configure_gui_logging(
    *,
    level: LogLevel | int = LogLevel.INFO,
) -> logging.Logger:
    """Configure logging for GUI applications.

    This provides defaults optimized for GUI apps:
    - INFO level
    - Quiets matplotlib and font manager logs
    - Standard format

    Args:
        level: The logging level.

    Returns:
        The root logger instance.

    Example:
        # In GUI application startup
        from src.shared.python.logging_config import configure_gui_logging
        configure_gui_logging()
    """
    return setup_logging(
        level=level,
        use_qt_handler=True,
    )


def add_file_handler(
    logger: logging.Logger | None = None,
    filename: str | Path = "golf_suite.log",
    level: LogLevel | int = LogLevel.DEBUG,
    format_string: str | None = None,
) -> logging.FileHandler:
    """Add a file handler to an existing logger.

    This allows adding file logging without reconfiguring the root logger.

    Args:
        logger: Logger to add handler to (default: root logger).
        filename: Log file path.
        level: Logging level for the file handler.
        format_string: Format string for the handler.

    Returns:
        The created FileHandler.

    Example:
        logger = get_logger(__name__)
        add_file_handler(logger, "debug.log", LogLevel.DEBUG)
    """
    if logger is None:
        logger = logging.getLogger()

    log_level = level.value if isinstance(level, LogLevel) else level
    fmt = format_string or DETAILED_LOG_FORMAT

    handler = logging.FileHandler(str(filename))
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    return handler
