"""Centralized logging configuration for the Golf Modeling Suite.

This module consolidates the common pattern of logging.basicConfig calls
across the codebase, addressing DRY violations identified in Pragmatic
Programmer reviews.

OBS-001: Enhanced with structlog integration, log rotation, and sensitive
data redaction for production-ready observability.

Usage:
    from src.shared.python.logging_pkg.logging_config import (
        setup_logging,
        get_logger,
        LogLevel,
    )

    # Basic setup with default configuration
    setup_logging()

    # Setup with custom level
    setup_logging(level=LogLevel.DEBUG)

    # Get a named logger (returns structlog BoundLogger when available)
    logger = get_logger(__name__)
    logger.info("Starting application...")

    # Setup for GUI applications (with Qt handler)
    setup_logging(use_qt_handler=True)

    # Add rotating file handler
    add_rotating_file_handler(filename="app.log")
"""

from __future__ import annotations

import logging
import re
import sys
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Try to import structlog; fall back gracefully so callers that only need
# stdlib logging continue to work without the extra dependency.
# ---------------------------------------------------------------------------
try:
    import structlog

    _STRUCTLOG_AVAILABLE = True
except ImportError:  # pragma: no cover
    _STRUCTLOG_AVAILABLE = False

# Standard log format used across the suite
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_LOG_FORMAT = "%(levelname)s: %(message)s"
DETAILED_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

# Rotation defaults
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5

# ---------------------------------------------------------------------------
# Sensitive-data redaction
# ---------------------------------------------------------------------------
_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?i)"
        r"("
        r"password|passwd|pwd"
        r"|api_key|apikey|api[-_]?secret"
        r"|secret_key|secret[-_]?token"
        r"|access_token|auth_token|bearer"
        r"|private_key"
        r")"
        r"[\s]*[=:]\s*['\"]?([^\s'\"]{1,})['\"]?"
    ),
]

_REDACTED = "***REDACTED***"


class SensitiveDataFilter(logging.Filter):
    """Logging filter that redacts sensitive data from log records.

    Scans the formatted message for patterns that look like credentials
    (passwords, API keys, tokens, etc.) and replaces the values with a
    redaction placeholder.

    Example:
        >>> handler = logging.StreamHandler()
        >>> handler.addFilter(SensitiveDataFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            # Format the message first so we can redact the result
            record.msg = str(record.msg) % record.args
            record.args = None
        record.msg = _redact_sensitive(str(record.msg))
        return True


def _redact_sensitive(text: str) -> str:
    """Replace sensitive values in *text* with a redaction placeholder."""
    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub(r"\1=***REDACTED***", text)
    return text


# ---------------------------------------------------------------------------
# Structlog redaction processor
# ---------------------------------------------------------------------------
def _structlog_redact_sensitive(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that redacts sensitive key-value pairs."""
    sensitive_keys = {
        "password",
        "passwd",
        "pwd",
        "api_key",
        "apikey",
        "api_secret",
        "secret_key",
        "secret_token",
        "access_token",
        "auth_token",
        "bearer",
        "private_key",
    }
    for key in list(event_dict.keys()):
        if key.lower() in sensitive_keys:
            event_dict[key] = _REDACTED
    # Also redact inside the event message string
    if "event" in event_dict and isinstance(event_dict["event"], str):
        event_dict["event"] = _redact_sensitive(event_dict["event"])
    return event_dict


# ---------------------------------------------------------------------------
# LogLevel enum
# ---------------------------------------------------------------------------
class LogLevel(Enum):
    """Log level enumeration for type-safe configuration."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# ---------------------------------------------------------------------------
# Global structlog configuration state (mutable holder avoids 'global')
# ---------------------------------------------------------------------------
_structlog_state: dict[str, bool] = {"configured": False}


def _configure_structlog(
    level: int,
    json_output: bool,
    dev_mode: bool,
) -> None:
    """Wire up structlog processors and configure the library."""
    if not _STRUCTLOG_AVAILABLE or _structlog_state["configured"]:
        return

    processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _structlog_redact_sensitive,
        structlog.processors.CallsiteParameterAdder(
            parameters={
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        structlog.processors.format_exc_info,
        structlog.processors.StackInfoRenderer(),
    ]

    # Output renderer
    if dev_mode and not json_output:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )
    elif json_output:
        processors.extend(
            [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ]
        )
    else:
        processors.append(structlog.processors.KeyValueRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _structlog_state["configured"] = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _resolve_log_level(level: LogLevel | int) -> int:
    """Convert a LogLevel enum or int to an integer log level.

    Args:
        level: The logging level as LogLevel enum or raw int.

    Returns:
        Integer log level suitable for stdlib logging.
    """
    return level.value if isinstance(level, LogLevel) else level


def _resolve_format_string(
    format_string: str | None,
    use_detailed_format: bool,
    use_simple_format: bool,
) -> str:
    """Determine the log format string from the configuration flags.

    Args:
        format_string: Explicit format string (takes precedence).
        use_detailed_format: If True, use file/line detailed format.
        use_simple_format: If True, use level-only simple format.

    Returns:
        The resolved format string.
    """
    if format_string:
        return format_string
    if use_detailed_format:
        return DETAILED_LOG_FORMAT
    if use_simple_format:
        return SIMPLE_LOG_FORMAT
    return DEFAULT_LOG_FORMAT


def _build_basic_config_kwargs(
    log_level: int,
    fmt: str,
    stream: TextIO | None,
    filename: str | Path | None,
    filemode: str,
    datefmt: str | None,
    force: bool,
) -> dict:
    """Build keyword arguments for ``logging.basicConfig``.

    Args:
        log_level: Resolved integer log level.
        fmt: Resolved format string.
        stream: Output stream (default: sys.stderr when no filename).
        filename: Optional log file path.
        filemode: File open mode ('a' or 'w').
        datefmt: Optional date format string.
        force: Whether to force re-configuration.

    Returns:
        Dictionary of kwargs ready for ``logging.basicConfig``.
    """
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

    return config_kwargs


def _attach_redaction_filters(root_logger: logging.Logger) -> None:
    """Attach a :class:`SensitiveDataFilter` to all handlers on *root_logger*.

    Skips handlers that already have the filter to avoid duplicates.
    """
    redaction_filter = SensitiveDataFilter()
    for handler in root_logger.handlers:
        if not any(isinstance(f, SensitiveDataFilter) for f in handler.filters):
            handler.addFilter(redaction_filter)


def _quiet_noisy_libraries(
    quiet_libraries: Sequence[str] | None,
    use_qt_handler: bool,
) -> None:
    """Set noisy third-party library loggers to WARNING level.

    Args:
        quiet_libraries: Explicit list of library names to quiet.
        use_qt_handler: When True, automatically quiets matplotlib/PIL.
    """
    default_quiet: list[str] = []
    if use_qt_handler:
        default_quiet.extend(["matplotlib", "matplotlib.font_manager", "PIL"])

    libraries_to_quiet = list(quiet_libraries or []) + default_quiet
    for lib_name in libraries_to_quiet:
        logging.getLogger(lib_name).setLevel(logging.WARNING)


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
    json_output: bool = False,
    dev_mode: bool = True,
    enable_structlog: bool = True,
    enable_redaction: bool = True,
) -> logging.Logger:
    """Setup logging configuration for the Golf Modeling Suite.

    This function provides a standardized way to configure logging across
    all modules in the suite.  When *structlog* is available and
    ``enable_structlog`` is ``True`` (the default) it will also wire up
    structlog processors for structured, machine-parseable logging.

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
        json_output: If True and structlog is enabled, render JSON output.
        dev_mode: If True, enable development-friendly console rendering.
        enable_structlog: If True (default), configure structlog processors.
        enable_redaction: If True (default), attach SensitiveDataFilter to
            all new handlers so credentials are never written to logs.

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

        # Production JSON mode
        setup_logging(json_output=True, dev_mode=False)
    """
    log_level = _resolve_log_level(level)
    fmt = _resolve_format_string(format_string, use_detailed_format, use_simple_format)

    config_kwargs = _build_basic_config_kwargs(
        log_level,
        fmt,
        stream,
        filename,
        filemode,
        datefmt,
        force,
    )
    logging.basicConfig(**config_kwargs)

    root_logger = logging.getLogger()

    if enable_redaction:
        _attach_redaction_filters(root_logger)

    _quiet_noisy_libraries(quiet_libraries, use_qt_handler)

    if enable_structlog and _STRUCTLOG_AVAILABLE:
        _configure_structlog(log_level, json_output, dev_mode)

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
        from src.shared.python.logging_pkg.logging_config import configure_test_logging
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
        from src.shared.python.logging_pkg.logging_config import configure_gui_logging
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
    handler.addFilter(SensitiveDataFilter())
    logger.addHandler(handler)

    return handler


def add_rotating_file_handler(
    logger: logging.Logger | None = None,
    filename: str | Path = "golf_suite.log",
    level: LogLevel | int = LogLevel.DEBUG,
    format_string: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    enable_redaction: bool = True,
) -> RotatingFileHandler:
    """Add a size-based rotating file handler to a logger.

    Logs are rotated when they reach *max_bytes* in size; up to
    *backup_count* old log files are kept (e.g. ``app.log.1``,
    ``app.log.2``, ...).

    Args:
        logger: Logger to add handler to (default: root logger).
        filename: Log file path.
        level: Logging level for the file handler.
        format_string: Format string for the handler.
        max_bytes: Maximum size in bytes before rotation (default 10 MB).
        backup_count: Number of rotated files to keep (default 5).
        enable_redaction: Attach SensitiveDataFilter (default True).

    Returns:
        The created RotatingFileHandler.

    Example:
        add_rotating_file_handler(filename="app.log")
        add_rotating_file_handler(
            filename="debug.log",
            max_bytes=5 * 1024 * 1024,
            backup_count=3,
        )
    """
    if logger is None:
        logger = logging.getLogger()

    log_level = level.value if isinstance(level, LogLevel) else level
    fmt = format_string or DETAILED_LOG_FORMAT

    handler = RotatingFileHandler(
        str(filename),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(fmt))

    if enable_redaction:
        handler.addFilter(SensitiveDataFilter())

    logger.addHandler(handler)
    return handler
