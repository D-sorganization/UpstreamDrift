"""Centralized logging configuration for shared Tools modules."""

from __future__ import annotations

import logging
import re
import sys
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TextIO

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_LOG_FORMAT = "%(levelname)s: %(message)s"
DETAILED_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 5

_SENSITIVE_PATTERNS = [
    re.compile(
        r"(?i)"
        r"(password|passwd|pwd|api_key|apikey|api[-_]?secret|secret_key|"
        r"secret[-_]?token|access_token|auth_token|bearer|private_key)"
        r"[\s]*[=:]\s*['\"]?([^\s'\"]{1,})['\"]?"
    ),
]


class LogLevel(Enum):
    """Type-safe logging levels used by shared modules."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class SensitiveDataFilter(logging.Filter):
    """Redact common credential-like values from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            record.msg = str(record.msg) % record.args
            record.args = None
        record.msg = _redact_sensitive(str(record.msg))
        return True


def _redact_sensitive(text: str) -> str:
    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub(r"\1=***REDACTED***", text)
    return text


def _resolve_log_level(level: LogLevel | int) -> int:
    return level.value if isinstance(level, LogLevel) else int(level)


def _resolve_format_string(
    format_string: str | None,
    use_detailed_format: bool,
    use_simple_format: bool,
) -> str:
    if format_string:
        return format_string
    if use_detailed_format:
        return DETAILED_LOG_FORMAT
    if use_simple_format:
        return SIMPLE_LOG_FORMAT
    return DEFAULT_LOG_FORMAT


def _attach_redaction_filters(root_logger: logging.Logger) -> None:
    redaction_filter = SensitiveDataFilter()
    for handler in root_logger.handlers:
        if not any(isinstance(item, SensitiveDataFilter) for item in handler.filters):
            handler.addFilter(redaction_filter)


def _quiet_libraries(
    quiet_libraries: list[str] | tuple[str, ...] | None,
    use_qt_handler: bool,
) -> None:
    libraries = list(quiet_libraries or [])
    if use_qt_handler:
        libraries.extend(["matplotlib", "matplotlib.font_manager", "PIL"])
    for name in libraries:
        logging.getLogger(name).setLevel(logging.WARNING)


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
    quiet_libraries: list[str] | tuple[str, ...] | None = None,
    json_output: bool = False,
    dev_mode: bool = True,
    enable_structlog: bool = True,
    enable_redaction: bool = True,
) -> logging.Logger:
    """Configure stdlib logging with the shared API expected by consumers."""
    del json_output, dev_mode, enable_structlog
    log_level = _resolve_log_level(level)
    fmt = _resolve_format_string(format_string, use_detailed_format, use_simple_format)
    if filename is not None:
        logging.basicConfig(
            filename=str(filename),
            filemode=filemode,
            level=log_level,
            format=fmt,
            datefmt=datefmt,
            force=force,
        )
    else:
        logging.basicConfig(
            stream=stream or sys.stderr,
            level=log_level,
            format=fmt,
            datefmt=datefmt,
            force=force,
        )
    root_logger = logging.getLogger()
    if enable_redaction:
        _attach_redaction_filters(root_logger)
    _quiet_libraries(quiet_libraries, use_qt_handler)
    return root_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a stdlib logger by name."""
    return logging.getLogger(name)


def configure_test_logging(
    *,
    level: LogLevel | int = LogLevel.WARNING,
    capture_warnings: bool = True,
) -> logging.Logger:
    """Configure low-noise logging for tests."""
    if capture_warnings:
        logging.captureWarnings(True)
    return setup_logging(
        level=level,
        use_simple_format=True,
        force=True,
        quiet_libraries=["matplotlib", "PIL", "urllib3", "asyncio"],
    )


def configure_gui_logging(*, level: LogLevel | int = LogLevel.INFO) -> logging.Logger:
    """Configure logging defaults for Qt-facing tools."""
    return setup_logging(level=level, use_qt_handler=True)


def add_rotating_file_handler(
    logger: logging.Logger | None = None,
    filename: str | Path = "tools.log",
    level: LogLevel | int = LogLevel.DEBUG,
    format_string: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    enable_redaction: bool = True,
) -> RotatingFileHandler:
    """Attach a size-rotating file handler to a logger."""
    target = logger or logging.getLogger()
    handler = RotatingFileHandler(
        str(filename),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    handler.setLevel(_resolve_log_level(level))
    handler.setFormatter(logging.Formatter(format_string or DETAILED_LOG_FORMAT))
    if enable_redaction:
        handler.addFilter(SensitiveDataFilter())
    target.addHandler(handler)
    return handler


def add_file_handler(
    logger: logging.Logger | None = None,
    filename: str | Path = "tools.log",
    level: LogLevel | int = LogLevel.DEBUG,
    format_string: str | None = None,
) -> logging.FileHandler:
    """Attach a plain file handler to a logger."""
    target = logger or logging.getLogger()
    handler = logging.FileHandler(str(filename))
    handler.setLevel(_resolve_log_level(level))
    handler.setFormatter(logging.Formatter(format_string or DETAILED_LOG_FORMAT))
    handler.addFilter(SensitiveDataFilter())
    target.addHandler(handler)
    return handler
