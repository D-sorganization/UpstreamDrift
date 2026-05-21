"""Logging utilities for sidekick."""

from __future__ import annotations

import logging
import sys

# Default log format. Previously imported from a top-level ``utils.logging_utils``
# module that does not exist in this repo, which raised ``ModuleNotFoundError``
# in CI contexts where ``utils`` was not on ``sys.path``. Inlined here to make
# the module self-contained.
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(
    name: str, level: int = logging.INFO, log_file: str | None = None
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Name of the logger
        level: Logging level (default: logging.INFO)
        log_file: Optional path to a log file

    Returns:
        Configured logger instance
    """
    if name is None:
        raise ValueError("name must be provided")
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(DEFAULT_FORMAT)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


class LogExecutionTime:
    """Context manager to log execution time of a block."""

    def __init__(self, name: str, logger: logging.Logger | None = None) -> None:
        self.name = name
        self.logger = logger or get_logger(name)

    def __enter__(self) -> LogExecutionTime:
        self.start_time = __import__("time").time()
        self.logger.debug(f"Starting {self.name}...")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        duration = __import__("time").time() - self.start_time
        self.logger.info(f"{self.name} completed in {duration:.4f}s")


def log_execution_time(name: str) -> LogExecutionTime:
    return LogExecutionTime(name)
