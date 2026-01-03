import logging
import os
import random
import time
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Respects the LOG_LEVEL environment variable (default: INFO).

    Args:
        name: The name for the logger (usually __name__)

    Returns:
        A configured logger instance
    """
    logger_instance = logging.getLogger(name)

    # Check environment variable for log level
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level_str, logging.INFO)
    logger_instance.setLevel(level)

    # Only add handler if it doesn't already have one and root logger has none
    if not logger_instance.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)

    return logger_instance


logger = get_logger(__name__)


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across numpy and random modules.

    Args:
        seed: Integer seed value for deterministic random number generation

    Returns:
        None

    Note:
        Sets seeds for both numpy.random and Python's random module to ensure
        complete reproducibility in scientific computations.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Seeds set: %d", seed)


@contextmanager
def log_execution_time(
    operation_name: str, logger_obj: logging.Logger | None = None
) -> Generator[None, None, None]:
    """Context manager to log the duration of an operation.

    Args:
        operation_name: logical name of the operation
        logger_obj: specific logger to use, or default logger if None
    """
    logr = logger_obj or logger
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        logr.info("Telemetry: %s took %.4f seconds", operation_name, duration)
