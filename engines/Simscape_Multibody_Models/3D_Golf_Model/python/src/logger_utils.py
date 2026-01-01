import logging
import random

import numpy as np


import os

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
