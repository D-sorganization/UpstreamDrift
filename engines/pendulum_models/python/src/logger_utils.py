"""Logging utilities for scientific computing with reproducibility.

This module provides logging setup and seed management for deterministic
scientific computations.
"""

import logging
import random
import sys

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, OSError):
    TORCH_AVAILABLE = False

# Reproducibility constants
DEFAULT_SEED: int = 42  # Answer to everything
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL: int = logging.INFO

logger = logging.getLogger(__name__)


def setup_logging(level: int = LOG_LEVEL, format_string: str = LOG_FORMAT) -> None:
    """Set up logging configuration for the application.

    Args:
        level: Logging level (default: INFO)
        format_string: Log message format string

    """
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info("Logging configured with level %s", logging.getLevelName(level))


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducible computations.

    Sets seeds for Python's random module, NumPy's random generator,
    and PyTorch if available.

    Args:
        seed: Random seed value (default: 42)

    """
    random.seed(seed)

    # Use modern NumPy Generator API for better statistical properties
    # Store the generator for potential future use while maintaining compatibility
    _ = np.random.default_rng(seed)
    # Also set legacy global state for compatibility with older code
    np.random.seed(seed)

    # Set PyTorch seeds if PyTorch is available
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
        logger.info("PyTorch seeds set: %d", seed)

    logger.info("All random seeds set to: %d", seed)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    """
    return logging.getLogger(name)
