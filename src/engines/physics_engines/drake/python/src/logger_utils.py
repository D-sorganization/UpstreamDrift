"""Logging utilities for scientific computing with reproducibility.

This module provides logging setup and seed management for deterministic
scientific computations.

Note: Logging setup now delegates to the centralized logging_config module.
"""

from src.shared.python.logging_config import get_logger
import random

from src.shared.python.engine_availability import PYTORCH_AVAILABLE
from src.shared.python.logging_config import (
    DEFAULT_LOG_FORMAT,
)
from src.shared.python.logging_config import (
    get_logger as _get_logger,
)
from src.shared.python.logging_config import (
    setup_logging as _setup_logging,
)

if PYTORCH_AVAILABLE:
    import torch

# Reproducibility constants
DEFAULT_SEED: int = 42  # Answer to everything
LOG_FORMAT: str = DEFAULT_LOG_FORMAT
LOG_LEVEL: int = logging.INFO

logger = get_logger(__name__)


def setup_logging(level: int = LOG_LEVEL, format_string: str = LOG_FORMAT) -> None:
    """Set up logging configuration for the application.

    This function delegates to the centralized logging_config module.

    Args:
        level: Logging level (default: INFO)
        format_string: Log message format string

    """
    _setup_logging(level=level, format_string=format_string)
    logger.info("Logging configured with level %s", logging.getLevelName(level))


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set random seeds for reproducible computations.

    Sets seeds for Python's random module, NumPy's random generator,
    and PyTorch if available.

    Args:
        seed: Random seed value (default: 42)

    """
    random.seed(seed)

    # Import numpy only when needed
    import numpy as np

    np.random.default_rng(seed)

    # Set PyTorch seeds if PyTorch is available
    if PYTORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
        logger.info("PyTorch seeds set: %d", seed)

    logger.info("All random seeds set to: %d", seed)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Delegates to the centralized logging_config module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    """
    return _get_logger(name)
