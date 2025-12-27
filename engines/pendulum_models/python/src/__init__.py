"""Scientific computing package for trajectory analysis and simulation.

This package provides utilities for scientific computing with a focus on
reproducibility, proper documentation, and adherence to quality standards.
"""

__version__ = "1.0.0"
__author__ = "Scientific Computing Team"
__email__ = "team@example.com"


# Export commonly used functions and constants
from shared.python.constants import (
    DEFAULT_RANDOM_SEED,
    GOLF_BALL_DIAMETER_M,
    GOLF_BALL_MASS_KG,
    GRAVITY_M_S2,
    PI,
    E,
)
from .logger_utils import get_logger, set_seeds, setup_logging

__all__ = [
    "DEFAULT_RANDOM_SEED",
    "GOLF_BALL_DIAMETER_M",
    "GOLF_BALL_MASS_KG",
    "GRAVITY_M_S2",
    "PI",
    "E",
    "__author__",
    "__email__",
    "__version__",
    "get_logger",
    "set_seeds",
    "setup_logging",
]
