"""Constants module - re-exports from centralized constants.

This module follows DRY principles from The Pragmatic Programmer.
All constants are imported from the central src.shared.python.constants module.
"""

from src.shared.python.constants import (
    GOLF_BALL_DIAMETER_M,
    GOLF_BALL_MASS_KG,
    GRAVITY_M_S2,
    PI,
    SPEED_OF_LIGHT_M_S,
    E,
)
from src.shared.python.reproducibility import DEFAULT_SEED as DEFAULT_RANDOM_SEED

__all__ = [
    "DEFAULT_RANDOM_SEED",
    "E",
    "GOLF_BALL_DIAMETER_M",
    "GOLF_BALL_MASS_KG",
    "GRAVITY_M_S2",
    "PI",
    "SPEED_OF_LIGHT_M_S",
]
