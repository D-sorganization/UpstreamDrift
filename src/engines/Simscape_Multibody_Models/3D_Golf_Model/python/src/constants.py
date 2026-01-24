"""Constants module - re-exports from centralized constants.

This module follows DRY principles from The Pragmatic Programmer.
All constants are imported from the central src.shared.python.physics_constants module.
"""

from src.shared.python.physics_constants import (
    AIR_DENSITY_SEA_LEVEL_KG_M3,
    BUNKER_DEPTH_MM,
    DRIVER_LOFT_TYPICAL_DEG,
    GOLF_BALL_DIAMETER_M,
    GOLF_BALL_DRAG_COEFFICIENT,
    GOLF_BALL_MASS_KG,
    GRAVITY_M_S2,
    GREEN_SPEED_STIMP,
    HUMIDITY_PERCENT,
    IRON_7_LOFT_DEG,
    PI,
    PRESSURE_HPA,
    PUTTER_LOFT_DEG,
    ROUGH_HEIGHT_MM,
    SPEED_OF_LIGHT_M_S,
    TEMPERATURE_C,
    E,
)

# Alias for backwards compatibility
DRIVER_LOFT_DEG = DRIVER_LOFT_TYPICAL_DEG

__all__ = [
    "AIR_DENSITY_SEA_LEVEL_KG_M3",
    "BUNKER_DEPTH_MM",
    "DRIVER_LOFT_DEG",
    "E",
    "GOLF_BALL_DIAMETER_M",
    "GOLF_BALL_DRAG_COEFFICIENT",
    "GOLF_BALL_MASS_KG",
    "GRAVITY_M_S2",
    "GREEN_SPEED_STIMP",
    "HUMIDITY_PERCENT",
    "IRON_7_LOFT_DEG",
    "PI",
    "PRESSURE_HPA",
    "PUTTER_LOFT_DEG",
    "ROUGH_HEIGHT_MM",
    "SPEED_OF_LIGHT_M_S",
    "TEMPERATURE_C",
]
