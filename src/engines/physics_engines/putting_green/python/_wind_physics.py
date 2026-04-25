from __future__ import annotations

import numpy as np

from src.shared.python.core.physics_constants import (
    AIR_DENSITY_SEA_LEVEL_KG_M3,
    GOLF_BALL_CROSS_SECTIONAL_AREA_M2,
    PUTTING_WIND_DRAG_COEFFICIENT,
    PUTTING_WIND_FORCE_SCALING,
)


def compute_wind_force(
    wind_speed: float,
    wind_direction: np.ndarray,
    ball_velocity: np.ndarray,
) -> np.ndarray:
    """Compute wind force on ball."""
    if wind_speed <= 0:
        return np.zeros(2)

    rho = AIR_DENSITY_SEA_LEVEL_KG_M3
    Cd = PUTTING_WIND_DRAG_COEFFICIENT
    A = GOLF_BALL_CROSS_SECTIONAL_AREA_M2

    relative_v = wind_direction * wind_speed - ball_velocity[:2]
    rel_speed = np.linalg.norm(relative_v)

    if rel_speed < 0.1:
        return np.zeros(2)

    force_mag = 0.5 * rho * Cd * A * rel_speed**2
    force_dir = relative_v / rel_speed

    return force_mag * force_dir * PUTTING_WIND_FORCE_SCALING
