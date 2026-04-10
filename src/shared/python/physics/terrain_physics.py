"""Terrain physics helpers."""

from __future__ import annotations

import functools
import math

import numpy as np

from src.shared.python.core.physics_constants import GRAVITY_M_S2

from .terrain import ElevationMap


@functools.lru_cache(maxsize=256)
def compute_gravity_on_slope(
    slope_angle_deg: float,
    gravity: float = float(GRAVITY_M_S2),
) -> tuple[float, float]:
    """Compute gravity components on a slope."""
    if not (slope_angle_deg is not None):
        raise ValueError("slope_angle_deg must be provided")
    slope_rad = math.radians(slope_angle_deg)
    g_parallel = gravity * math.sin(slope_rad)
    g_perpendicular = gravity * math.cos(slope_rad)
    return g_parallel, g_perpendicular


def compute_roll_direction(
    elevation: ElevationMap,
    x: float,
    y: float,
) -> np.ndarray:
    """Compute the downhill roll direction on terrain."""
    if not (elevation is not None):
        raise ValueError("elevation must be provided")
    dzdx, dzdy = elevation.get_gradient(x, y)
    roll_dir = np.array([-dzdx, -dzdy])
    magnitude = np.linalg.norm(roll_dir)
    if magnitude < 1e-10:
        return np.zeros(2)
    return roll_dir / magnitude


def get_contact_normal(
    elevation: ElevationMap,
    x: float,
    y: float,
) -> np.ndarray:
    """Get the terrain contact normal for physics integrations."""
    return elevation.get_normal(x, y)
