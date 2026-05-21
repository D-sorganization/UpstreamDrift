from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ContourPoint:
    """A single point defining elevation on the green surface.

    Attributes:
        x: X coordinate [m]
        y: Y coordinate [m]
        elevation: Height above reference [m]
    """

    x: float
    y: float
    elevation: float

    def as_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, elevation]."""
        return np.array([self.x, self.y, self.elevation])


@dataclass
class SlopeRegion:
    """Defines a region with uniform slope.

    Attributes:
        center: Center point of the slope region [m, m]
        radius: Radius of influence [m]
        slope_direction: Direction of downhill slope (unit vector)
        slope_magnitude: Steepness of slope (rise/run, e.g., 0.02 = 2%)
        falloff: How quickly slope fades at edges (0-1, 1 = sharp)
    """

    center: np.ndarray
    radius: float
    slope_direction: np.ndarray
    slope_magnitude: float
    falloff: float = 0.5

    def __post_init__(self) -> None:
        """Normalize slope direction."""
        mag = math.hypot(*self.slope_direction)
        if mag > 0:
            self.slope_direction = self.slope_direction / mag

    def contains(self, position: np.ndarray) -> bool:
        """Check if position is within region."""
        if position is None:
            raise ValueError("position must be provided")
        arr = np.asarray(position[:2] - self.center[:2], dtype=float).reshape(-1)
        distance = 0.0 if arr.size == 0 else math.hypot(*arr)
        return bool(distance <= self.radius)

    def get_weight(self, position: np.ndarray) -> float:
        """Get influence weight at position (0-1)."""
        if position is None:
            raise ValueError("position must be provided")
        arr = np.asarray(position[:2] - self.center[:2], dtype=float).reshape(-1)
        distance = 0.0 if arr.size == 0 else math.hypot(*arr)
        if distance >= self.radius:
            return 0.0

        # Smooth falloff
        normalized_dist = distance / self.radius
        return float(1.0 - normalized_dist ** (1.0 / (1.0 - self.falloff + 0.1)))
