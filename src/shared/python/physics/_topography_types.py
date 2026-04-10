from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class ElevationPoint:
    """A single point with elevation data.

    Attributes:
        x: X coordinate [m]
        y: Y coordinate [m]
        z: Elevation [m]
    """

    x: float
    y: float
    z: float

    def as_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z])


@dataclass
class TopographyBounds:
    """Bounds of a topographical region.

    Attributes:
        min_x: Minimum X coordinate [m]
        max_x: Maximum X coordinate [m]
        min_y: Minimum Y coordinate [m]
        max_y: Maximum Y coordinate [m]
        min_z: Minimum elevation [m]
        max_z: Maximum elevation [m]
    """

    min_x: float = 0.0
    max_x: float = 100.0
    min_y: float = 0.0
    max_y: float = 100.0
    min_z: float = 0.0
    max_z: float = 10.0

    @property
    def width(self) -> float:
        """Width in X dimension."""
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        """Height in Y dimension."""
        return self.max_y - self.min_y

    @property
    def elevation_range(self) -> float:
        """Range of elevations."""
        return self.max_z - self.min_z


@runtime_checkable
class TopographyProvider(Protocol):
    """Protocol for objects that provide topographical data."""

    def get_elevation_at(self, position: np.ndarray) -> float:
        """Get elevation at a position.

        Args:
            position: [x, y] position [m]

        Returns:
            Elevation [m]
        """
        ...

    def get_gradient_at(self, position: np.ndarray) -> np.ndarray:
        """Get elevation gradient at position.

        Args:
            position: [x, y] position [m]

        Returns:
            [dz/dx, dz/dy] gradient vector
        """
        ...

    @property
    def bounds(self) -> TopographyBounds:
        """Get the bounds of the topography."""
        ...
