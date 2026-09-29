# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Green Surface Model for Putting Simulation.

This module defines the putting green surface including slopes,
undulations, elevation contours, and hole position.

Supports loading topographical data from various formats:
- NumPy arrays (heightmaps)
- CSV files
- GeoTIFF (if rasterio available)
- JSON contour definitions

Design by Contract:
    - All positions are in meters
    - Elevations are relative to a reference plane
    - Slopes are expressed as gradients (rise/run)
"""

from __future__ import annotations

import math

import numpy as np

from src.engines.physics_engines.putting_green.python._surface_analysis import (
    SurfaceAnalysisMixin,
)
from src.engines.physics_engines.putting_green.python._surface_data import (
    ContourPoint,
    SlopeRegion,
)
from src.engines.physics_engines.putting_green.python._surface_geometry import (
    SurfaceGeometryMixin,
)
from src.engines.physics_engines.putting_green.python._surface_io import (
    SurfaceIOMixin,
)
from src.engines.physics_engines.putting_green.python._surface_presets import (
    SurfacePresetsMixin,
)
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)

__all__ = [
    "ContourPoint",
    "GreenSurface",
    "SlopeRegion",
]


class GreenSurface(
    SurfaceGeometryMixin,
    SurfaceAnalysisMixin,
    SurfaceIOMixin,
    SurfacePresetsMixin,
):
    """Putting green surface model with elevation and slope data.

    Supports multiple ways to define the surface:
    1. Flat surface (default)
    2. Slope regions (circular areas with uniform slope)
    3. Contour points (scattered elevation samples, interpolated)
    4. Heightmap (2D array of elevations)

    Attributes:
        width: Width of green [m]
        height: Height of green [m]
        turf: Turf properties
        hole_position: Position of hole [m, m]
        hole_radius: Radius of hole [m] (standard = 0.054m = 4.25"/2)
    """

    STANDARD_HOLE_RADIUS = 0.054  # 4.25 inches diameter / 2

    def __init__(
        self,
        width: float = 20.0,
        height: float = 20.0,
        turf: TurfProperties | None = None,
    ) -> None:
        """Initialize green surface.

        Args:
            width: Width of putting surface [m]
            height: Height of putting surface [m]
            turf: Turf properties (defaults to standard)
        """
        if width is None:
            raise ValueError("width must be provided")
        self.width = width
        self.height = height
        self.turf = turf or TurfProperties()

        self._init_geometry()

        # Hole
        self._hole_position = np.array([width / 2, height / 2])
        self.hole_radius = self.STANDARD_HOLE_RADIUS

    @property
    def hole_position(self) -> np.ndarray:
        """Get hole position."""
        return self._hole_position

    def set_hole_position(self, position: np.ndarray) -> None:
        """Set hole position."""
        self._hole_position = np.array(position[:2])

    def is_in_hole(
        self, position: np.ndarray, velocity: np.ndarray | None = None
    ) -> bool:
        """Check if ball position is in the hole.

        A ball is considered holed if:
        1. It's within the hole radius
        2. If moving, velocity is low enough to not lip out

        Args:
            position: Ball position [m, m]
            velocity: Ball velocity [m/s] (optional, for lip-out check)

        Returns:
            True if ball is holed
        """
        if position is None:
            raise ValueError("position must be provided")
        arr = np.asarray(position[:2] - self._hole_position, dtype=float).reshape(-1)
        distance = 0.0 if arr.size == 0 else math.hypot(*arr)

        if distance > self.hole_radius:
            return False

        # Check for lip-out at high speeds
        if velocity is not None:
            speed = math.hypot(*velocity)
            # Empirical: ball lips out if going too fast near edge
            max_speed_at_edge = 1.5  # m/s
            if distance > self.hole_radius * 0.5 and speed > max_speed_at_edge:
                return False

        return True

    def is_on_green(self, position: np.ndarray) -> bool:
        """Check if position is on the green surface."""
        if position is None:
            raise ValueError("position must be provided")
        x, y = position[:2]
        return 0 <= x <= self.width and 0 <= y <= self.height
