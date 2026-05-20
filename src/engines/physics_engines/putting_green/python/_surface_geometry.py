from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import interpolate, ndimage

from src.engines.physics_engines.putting_green.python._surface_data import (
    ContourPoint,
    SlopeRegion,
)
from src.shared.python.core.physics_constants import GRAVITY_M_S2


class SurfaceGeometryMixin:
    """Mixin providing elevation, slope, ridge, and depression logic for GreenSurface."""

    def _init_geometry(self) -> None:
        self._slope_regions: list[SlopeRegion] = []
        self._contour_points: list[ContourPoint] = []
        self._heightmap: np.ndarray | None = None
        self._heightmap_interpolator: Any = None
        self._ridges: list[dict[str, Any]] = []
        self._depressions: list[dict[str, Any]] = []

    def add_slope_region(self, region: SlopeRegion) -> None:
        """Add a slope region to the green."""
        self._slope_regions.append(region)

    def set_contour_points(self, points: list[ContourPoint]) -> None:
        """Set elevation from scattered contour points.

        Points are interpolated to create a smooth surface.

        Args:
            points: List of ContourPoint objects
        """
        if points is None:
            raise ValueError("points must be provided")
        self._contour_points = points
        self._build_contour_interpolator()

    def _build_contour_interpolator(self) -> None:
        """Build interpolator from contour points."""
        if not self._contour_points:
            return

        x = np.array([p.x for p in self._contour_points])
        y = np.array([p.y for p in self._contour_points])
        z = np.array([p.elevation for p in self._contour_points])

        # Use RBF interpolation for smooth surface
        try:
            self._heightmap_interpolator = interpolate.RBFInterpolator(
                np.column_stack([x, y]),
                z,
                kernel="thin_plate_spline",
                smoothing=0.01,
            )
        except (ValueError, TypeError, RuntimeError):
            # Fallback to linear interpolation
            self._heightmap_interpolator = interpolate.LinearNDInterpolator(
                np.column_stack([x, y]), z, fill_value=0.0
            )

    def set_heightmap(
        self,
        heightmap: np.ndarray,
        smooth: bool = True,
        smooth_sigma: float = 1.0,
    ) -> None:
        """Set surface from 2D heightmap array.

        Args:
            heightmap: 2D array of elevation values [m]
            smooth: Whether to smooth the heightmap
            smooth_sigma: Gaussian smoothing sigma
        """
        if heightmap is None:
            raise ValueError("heightmap must be provided")
        if smooth:
            heightmap = ndimage.gaussian_filter(heightmap, sigma=smooth_sigma)

        self._heightmap = heightmap.astype(np.float64)

        # Build interpolator
        ny, nx = heightmap.shape
        x = np.linspace(0, self.width, nx)  # type: ignore[attr-defined]
        y = np.linspace(0, self.height, ny)  # type: ignore[attr-defined]

        self._heightmap_interpolator = interpolate.RegularGridInterpolator(
            (y, x),
            self._heightmap,
            method="cubic",
            bounds_error=False,
            fill_value=0.0,
        )

    def get_elevation_at(self, position: np.ndarray) -> float:
        """Get elevation at a position.

        Args:
            position: [x, y] position on green [m]

        Returns:
            Elevation at position [m]
        """
        if position is None:
            raise ValueError("position must be provided")
        pos = np.clip(position[:2], [0, 0], [self.width, self.height])  # type: ignore[attr-defined]

        # Base elevation from heightmap or contours
        elevation = 0.0

        if self._heightmap_interpolator is not None:
            if self._heightmap is not None:
                # Regular grid interpolator (y, x order)
                elevation = float(self._heightmap_interpolator([[pos[1], pos[0]]])[0])
            else:
                # RBF interpolator (x, y order)
                elevation = float(self._heightmap_interpolator([[pos[0], pos[1]]])[0])

        # Add ridge contributions
        for ridge in self._ridges:
            elevation += self._ridge_elevation(pos, ridge)

        # Add depression contributions
        for depression in self._depressions:
            elevation += self._depression_elevation(pos, depression)

        return elevation

    def get_gradient_at(self, position: np.ndarray, delta: float = 0.01) -> np.ndarray:
        """Get elevation gradient at position.

        Uses numerical differentiation.

        Args:
            position: [x, y] position on green [m]
            delta: Step size for numerical gradient [m]

        Returns:
            [dz/dx, dz/dy] gradient vector
        """
        if position is None:
            raise ValueError("position must be provided")
        pos = position[:2]

        # Central difference
        dzdx = (
            self.get_elevation_at(pos + [delta, 0])
            - self.get_elevation_at(pos - [delta, 0])
        ) / (2 * delta)

        dzdy = (
            self.get_elevation_at(pos + [0, delta])
            - self.get_elevation_at(pos - [0, delta])
        ) / (2 * delta)

        return np.array([dzdx, dzdy])

    def get_slope_at(self, position: np.ndarray) -> np.ndarray:
        """Get slope vector at position.

        Combines contributions from slope regions and elevation gradient.

        Args:
            position: [x, y] position on green [m]

        Returns:
            [slope_x, slope_y] slope vector (gradient)
        """
        if position is None:
            raise ValueError("position must be provided")
        pos = position[:2]
        total_slope = np.zeros(2)

        # Contribution from slope regions
        for region in self._slope_regions:
            weight = region.get_weight(pos)
            if weight > 0:
                total_slope += weight * region.slope_magnitude * region.slope_direction

        # Contribution from elevation gradient
        if self._heightmap_interpolator is not None:
            gradient = self.get_gradient_at(pos)
            total_slope += gradient

        return total_slope

    def get_gravitational_acceleration(self, position: np.ndarray) -> np.ndarray:
        """Get gravitational acceleration component on sloped surface.

        On a slope, gravity has a component parallel to the surface
        that accelerates the ball downhill.

        Args:
            position: [x, y] position on green [m]

        Returns:
            [ax, ay] gravitational acceleration [m/s²]
        """
        if position is None:
            raise ValueError("position must be provided")
        slope = self.get_slope_at(position)
        # Acceleration is proportional to slope and points downhill
        # a = g * sin(theta) ≈ g * slope for small slopes
        return -GRAVITY_M_S2 * slope

    def add_ridge(
        self,
        start: np.ndarray,
        end: np.ndarray,
        height: float,
        width: float,
    ) -> None:
        """Add a ridge (raised linear feature) to the green.

        Args:
            start: Start point of ridge [m, m]
            end: End point of ridge [m, m]
            height: Maximum height of ridge [m]
            width: Width of ridge influence [m]
        """
        self._ridges.append(
            {
                "start": np.array(start[:2]),
                "end": np.array(end[:2]),
                "height": height,
                "width": width,
            }
        )

    def add_depression(
        self,
        center: np.ndarray,
        radius: float,
        depth: float,
    ) -> None:
        """Add a depression (bowl/hollow) to the green.

        Args:
            center: Center of depression [m, m]
            radius: Radius of depression [m]
            depth: Maximum depth [m]
        """
        self._depressions.append(
            {
                "center": np.array(center[:2]),
                "radius": radius,
                "depth": depth,
            }
        )

    def _ridge_elevation(self, position: np.ndarray, ridge: dict[str, Any]) -> float:
        """Compute elevation contribution from a ridge."""
        if position is None:
            raise ValueError("position must be provided")
        start = ridge["start"]
        end = ridge["end"]
        height = ridge["height"]
        width = ridge["width"]

        # Project point onto ridge line
        line_vec = end - start
        line_len = math.hypot(*line_vec)
        if line_len < 1e-10:
            return 0.0

        line_dir = line_vec / line_len
        to_point = position - start
        projection = np.dot(to_point, line_dir)

        # Check if projection is on line segment
        if projection < 0 or projection > line_len:
            return 0.0

        # Distance from line
        closest_on_line = start + projection * line_dir
        arr = np.asarray(position - closest_on_line, dtype=float).reshape(-1)
        distance = 0.0 if arr.size == 0 else math.hypot(*arr)

        if distance > width:
            return 0.0

        # Gaussian profile
        elevation = height * np.exp(-0.5 * (distance / (width / 2.5)) ** 2)
        return elevation

    def _depression_elevation(
        self, position: np.ndarray, depression: dict[str, Any]
    ) -> float:
        """Compute elevation contribution from a depression."""
        if position is None:
            raise ValueError("position must be provided")
        center = depression["center"]
        radius = depression["radius"]
        depth = depression["depth"]

        arr = np.asarray(position - center, dtype=float).reshape(-1)
        distance = 0.0 if arr.size == 0 else math.hypot(*arr)

        if distance > radius:
            return 0.0

        # Parabolic profile
        normalized_dist = distance / radius
        elevation = -depth * (1 - normalized_dist**2)
        return elevation
