from __future__ import annotations

from typing import Any

import numpy as np
from scipy import interpolate, ndimage

from ._topography_io import _TopographyIOMixin
from ._topography_types import ElevationPoint, TopographyBounds


class TopographyData(_TopographyIOMixin):
    """Container for topographical elevation data.

    Provides elevation queries with smooth interpolation
    between data points.

    Example:
        >>> topo = TopographyData.from_file("terrain.npy", width=100.0, height=100.0)
        >>> elevation = topo.get_elevation_at(np.array([50.0, 50.0]))
        >>> gradient = topo.get_gradient_at(np.array([50.0, 50.0]))
    """

    def __init__(
        self,
        bounds: TopographyBounds | None = None,
    ) -> None:
        """Initialize topography data container.

        Args:
            bounds: Physical bounds of the data
        """
        self._bounds = bounds or TopographyBounds()
        self._heightmap: np.ndarray | None = None
        self._interpolator: Any = None
        self._contour_points: list[ElevationPoint] = []
        self._is_loaded = False

    @property
    def bounds(self) -> TopographyBounds:
        """Get the bounds of the topography."""
        return self._bounds

    @property
    def is_loaded(self) -> bool:
        """Check if data is loaded."""
        return self._is_loaded

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

        self._bounds.min_z = float(np.min(heightmap))
        self._bounds.max_z = float(np.max(heightmap))

        ny, nx = heightmap.shape
        x = np.linspace(self._bounds.min_x, self._bounds.max_x, nx)
        y = np.linspace(self._bounds.min_y, self._bounds.max_y, ny)

        self._interpolator = interpolate.RegularGridInterpolator(
            (y, x),
            self._heightmap,
            method="cubic",
            bounds_error=False,
            fill_value=0.0,
        )

        self._is_loaded = True

    def set_contour_points(self, points: list[ElevationPoint]) -> None:
        """Set elevation from scattered contour points.

        Points are interpolated to create a smooth surface.

        Args:
            points: List of ElevationPoint objects
        """
        if points is None:
            raise ValueError("points must be provided")
        self._contour_points = points

        if not points:
            return

        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])
        z = np.array([p.z for p in points])

        self._bounds.min_x = float(np.min(x))
        self._bounds.max_x = float(np.max(x))
        self._bounds.min_y = float(np.min(y))
        self._bounds.max_y = float(np.max(y))
        self._bounds.min_z = float(np.min(z))
        self._bounds.max_z = float(np.max(z))

        try:
            self._interpolator = interpolate.RBFInterpolator(
                np.column_stack([x, y]),
                z,
                kernel="thin_plate_spline",
                smoothing=0.01,
            )
        except (ValueError, TypeError, RuntimeError):
            self._interpolator = interpolate.LinearNDInterpolator(
                np.column_stack([x, y]), z, fill_value=0.0
            )

        self._is_loaded = True

    def get_elevation_at(self, position: np.ndarray) -> float:
        """Get elevation at a position.

        Args:
            position: [x, y] position [m]

        Returns:
            Elevation [m]
        """
        if position is None:
            raise ValueError("position must be provided")
        if not self._is_loaded or self._interpolator is None:
            return 0.0

        pos = position[:2]

        pos = np.clip(
            pos,
            [self._bounds.min_x, self._bounds.min_y],
            [self._bounds.max_x, self._bounds.max_y],
        )

        if self._heightmap is not None:
            return float(self._interpolator([[pos[1], pos[0]]])[0])
        return float(self._interpolator([[pos[0], pos[1]]])[0])

    def get_gradient_at(self, position: np.ndarray, delta: float = 0.01) -> np.ndarray:
        """Get elevation gradient at position.

        Uses numerical differentiation.

        Args:
            position: [x, y] position [m]
            delta: Step size for numerical gradient [m]

        Returns:
            [dz/dx, dz/dy] gradient vector
        """
        if position is None:
            raise ValueError("position must be provided")
        pos = position[:2]

        dzdx = (
            self.get_elevation_at(pos + [delta, 0])
            - self.get_elevation_at(pos - [delta, 0])
        ) / (2 * delta)

        dzdy = (
            self.get_elevation_at(pos + [0, delta])
            - self.get_elevation_at(pos - [0, delta])
        ) / (2 * delta)

        return np.array([dzdx, dzdy])

    def get_normal_at(self, position: np.ndarray) -> np.ndarray:
        """Get surface normal vector at position.

        Args:
            position: [x, y] position [m]

        Returns:
            [nx, ny, nz] unit normal vector
        """
        if position is None:
            raise ValueError("position must be provided")
        gradient = self.get_gradient_at(position)

        normal = np.array([-gradient[0], -gradient[1], 1.0])
        return normal / np.linalg.norm(normal)

    def to_heightmap(self, resolution: int = 100) -> np.ndarray:
        """Export as heightmap array.

        Args:
            resolution: Number of points in each dimension

        Returns:
            2D array of elevations [resolution x resolution]
        """
        return self.sample_uniform(resolution, resolution)

    def sample_uniform(self, nx: int, ny: int) -> np.ndarray:
        """Sample elevations on uniform grid.

        Uses vectorized evaluation when available for performance.
        For a regular-grid interpolator the entire grid is evaluated
        in a single C-level numpy call instead of O(nx * ny) Python loops.

        Args:
            nx: Number of samples in X
            ny: Number of samples in Y

        Returns:
            Array of shape (ny, nx) with elevations
        """
        if nx is None:
            raise ValueError("nx must be provided")
        if not self._is_loaded or self._interpolator is None:
            return np.zeros((ny, nx))

        x = np.linspace(self._bounds.min_x, self._bounds.max_x, nx)
        y = np.linspace(self._bounds.min_y, self._bounds.max_y, ny)
        X, Y = np.meshgrid(x, y)

        X_clamped = np.clip(X, self._bounds.min_x, self._bounds.max_x)
        Y_clamped = np.clip(Y, self._bounds.min_y, self._bounds.max_y)

        if self._heightmap is not None:
            pts = np.column_stack([Y_clamped.ravel(), X_clamped.ravel()])
            result = self._interpolator(pts)
        else:
            pts = np.column_stack([X_clamped.ravel(), Y_clamped.ravel()])
            result = self._interpolator(pts)

        return result.reshape(ny, nx)

    def get_statistics(self) -> dict[str, float]:
        """Get statistics about the topography.

        Returns:
            Dictionary with min, max, mean, std elevation
        """
        heightmap = (
            self._heightmap if self._heightmap is not None else self.to_heightmap(50)
        )

        return {
            "min_elevation": float(np.min(heightmap)),
            "max_elevation": float(np.max(heightmap)),
            "mean_elevation": float(np.mean(heightmap)),
            "std_elevation": float(np.std(heightmap)),
            "elevation_range": float(np.max(heightmap) - np.min(heightmap)),
        }
