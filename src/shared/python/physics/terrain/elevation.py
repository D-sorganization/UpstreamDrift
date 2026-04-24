import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ElevationMap:
    """Height map for terrain elevation.

    Stores elevation data on a regular grid and provides interpolated
    queries for arbitrary positions.
    """

    data: np.ndarray
    resolution: float
    width: float
    length: float
    origin_x: float = 0.0
    origin_y: float = 0.0

    @classmethod
    def flat(
        cls,
        width: float,
        length: float,
        resolution: float,
        base_elevation: float = 0.0,
    ) -> "ElevationMap":
        """Create a flat elevation map."""
        if width <= 0 or length <= 0:
            raise ValueError("Width and length must be positive")
        if resolution <= 0:
            raise ValueError("Resolution must be positive")

        n_cols = int(width / resolution)
        n_rows = int(length / resolution)

        data = np.full((n_rows, n_cols), base_elevation, dtype=np.float64)

        return cls(
            data=data,
            resolution=resolution,
            width=width,
            length=length,
        )

    @classmethod
    def sloped(
        cls,
        width: float,
        length: float,
        resolution: float,
        slope_angle_deg: float,
        slope_direction_deg: float,
        base_elevation: float = 0.0,
    ) -> "ElevationMap":
        """Create a uniformly sloped elevation map."""
        if width <= 0 or length <= 0:
            raise ValueError("Width and length must be positive")
        if resolution <= 0:
            raise ValueError("Resolution must be positive")
        if abs(slope_angle_deg) > 89:
            logger.warning(f"Steep slope angle: {slope_angle_deg} degrees")

        n_cols = int(width / resolution)
        n_rows = int(length / resolution)

        x = np.arange(n_cols) * resolution
        y = np.arange(n_rows) * resolution
        X, Y = np.meshgrid(x, y)

        slope_rad = math.radians(slope_angle_deg)
        dir_rad = math.radians(slope_direction_deg)

        grad_magnitude = math.tan(slope_rad)
        grad_x = grad_magnitude * math.cos(dir_rad)
        grad_y = grad_magnitude * math.sin(dir_rad)

        data = base_elevation + grad_x * X + grad_y * Y

        return cls(
            data=data,
            resolution=resolution,
            width=width,
            length=length,
        )

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        resolution: float,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ) -> "ElevationMap":
        """Create elevation map from numpy array."""
        if resolution <= 0:
            raise ValueError("Resolution must be positive")

        n_rows, n_cols = data.shape
        width = n_cols * resolution
        length = n_rows * resolution

        return cls(
            data=data.astype(np.float64),
            resolution=resolution,
            width=width,
            length=length,
            origin_x=origin_x,
            origin_y=origin_y,
        )

    def _to_grid_coords(self, x: float, y: float) -> tuple[float, float]:
        """Convert world coordinates to grid coordinates."""
        if x is None:
            raise ValueError("x must be provided")
        gx = (x - self.origin_x) / self.resolution
        gy = (y - self.origin_y) / self.resolution
        return gx, gy

    def _check_bounds(self, x: float, y: float) -> None:
        """Check if coordinates are within bounds."""
        if x < self.origin_x or x > self.origin_x + self.width:
            raise ValueError(
                f"X coordinate {x} out of bounds [{self.origin_x}, {self.origin_x + self.width}]"
            )
        if y < self.origin_y or y > self.origin_y + self.length:
            raise ValueError(
                f"Y coordinate {y} out of bounds [{self.origin_y}, {self.origin_y + self.length}]"
            )

    def get_elevation(self, x: float, y: float) -> float:
        """Get interpolated elevation at a point."""
        if x is None:
            raise ValueError("x must be provided")
        self._check_bounds(x, y)

        gx, gy = self._to_grid_coords(x, y)
        n_rows, n_cols = self.data.shape

        gx = max(0, min(gx, n_cols - 1))
        gy = max(0, min(gy, n_rows - 1))

        ix = int(gx)
        iy = int(gy)
        fx = gx - ix
        fy = gy - iy

        ix1 = min(ix + 1, n_cols - 1)
        iy1 = min(iy + 1, n_rows - 1)

        h00 = self.data[iy, ix]
        h10 = self.data[iy, ix1]
        h01 = self.data[iy1, ix]
        h11 = self.data[iy1, ix1]

        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx
        h = h0 * (1 - fy) + h1 * fy

        return float(h)

    def get_gradient(self, x: float, y: float) -> tuple[float, float]:
        """Get elevation gradient (slope) at a point."""
        if x is None:
            raise ValueError("x must be provided")
        self._check_bounds(x, y)

        gx, gy = self._to_grid_coords(x, y)
        n_rows, n_cols = self.data.shape

        ix = int(max(0, min(gx, n_cols - 1)))
        iy = int(max(0, min(gy, n_rows - 1)))

        if ix > 0 and ix < n_cols - 1:
            dzdx = (self.data[iy, ix + 1] - self.data[iy, ix - 1]) / (
                2 * self.resolution
            )
        elif ix == 0:
            dzdx = (self.data[iy, ix + 1] - self.data[iy, ix]) / self.resolution
        else:
            dzdx = (self.data[iy, ix] - self.data[iy, ix - 1]) / self.resolution

        if iy > 0 and iy < n_rows - 1:
            dzdy = (self.data[iy + 1, ix] - self.data[iy - 1, ix]) / (
                2 * self.resolution
            )
        elif iy == 0:
            dzdy = (self.data[iy + 1, ix] - self.data[iy, ix]) / self.resolution
        else:
            dzdy = (self.data[iy, ix] - self.data[iy - 1, ix]) / self.resolution

        return float(dzdx), float(dzdy)

    def get_normal(self, x: float, y: float) -> np.ndarray:
        """Get surface normal vector at a point."""
        if x is None:
            raise ValueError("x must be provided")
        dzdx, dzdy = self.get_gradient(x, y)
        normal = np.array([-dzdx, -dzdy, 1.0])
        normal = normal / np.linalg.norm(normal)
        return normal

    def get_slope_angle(self, x: float, y: float) -> float:
        """Get slope angle at a point."""
        if x is None:
            raise ValueError("x must be provided")
        dzdx, dzdy = self.get_gradient(x, y)
        slope_magnitude = math.sqrt(dzdx**2 + dzdy**2)
        return math.degrees(math.atan(slope_magnitude))

    def to_dict(self) -> dict[str, Any]:
        """Serialize elevation map to dictionary."""
        return {
            "data": self.data.tolist(),
            "resolution": self.resolution,
            "width": self.width,
            "length": self.length,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ElevationMap":
        """Create elevation map from dictionary."""
        if data is None:
            raise ValueError("data must be provided")
        return cls(
            data=np.array(data["data"], dtype=np.float64),
            resolution=data["resolution"],
            width=data["width"],
            length=data["length"],
            origin_x=data.get("origin_x", 0.0),
            origin_y=data.get("origin_y", 0.0),
        )
