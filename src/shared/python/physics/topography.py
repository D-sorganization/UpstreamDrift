"""Shared Topographical Data Module.

This module provides reusable topographical/elevation data handling that can be
used across multiple physics models including:
- Putting green surfaces
- Golf course terrain
- General ground surfaces for ball flight

Supported formats:
- NumPy arrays (.npy)
- CSV files (x, y, elevation columns)
- GeoTIFF (.tif, .tiff) - requires rasterio
- JSON contour definitions
- Image-based heightmaps (.png, .jpg)

Design by Contract:
    - All elevations are in meters
    - All coordinates are in meters unless specified
    - Interpolation is provided for smooth surface queries
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy import interpolate, ndimage


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


class TopographyData:
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
        if smooth:
            heightmap = ndimage.gaussian_filter(heightmap, sigma=smooth_sigma)

        self._heightmap = heightmap.astype(np.float64)

        # Update elevation bounds
        self._bounds.min_z = float(np.min(heightmap))
        self._bounds.max_z = float(np.max(heightmap))

        # Build interpolator
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
        self._contour_points = points

        if not points:
            return

        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])
        z = np.array([p.z for p in points])

        # Update bounds
        self._bounds.min_x = float(np.min(x))
        self._bounds.max_x = float(np.max(x))
        self._bounds.min_y = float(np.min(y))
        self._bounds.max_y = float(np.max(y))
        self._bounds.min_z = float(np.min(z))
        self._bounds.max_z = float(np.max(z))

        # Use RBF interpolation for smooth surface
        try:
            self._interpolator = interpolate.RBFInterpolator(
                np.column_stack([x, y]),
                z,
                kernel="thin_plate_spline",
                smoothing=0.01,
            )
        except Exception:
            # Fallback to linear interpolation
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
        if not self._is_loaded or self._interpolator is None:
            return 0.0

        pos = position[:2]

        # Clamp to bounds
        pos = np.clip(
            pos,
            [self._bounds.min_x, self._bounds.min_y],
            [self._bounds.max_x, self._bounds.max_y],
        )

        if self._heightmap is not None:
            # Regular grid interpolator (y, x order)
            return float(self._interpolator([[pos[1], pos[0]]])[0])
        else:
            # RBF interpolator (x, y order)
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

    def get_normal_at(self, position: np.ndarray) -> np.ndarray:
        """Get surface normal vector at position.

        Args:
            position: [x, y] position [m]

        Returns:
            [nx, ny, nz] unit normal vector
        """
        gradient = self.get_gradient_at(position)

        # Surface normal from gradient: n = normalize([-dz/dx, -dz/dy, 1])
        normal = np.array([-gradient[0], -gradient[1], 1.0])
        return normal / np.linalg.norm(normal)

    def to_heightmap(self, resolution: int = 100) -> np.ndarray:
        """Export as heightmap array.

        Args:
            resolution: Number of points in each dimension

        Returns:
            2D array of elevations [resolution x resolution]
        """
        x = np.linspace(self._bounds.min_x, self._bounds.max_x, resolution)
        y = np.linspace(self._bounds.min_y, self._bounds.max_y, resolution)

        heightmap = np.zeros((resolution, resolution))

        for i, yi in enumerate(y):
            for j, xi in enumerate(x):
                heightmap[i, j] = self.get_elevation_at(np.array([xi, yi]))

        return heightmap

    @classmethod
    def from_file(
        cls,
        filepath: str | Path,
        width: float | None = None,
        height: float | None = None,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> TopographyData:
        """Load topography from file.

        Args:
            filepath: Path to data file
            width: Physical width [m] (auto-detected if None)
            height: Physical height [m] (auto-detected if None)
            origin: Origin point (min_x, min_y)

        Returns:
            TopographyData instance
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        topo = cls()
        topo._bounds.min_x = origin[0]
        topo._bounds.min_y = origin[1]

        if suffix == ".npy":
            topo._load_numpy(filepath, width, height)
        elif suffix == ".csv":
            topo._load_csv(filepath, width, height)
        elif suffix == ".json":
            topo._load_json(filepath, width, height)
        elif suffix in (".tif", ".tiff"):
            topo._load_geotiff(filepath, width, height)
        elif suffix in (".png", ".jpg", ".jpeg"):
            topo._load_image(filepath, width, height)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return topo

    def _load_numpy(
        self, filepath: Path, width: float | None, height: float | None
    ) -> None:
        """Load from NumPy file."""
        heightmap = np.load(filepath)

        if width is not None:
            self._bounds.max_x = self._bounds.min_x + width
        else:
            self._bounds.max_x = self._bounds.min_x + heightmap.shape[1]

        if height is not None:
            self._bounds.max_y = self._bounds.min_y + height
        else:
            self._bounds.max_y = self._bounds.min_y + heightmap.shape[0]

        self.set_heightmap(heightmap, smooth=False)

    def _load_csv(
        self, filepath: Path, width: float | None, height: float | None
    ) -> None:
        """Load from CSV file with x, y, elevation columns."""
        import csv

        points = []
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try different column name conventions
                x = float(row.get("x", row.get("X", row.get("easting", 0))))
                y = float(row.get("y", row.get("Y", row.get("northing", 0))))
                z = float(
                    row.get(
                        "elevation", row.get("z", row.get("Z", row.get("height", 0)))
                    )
                )
                points.append(ElevationPoint(x=x, y=y, z=z))

        self.set_contour_points(points)

        # Override bounds if provided
        if width is not None:
            self._bounds.max_x = self._bounds.min_x + width
        if height is not None:
            self._bounds.max_y = self._bounds.min_y + height

    def _load_json(
        self, filepath: Path, width: float | None, height: float | None
    ) -> None:
        """Load from JSON file."""
        with open(filepath) as f:
            data = json.load(f)

        # Check for different JSON structures
        if "contours" in data:
            points = [
                ElevationPoint(x=p["x"], y=p["y"], z=p.get("z", p.get("elevation", 0)))
                for p in data["contours"]
            ]
            self.set_contour_points(points)
        elif "heightmap" in data:
            heightmap = np.array(data["heightmap"])
            w = data.get("width", width or heightmap.shape[1])
            h = data.get("height", height or heightmap.shape[0])
            self._bounds.max_x = self._bounds.min_x + w
            self._bounds.max_y = self._bounds.min_y + h
            self.set_heightmap(heightmap)

        if width is not None:
            self._bounds.max_x = self._bounds.min_x + width
        if height is not None:
            self._bounds.max_y = self._bounds.min_y + height

    def _load_geotiff(
        self, filepath: Path, width: float | None, height: float | None
    ) -> None:
        """Load from GeoTIFF file."""
        try:
            import rasterio  # type: ignore
        except ImportError as err:
            raise ImportError(
                "rasterio required for GeoTIFF support. Install with: pip install rasterio"
            ) from err

        with rasterio.open(filepath) as src:
            heightmap = src.read(1)

            # Get bounds from GeoTIFF metadata
            if width is None:
                self._bounds.min_x = src.bounds.left
                self._bounds.max_x = src.bounds.right
            else:
                self._bounds.max_x = self._bounds.min_x + width

            if height is None:
                self._bounds.min_y = src.bounds.bottom
                self._bounds.max_y = src.bounds.top
            else:
                self._bounds.max_y = self._bounds.min_y + height

        self.set_heightmap(heightmap, smooth=False)

    def _load_image(
        self, filepath: Path, width: float | None, height: float | None
    ) -> None:
        """Load from image file (grayscale as elevation)."""
        try:
            from PIL import Image  # type: ignore
        except ImportError:
            # Fall back to matplotlib if PIL not available
            import matplotlib.pyplot as plt

            img = plt.imread(str(filepath))
            if len(img.shape) == 3:
                # Convert to grayscale
                heightmap = np.mean(img, axis=2)
            else:
                heightmap = img
        else:
            pil_img = Image.open(filepath).convert("L")
            heightmap = np.array(pil_img) / 255.0  # Normalize to 0-1

        if width is not None:
            self._bounds.max_x = self._bounds.min_x + width
        else:
            self._bounds.max_x = self._bounds.min_x + heightmap.shape[1]

        if height is not None:
            self._bounds.max_y = self._bounds.min_y + height
        else:
            self._bounds.max_y = self._bounds.min_y + heightmap.shape[0]

        self.set_heightmap(heightmap)

    def save_to_file(self, filepath: str | Path, format: str | None = None) -> None:
        """Save topography to file.

        Args:
            filepath: Output file path
            format: Output format ("npy", "csv", "json") - auto-detected from suffix if None
        """
        filepath = Path(filepath)
        fmt = format or filepath.suffix.lower().lstrip(".")

        if fmt == "npy":
            if self._heightmap is not None:
                np.save(filepath, self._heightmap)
            else:
                np.save(filepath, self.to_heightmap())
        elif fmt == "csv":
            self._save_csv(filepath)
        elif fmt == "json":
            self._save_json(filepath)
        else:
            raise ValueError(f"Unsupported output format: {fmt}")

    def _save_csv(self, filepath: Path) -> None:
        """Save to CSV file."""
        import csv

        heightmap = (
            self._heightmap if self._heightmap is not None else self.to_heightmap()
        )
        ny, nx = heightmap.shape

        x_coords = np.linspace(self._bounds.min_x, self._bounds.max_x, nx)
        y_coords = np.linspace(self._bounds.min_y, self._bounds.max_y, ny)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "elevation"])
            for i, y in enumerate(y_coords):
                for j, x in enumerate(x_coords):
                    writer.writerow([x, y, heightmap[i, j]])

    def _save_json(self, filepath: Path) -> None:
        """Save to JSON file."""
        heightmap = (
            self._heightmap if self._heightmap is not None else self.to_heightmap()
        )

        data = {
            "bounds": {
                "min_x": self._bounds.min_x,
                "max_x": self._bounds.max_x,
                "min_y": self._bounds.min_y,
                "max_y": self._bounds.max_y,
                "min_z": self._bounds.min_z,
                "max_z": self._bounds.max_z,
            },
            "heightmap": heightmap.tolist(),
            "width": self._bounds.width,
            "height": self._bounds.height,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def sample_uniform(self, nx: int, ny: int) -> np.ndarray:
        """Sample elevations on uniform grid.

        Args:
            nx: Number of samples in X
            ny: Number of samples in Y

        Returns:
            Array of shape (ny, nx) with elevations
        """
        x = np.linspace(self._bounds.min_x, self._bounds.max_x, nx)
        y = np.linspace(self._bounds.min_y, self._bounds.max_y, ny)

        result = np.zeros((ny, nx))
        for i, yi in enumerate(y):
            for j, xi in enumerate(x):
                result[i, j] = self.get_elevation_at(np.array([xi, yi]))

        return result

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


def create_flat_terrain(
    width: float = 100.0,
    height: float = 100.0,
    elevation: float = 0.0,
) -> TopographyData:
    """Create flat terrain.

    Args:
        width: Width [m]
        height: Height [m]
        elevation: Constant elevation [m]

    Returns:
        TopographyData with flat surface
    """
    topo = TopographyData(
        bounds=TopographyBounds(min_x=0, max_x=width, min_y=0, max_y=height)
    )
    heightmap = np.full((10, 10), elevation)
    topo.set_heightmap(heightmap, smooth=False)
    return topo


def create_sloped_terrain(
    width: float = 100.0,
    height: float = 100.0,
    slope_direction: np.ndarray = np.array([1.0, 0.0]),
    slope_magnitude: float = 0.02,
    base_elevation: float = 0.0,
) -> TopographyData:
    """Create uniformly sloped terrain.

    Args:
        width: Width [m]
        height: Height [m]
        slope_direction: Direction of downhill slope
        slope_magnitude: Steepness (rise/run)
        base_elevation: Elevation at origin [m]

    Returns:
        TopographyData with sloped surface
    """
    topo = TopographyData(
        bounds=TopographyBounds(min_x=0, max_x=width, min_y=0, max_y=height)
    )

    # Normalize direction
    slope_dir = slope_direction / np.linalg.norm(slope_direction)

    resolution = 50
    x = np.linspace(0, width, resolution)
    y = np.linspace(0, height, resolution)
    X, Y = np.meshgrid(x, y)

    # Elevation = base - slope * (x * dir_x + y * dir_y)
    heightmap = base_elevation - slope_magnitude * (X * slope_dir[0] + Y * slope_dir[1])

    topo.set_heightmap(heightmap, smooth=False)
    return topo


def create_undulating_terrain(
    width: float = 100.0,
    height: float = 100.0,
    amplitude: float = 1.0,
    wavelength: float = 20.0,
    base_elevation: float = 0.0,
) -> TopographyData:
    """Create undulating (sine wave) terrain.

    Args:
        width: Width [m]
        height: Height [m]
        amplitude: Wave amplitude [m]
        wavelength: Wave wavelength [m]
        base_elevation: Mean elevation [m]

    Returns:
        TopographyData with undulating surface
    """
    topo = TopographyData(
        bounds=TopographyBounds(min_x=0, max_x=width, min_y=0, max_y=height)
    )

    resolution = 100
    x = np.linspace(0, width, resolution)
    y = np.linspace(0, height, resolution)
    X, Y = np.meshgrid(x, y)

    # Sinusoidal terrain
    k = 2 * np.pi / wavelength
    heightmap = base_elevation + amplitude * (
        np.sin(k * X) * np.cos(k * Y) + 0.5 * np.sin(2 * k * X)
    )

    topo.set_heightmap(heightmap, smooth=True)
    return topo
