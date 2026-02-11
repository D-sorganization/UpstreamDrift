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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import interpolate, ndimage

from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)
from src.shared.python.core.physics_constants import GRAVITY_M_S2


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
        mag = np.linalg.norm(self.slope_direction)
        if mag > 0:
            self.slope_direction = self.slope_direction / mag

    def contains(self, position: np.ndarray) -> bool:
        """Check if position is within region."""
        distance = np.linalg.norm(position[:2] - self.center[:2])
        return bool(distance <= self.radius)

    def get_weight(self, position: np.ndarray) -> float:
        """Get influence weight at position (0-1)."""
        distance = np.linalg.norm(position[:2] - self.center[:2])
        if distance >= self.radius:
            return 0.0

        # Smooth falloff
        normalized_dist = distance / self.radius
        return float(1.0 - normalized_dist ** (1.0 / (1.0 - self.falloff + 0.1)))


class GreenSurface:
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
        self.width = width
        self.height = height
        self.turf = turf or TurfProperties()

        # Surface definition
        self._slope_regions: list[SlopeRegion] = []
        self._contour_points: list[ContourPoint] = []
        self._heightmap: np.ndarray | None = None
        self._heightmap_interpolator: Any = None

        # Features
        self._ridges: list[dict[str, Any]] = []
        self._depressions: list[dict[str, Any]] = []

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

    def add_slope_region(self, region: SlopeRegion) -> None:
        """Add a slope region to the green."""
        self._slope_regions.append(region)

    def set_contour_points(self, points: list[ContourPoint]) -> None:
        """Set elevation from scattered contour points.

        Points are interpolated to create a smooth surface.

        Args:
            points: List of ContourPoint objects
        """
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
        if smooth:
            heightmap = ndimage.gaussian_filter(heightmap, sigma=smooth_sigma)

        self._heightmap = heightmap.astype(np.float64)

        # Build interpolator
        ny, nx = heightmap.shape
        x = np.linspace(0, self.width, nx)
        y = np.linspace(0, self.height, ny)

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
        pos = np.clip(position[:2], [0, 0], [self.width, self.height])

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
        slope = self.get_slope_at(position)
        # Acceleration is proportional to slope and points downhill
        # a = g * sin(theta) ≈ g * slope for small slopes
        return -GRAVITY_M_S2 * slope

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
        distance = np.linalg.norm(position[:2] - self._hole_position)

        if distance > self.hole_radius:
            return False

        # Check for lip-out at high speeds
        if velocity is not None:
            speed = np.linalg.norm(velocity)
            # Empirical: ball lips out if going too fast near edge
            max_speed_at_edge = 1.5  # m/s
            if distance > self.hole_radius * 0.5 and speed > max_speed_at_edge:
                return False

        return True

    def is_on_green(self, position: np.ndarray) -> bool:
        """Check if position is on the green surface."""
        x, y = position[:2]
        return 0 <= x <= self.width and 0 <= y <= self.height

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
        start = ridge["start"]
        end = ridge["end"]
        height = ridge["height"]
        width = ridge["width"]

        # Project point onto ridge line
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
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
        distance = np.linalg.norm(position - closest_on_line)

        if distance > width:
            return 0.0

        # Gaussian profile
        elevation = height * np.exp(-0.5 * (distance / (width / 2.5)) ** 2)
        return elevation

    def _depression_elevation(
        self, position: np.ndarray, depression: dict[str, Any]
    ) -> float:
        """Compute elevation contribution from a depression."""
        center = depression["center"]
        radius = depression["radius"]
        depth = depression["depth"]

        distance = np.linalg.norm(position - center)

        if distance > radius:
            return 0.0

        # Parabolic profile
        normalized_dist = distance / radius
        elevation = -depth * (1 - normalized_dist**2)
        return elevation

    def calculate_break(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_samples: int = 20,
    ) -> dict[str, Any]:
        """Calculate break (lateral deviation) for a putt line.

        Args:
            start: Starting position [m, m]
            end: Target position [m, m]
            num_samples: Number of sample points along line

        Returns:
            Dictionary with break analysis
        """
        # Sample points along intended line
        t_values = np.linspace(0, 1, num_samples)
        positions = [start + t * (end - start) for t in t_values]

        slopes = [self.get_slope_at(p) for p in positions]

        # Perpendicular to putt direction
        putt_dir = end - start
        putt_len = np.linalg.norm(putt_dir)
        if putt_len < 1e-10:
            return {
                "total_break": 0.0,
                "break_direction": np.zeros(2),
                "average_slope": np.zeros(2),
            }

        putt_dir_norm = putt_dir / putt_len
        perp_dir = np.array([-putt_dir_norm[1], putt_dir_norm[0]])

        # Integrate cross-slope component
        total_break = 0.0
        for slope in slopes:
            cross_slope = np.dot(slope, perp_dir)
            total_break += cross_slope * (putt_len / num_samples)

        # Convert to actual break distance (approximation)
        # Break ≈ cross_slope * distance^2 / (4 * initial_velocity^2) * g
        avg_slope = np.mean(slopes, axis=0)
        break_magnitude = abs(total_break) * putt_len * 0.25

        break_direction = perp_dir * np.sign(total_break)

        return {
            "total_break": break_magnitude,
            "break_direction": break_direction,
            "average_slope": avg_slope,
            "cross_slopes": [np.dot(s, perp_dir) for s in slopes],
        }

    def read_putt_line(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_samples: int = 20,
    ) -> dict[str, Any]:
        """Read the putt line for elevations and slopes.

        Args:
            start: Starting position [m, m]
            end: Target position [m, m]
            num_samples: Number of sample points

        Returns:
            Dictionary with positions, elevations, and slopes along line
        """
        t_values = np.linspace(0, 1, num_samples)
        positions = [start + t * (end - start) for t in t_values]

        return {
            "positions": np.array(positions),
            "elevations": np.array([self.get_elevation_at(p) for p in positions]),
            "slopes": np.array([self.get_slope_at(p) for p in positions]),
            "distance": np.linalg.norm(end - start),
        }

    def to_heightmap(self, resolution: int = 100) -> np.ndarray:
        """Export surface as heightmap array.

        Args:
            resolution: Number of points in each dimension

        Returns:
            2D array of elevations [resolution x resolution]
        """
        x = np.linspace(0, self.width, resolution)
        y = np.linspace(0, self.height, resolution)

        heightmap = np.zeros((resolution, resolution))

        for i, yi in enumerate(y):
            for j, xi in enumerate(x):
                heightmap[i, j] = self.get_elevation_at(np.array([xi, yi]))

        return heightmap

    @classmethod
    def from_heightmap(
        cls,
        heightmap: np.ndarray,
        width: float,
        height: float,
        turf: TurfProperties | None = None,
    ) -> GreenSurface:
        """Create green surface from heightmap array.

        Args:
            heightmap: 2D array of elevations
            width: Physical width [m]
            height: Physical height [m]
            turf: Turf properties

        Returns:
            GreenSurface instance
        """
        green = cls(width=width, height=height, turf=turf)
        green.set_heightmap(heightmap)
        return green

    @classmethod
    def create_preset(cls, name: str) -> GreenSurface:
        """Create a preset green configuration.

        Available presets:
            - flat_practice: Flat practice green
            - undulating_championship: Championship undulating green
            - severe_slopes: Augusta-style severe slopes
            - tiered: Two-tier green with ridge

        Args:
            name: Name of preset

        Returns:
            Configured GreenSurface

        Raises:
            ValueError: If preset unknown
        """
        if name == "flat_practice":
            return cls(
                width=15.0,
                height=15.0,
                turf=TurfProperties.create_preset("practice_green"),
            )

        elif name == "undulating_championship":
            green = cls(
                width=25.0,
                height=25.0,
                turf=TurfProperties.create_preset("tournament_fast"),
            )
            # Add multiple subtle slopes
            green.add_slope_region(
                SlopeRegion(
                    center=np.array([8.0, 8.0]),
                    radius=6.0,
                    slope_direction=np.array([1.0, 0.5]),
                    slope_magnitude=0.02,
                )
            )
            green.add_slope_region(
                SlopeRegion(
                    center=np.array([17.0, 17.0]),
                    radius=5.0,
                    slope_direction=np.array([-0.5, 1.0]),
                    slope_magnitude=0.015,
                )
            )
            green.add_depression(
                center=np.array([12.0, 12.0]),
                radius=3.0,
                depth=0.02,
            )
            green.set_hole_position(np.array([15.0, 15.0]))
            return green

        elif name == "severe_slopes":
            green = cls(
                width=20.0,
                height=20.0,
                turf=TurfProperties.create_preset("augusta_like"),
            )
            # Add severe tier
            green.add_slope_region(
                SlopeRegion(
                    center=np.array([10.0, 10.0]),
                    radius=8.0,
                    slope_direction=np.array([1.0, 0.0]),
                    slope_magnitude=0.05,  # 5% slope
                )
            )
            green.add_ridge(
                start=np.array([5.0, 15.0]),
                end=np.array([15.0, 15.0]),
                height=0.05,
                width=2.0,
            )
            green.set_hole_position(np.array([15.0, 10.0]))
            return green

        elif name == "tiered":
            green = cls(
                width=20.0,
                height=20.0,
                turf=TurfProperties.create_preset("tournament_standard"),
            )
            # Create a tier with ridge
            green.add_ridge(
                start=np.array([0.0, 10.0]),
                end=np.array([20.0, 10.0]),
                height=0.08,
                width=3.0,
            )
            green.set_hole_position(np.array([15.0, 5.0]))
            return green

        else:
            raise ValueError(f"Unknown preset: {name}")

    def load_from_file(self, filepath: str | Path) -> None:
        """Load topographical data from file.

        Supports:
            - .npy: NumPy array
            - .csv: CSV with x,y,elevation columns
            - .json: JSON configuration
            - .tif/.tiff: GeoTIFF (requires rasterio)

        Args:
            filepath: Path to data file
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        if suffix == ".npy":
            heightmap = np.load(filepath)
            self.set_heightmap(heightmap)

        elif suffix == ".csv":
            self._load_csv_topography(filepath)

        elif suffix == ".json":
            self._load_json_topography(filepath)

        elif suffix in (".tif", ".tiff"):
            self._load_geotiff_topography(filepath)

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _load_csv_topography(self, filepath: Path) -> None:
        """Load topography from CSV file."""
        import csv

        points = []
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                points.append(
                    ContourPoint(
                        x=float(row.get("x", row.get("X", 0))),
                        y=float(row.get("y", row.get("Y", 0))),
                        elevation=float(
                            row.get("elevation", row.get("z", row.get("Z", 0)))
                        ),
                    )
                )

        self.set_contour_points(points)

    def _load_json_topography(self, filepath: Path) -> None:
        """Load topography from JSON file."""
        import json

        with open(filepath) as f:
            data = json.load(f)

        # Load contour points if present
        if "contours" in data:
            points = [
                ContourPoint(x=p["x"], y=p["y"], elevation=p["elevation"])
                for p in data["contours"]
            ]
            self.set_contour_points(points)

        # Load slope regions if present
        if "slopes" in data:
            for s in data["slopes"]:
                self.add_slope_region(
                    SlopeRegion(
                        center=np.array(s["center"]),
                        radius=s["radius"],
                        slope_direction=np.array(s["direction"]),
                        slope_magnitude=s["magnitude"],
                    )
                )

        # Load hole position if present
        if "hole_position" in data:
            self.set_hole_position(np.array(data["hole_position"]))

    def _load_geotiff_topography(self, filepath: Path) -> None:
        """Load topography from GeoTIFF file."""
        try:
            import rasterio  # type: ignore
        except ImportError as err:
            raise ImportError(
                "rasterio required for GeoTIFF support. Install with: pip install rasterio"
            ) from err

        with rasterio.open(filepath) as src:
            heightmap = src.read(1)
            self.set_heightmap(heightmap)
