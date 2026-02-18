"""Terrain modeling system for golf simulation.

Provides comprehensive terrain features including:
- Elevation maps with slopes and uneven surfaces
- Multiple terrain types (fairway, rough, green, bunker, etc.)
- Surface material properties (friction, restitution, hardness)
- Physics integration for all simulation engines

Design by Contract:
    Preconditions:
        - Terrain dimensions must be positive
        - Resolution must be positive
        - Slope angles must be valid (-90 to 90 degrees)

    Postconditions:
        - Elevation queries return valid heights
        - Normal vectors are unit vectors
        - Contact parameters are physically valid

    Invariants:
        - Elevation data shape matches dimensions/resolution
        - All materials have valid physics properties
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.physics_constants import GRAVITY_M_S2
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class TerrainType(Enum):
    """Golf course terrain types."""

    FAIRWAY = auto()
    ROUGH = auto()
    GREEN = auto()
    BUNKER = auto()
    TEE = auto()
    FRINGE = auto()
    WATER = auto()
    CART_PATH = auto()
    OUT_OF_BOUNDS = auto()


@dataclass
class SurfaceMaterial:
    """Physical properties of a surface material.

    Attributes:
        name: Identifier for this material
        friction_coefficient: Coulomb friction coefficient (mu)
        rolling_resistance: Rolling resistance coefficient
        restitution: Coefficient of restitution (bounce)
        hardness: Surface hardness (0=soft, 1=hard)
        grass_height_m: Height of grass in meters
        compressibility: Surface compressibility (0=rigid, 1=very soft)
        compression_damping: Damping ratio for compression (0-1)
        turf_density: Turf/grass density affecting resistance (kg/m^3)
        moisture_content: Moisture level affecting properties (0=dry, 1=saturated)
    """

    name: str
    friction_coefficient: float = 0.5
    rolling_resistance: float = 0.1
    restitution: float = 0.6
    hardness: float = 0.7
    grass_height_m: float = 0.0
    compressibility: float = 0.0
    compression_damping: float = 0.3
    turf_density: float = 0.0
    moisture_content: float = 0.3

    def __post_init__(self) -> None:
        """Validate material properties."""
        if self.friction_coefficient < 0:
            raise ValueError("friction_coefficient must be non-negative")
        if self.rolling_resistance < 0:
            raise ValueError("rolling_resistance must be non-negative")
        if not 0 <= self.restitution <= 1:
            raise ValueError("restitution must be between 0 and 1")
        if not 0 <= self.hardness <= 1:
            raise ValueError("hardness must be between 0 and 1")
        if self.grass_height_m < 0:
            raise ValueError("grass_height_m must be non-negative")
        if not 0 <= self.compressibility <= 1:
            raise ValueError("compressibility must be between 0 and 1")
        if not 0 <= self.compression_damping <= 1:
            raise ValueError("compression_damping must be between 0 and 1")
        if not 0 <= self.moisture_content <= 1:
            raise ValueError("moisture_content must be between 0 and 1")

    @property
    def is_compressible(self) -> bool:
        """Check if this material is compressible."""
        return self.compressibility > 0.01

    def get_effective_stiffness(self, base_stiffness: float = 1e5) -> float:
        """Get effective stiffness considering compressibility.

        Args:
            base_stiffness: Base stiffness for rigid surfaces (N/m)

        Returns:
            Effective stiffness (N/m)
        """
        # Higher compressibility = lower stiffness
        return base_stiffness * (1.0 - 0.9 * self.compressibility)

    def get_max_compression_depth(self) -> float:
        """Get maximum compression depth in meters.

        Returns:
            Maximum compression depth based on grass height and compressibility
        """
        # Turf can compress up to 80% of grass height
        base_depth = self.grass_height_m * 0.8 * self.compressibility
        # Moisture increases compression
        moisture_factor = 1.0 + 0.5 * self.moisture_content
        return base_depth * moisture_factor


# Predefined materials for common terrain types
MATERIALS: dict[str, SurfaceMaterial] = {
    "fairway": SurfaceMaterial(
        name="fairway",
        friction_coefficient=0.45,
        rolling_resistance=0.08,
        restitution=0.65,
        hardness=0.75,
        grass_height_m=0.015,
        compressibility=0.15,  # Slight compression
        compression_damping=0.25,
        turf_density=120.0,  # kg/m^3
        moisture_content=0.3,
    ),
    "rough": SurfaceMaterial(
        name="rough",
        friction_coefficient=0.55,
        rolling_resistance=0.20,
        restitution=0.45,
        hardness=0.65,
        grass_height_m=0.050,
        compressibility=0.35,  # More compressible
        compression_damping=0.40,
        turf_density=80.0,
        moisture_content=0.35,
    ),
    "green": SurfaceMaterial(
        name="green",
        friction_coefficient=0.35,
        rolling_resistance=0.05,
        restitution=0.70,
        hardness=0.80,
        grass_height_m=0.004,
        compressibility=0.05,  # Very firm
        compression_damping=0.15,
        turf_density=200.0,
        moisture_content=0.25,
    ),
    "bunker": SurfaceMaterial(
        name="bunker",
        friction_coefficient=0.80,
        rolling_resistance=0.40,
        restitution=0.30,
        hardness=0.30,
        grass_height_m=0.0,
        compressibility=0.70,  # Sand is highly compressible
        compression_damping=0.60,
        turf_density=1500.0,  # Sand density
        moisture_content=0.10,
    ),
    "tee": SurfaceMaterial(
        name="tee",
        friction_coefficient=0.45,
        rolling_resistance=0.08,
        restitution=0.65,
        hardness=0.80,
        grass_height_m=0.010,
        compressibility=0.10,  # Firm, well-maintained
        compression_damping=0.20,
        turf_density=150.0,
        moisture_content=0.25,
    ),
    "fringe": SurfaceMaterial(
        name="fringe",
        friction_coefficient=0.42,
        rolling_resistance=0.10,
        restitution=0.60,
        hardness=0.75,
        grass_height_m=0.012,
        compressibility=0.12,
        compression_damping=0.22,
        turf_density=140.0,
        moisture_content=0.28,
    ),
    "cart_path": SurfaceMaterial(
        name="cart_path",
        friction_coefficient=0.70,
        rolling_resistance=0.02,
        restitution=0.80,
        hardness=0.95,
        grass_height_m=0.0,
        compressibility=0.0,  # Rigid surface
        compression_damping=0.0,
        turf_density=0.0,
        moisture_content=0.0,
    ),
    "water": SurfaceMaterial(
        name="water",
        friction_coefficient=0.01,
        rolling_resistance=0.90,
        restitution=0.10,
        hardness=0.0,
        grass_height_m=0.0,
        compressibility=1.0,  # Water compresses (ball sinks)
        compression_damping=0.90,
        turf_density=1000.0,  # Water density
        moisture_content=1.0,
    ),
    # New compressible turf materials
    "soft_turf": SurfaceMaterial(
        name="soft_turf",
        friction_coefficient=0.50,
        rolling_resistance=0.15,
        restitution=0.50,
        hardness=0.50,
        grass_height_m=0.025,
        compressibility=0.45,  # Notably compressible
        compression_damping=0.45,
        turf_density=100.0,
        moisture_content=0.45,
    ),
    "wet_fairway": SurfaceMaterial(
        name="wet_fairway",
        friction_coefficient=0.35,
        rolling_resistance=0.12,
        restitution=0.55,
        hardness=0.60,
        grass_height_m=0.015,
        compressibility=0.30,  # Wet ground compresses more
        compression_damping=0.35,
        turf_density=120.0,
        moisture_content=0.70,
    ),
    "divot": SurfaceMaterial(
        name="divot",
        friction_coefficient=0.60,
        rolling_resistance=0.25,
        restitution=0.40,
        hardness=0.40,
        grass_height_m=0.005,  # Damaged turf
        compressibility=0.50,  # Loose soil
        compression_damping=0.50,
        turf_density=90.0,
        moisture_content=0.35,
    ),
}

# Mapping from terrain type to default material
TERRAIN_MATERIAL_MAP: dict[TerrainType, str] = {
    TerrainType.FAIRWAY: "fairway",
    TerrainType.ROUGH: "rough",
    TerrainType.GREEN: "green",
    TerrainType.BUNKER: "bunker",
    TerrainType.TEE: "tee",
    TerrainType.FRINGE: "fringe",
    TerrainType.CART_PATH: "cart_path",
    TerrainType.WATER: "water",
    TerrainType.OUT_OF_BOUNDS: "rough",
}


@dataclass
class ElevationMap:
    """Height map for terrain elevation.

    Stores elevation data on a regular grid and provides interpolated
    queries for arbitrary positions.

    Attributes:
        data: 2D array of elevation values (rows=Y, cols=X)
        resolution: Grid spacing in meters
        width: Total width in X direction (meters)
        length: Total length in Y direction (meters)
        origin_x: X coordinate of grid origin
        origin_y: Y coordinate of grid origin
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
    ) -> ElevationMap:
        """Create a flat elevation map.

        Args:
            width: Width in X direction (meters)
            length: Length in Y direction (meters)
            resolution: Grid spacing (meters)
            base_elevation: Constant elevation value (meters)

        Returns:
            Flat elevation map
        """
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
    ) -> ElevationMap:
        """Create a uniformly sloped elevation map.

        Args:
            width: Width in X direction (meters)
            length: Length in Y direction (meters)
            resolution: Grid spacing (meters)
            slope_angle_deg: Slope angle in degrees (positive = uphill)
            slope_direction_deg: Direction of uphill slope (0=+X, 90=+Y)
            base_elevation: Elevation at origin (meters)

        Returns:
            Sloped elevation map
        """
        if width <= 0 or length <= 0:
            raise ValueError("Width and length must be positive")
        if resolution <= 0:
            raise ValueError("Resolution must be positive")
        if abs(slope_angle_deg) > 89:
            logger.warning(f"Steep slope angle: {slope_angle_deg} degrees")

        n_cols = int(width / resolution)
        n_rows = int(length / resolution)

        # Create coordinate grids
        x = np.arange(n_cols) * resolution
        y = np.arange(n_rows) * resolution
        X, Y = np.meshgrid(x, y)

        # Calculate elevation based on slope
        slope_rad = math.radians(slope_angle_deg)
        dir_rad = math.radians(slope_direction_deg)

        # Slope gradient components
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
    ) -> ElevationMap:
        """Create elevation map from numpy array.

        Args:
            data: 2D array of elevation values (rows=Y, cols=X)
            resolution: Grid spacing (meters)
            origin_x: X coordinate of grid origin
            origin_y: Y coordinate of grid origin

        Returns:
            Elevation map from array data
        """
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
        """Get interpolated elevation at a point.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Elevation at the point (meters)
        """
        self._check_bounds(x, y)

        gx, gy = self._to_grid_coords(x, y)

        # Bilinear interpolation
        n_rows, n_cols = self.data.shape

        # Clamp to valid grid indices
        gx = max(0, min(gx, n_cols - 1))
        gy = max(0, min(gy, n_rows - 1))

        # Get integer and fractional parts
        ix = int(gx)
        iy = int(gy)
        fx = gx - ix
        fy = gy - iy

        # Clamp indices for edge cases
        ix1 = min(ix + 1, n_cols - 1)
        iy1 = min(iy + 1, n_rows - 1)

        # Bilinear interpolation
        h00 = self.data[iy, ix]
        h10 = self.data[iy, ix1]
        h01 = self.data[iy1, ix]
        h11 = self.data[iy1, ix1]

        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx
        h = h0 * (1 - fy) + h1 * fy

        return float(h)

    def get_gradient(self, x: float, y: float) -> tuple[float, float]:
        """Get elevation gradient (slope) at a point.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Tuple of (dz/dx, dz/dy) gradient components
        """
        self._check_bounds(x, y)

        gx, gy = self._to_grid_coords(x, y)
        n_rows, n_cols = self.data.shape

        ix = int(max(0, min(gx, n_cols - 1)))
        iy = int(max(0, min(gy, n_rows - 1)))

        # Central differences where possible
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
        """Get surface normal vector at a point.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Unit normal vector (3,)
        """
        dzdx, dzdy = self.get_gradient(x, y)

        # Normal from gradient: n = (-dz/dx, -dz/dy, 1) normalized
        normal = np.array([-dzdx, -dzdy, 1.0])
        normal = normal / np.linalg.norm(normal)

        return normal

    def get_slope_angle(self, x: float, y: float) -> float:
        """Get slope angle at a point.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Slope angle in degrees
        """
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
    def from_dict(cls, data: dict[str, Any]) -> ElevationMap:
        """Create elevation map from dictionary."""
        return cls(
            data=np.array(data["data"], dtype=np.float64),
            resolution=data["resolution"],
            width=data["width"],
            length=data["length"],
            origin_x=data.get("origin_x", 0.0),
            origin_y=data.get("origin_y", 0.0),
        )


@dataclass
class TerrainPatch:
    """A rectangular region with uniform terrain type.

    Attributes:
        terrain_type: Type of terrain in this patch
        x_min: Minimum X coordinate
        x_max: Maximum X coordinate
        y_min: Minimum Y coordinate
        y_max: Maximum Y coordinate
        material: Optional custom material (overrides default for terrain type)
    """

    terrain_type: TerrainType
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    material: SurfaceMaterial | None = None

    def contains(self, x: float, y: float) -> bool:
        """Check if a point is within this patch."""
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def get_material(self) -> SurfaceMaterial:
        """Get the material for this patch."""
        if self.material is not None:
            return self.material
        material_name = TERRAIN_MATERIAL_MAP.get(self.terrain_type, "rough")
        return MATERIALS[material_name]

    def to_dict(self) -> dict[str, Any]:
        """Serialize patch to dictionary."""
        result = {
            "terrain_type": self.terrain_type.name.lower(),
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }
        if self.material is not None:
            result["material"] = {
                "name": self.material.name,
                "friction_coefficient": self.material.friction_coefficient,
                "rolling_resistance": self.material.rolling_resistance,
                "restitution": self.material.restitution,
                "hardness": self.material.hardness,
                "grass_height_m": self.material.grass_height_m,
            }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TerrainPatch:
        """Create patch from dictionary."""
        terrain_type = TerrainType[data["terrain_type"].upper()]
        material = None
        if "material" in data:
            material = SurfaceMaterial(**data["material"])
        return cls(
            terrain_type=terrain_type,
            x_min=data["x_min"],
            x_max=data["x_max"],
            y_min=data["y_min"],
            y_max=data["y_max"],
            material=material,
        )


@dataclass
class TerrainRegion:
    """A terrain region with complex shape (circle, polygon, etc.).

    Attributes:
        terrain_type: Type of terrain in this region
        shape_type: Type of shape ('circle', 'polygon')
        shape_data: Shape-specific parameters
        material: Optional custom material
    """

    terrain_type: TerrainType
    shape_type: str
    shape_data: dict[str, Any]
    material: SurfaceMaterial | None = None

    @classmethod
    def circle(
        cls,
        terrain_type: TerrainType,
        center_x: float,
        center_y: float,
        radius: float,
        material: SurfaceMaterial | None = None,
    ) -> TerrainRegion:
        """Create a circular terrain region."""
        return cls(
            terrain_type=terrain_type,
            shape_type="circle",
            shape_data={"center_x": center_x, "center_y": center_y, "radius": radius},
            material=material,
        )

    @classmethod
    def polygon(
        cls,
        terrain_type: TerrainType,
        vertices: list[tuple[float, float]],
        material: SurfaceMaterial | None = None,
    ) -> TerrainRegion:
        """Create a polygon terrain region."""
        return cls(
            terrain_type=terrain_type,
            shape_type="polygon",
            shape_data={"vertices": vertices},
            material=material,
        )

    def contains(self, x: float, y: float) -> bool:
        """Check if a point is within this region."""
        if self.shape_type == "circle":
            cx = self.shape_data["center_x"]
            cy = self.shape_data["center_y"]
            r = self.shape_data["radius"]
            return (x - cx) ** 2 + (y - cy) ** 2 <= r**2

        elif self.shape_type == "polygon":
            vertices = self.shape_data["vertices"]
            return self._point_in_polygon(x, y, vertices)

        return False

    @staticmethod
    def _point_in_polygon(
        x: float, y: float, vertices: list[tuple[float, float]]
    ) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(vertices)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]

            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi
            ):
                inside = not inside
            j = i

        return inside

    def get_material(self) -> SurfaceMaterial:
        """Get the material for this region."""
        if self.material is not None:
            return self.material
        material_name = TERRAIN_MATERIAL_MAP.get(self.terrain_type, "rough")
        return MATERIALS[material_name]

    def to_dict(self) -> dict[str, Any]:
        """Serialize region to dictionary."""
        result: dict[str, Any] = {
            "terrain_type": self.terrain_type.name.lower(),
            "shape_type": self.shape_type,
            "shape_data": self.shape_data,
        }
        if self.material is not None:
            result["material"] = {
                "name": self.material.name,
                "friction": self.material.friction_coefficient,
                "restitution": self.material.restitution,
            }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TerrainRegion:
        """Deserialize region from dictionary."""
        material = None
        if "material" in data:
            mat = data["material"]
            material = SurfaceMaterial(
                name=mat["name"],
                friction_coefficient=mat["friction"],
                restitution=mat["restitution"],
            )
        return cls(
            terrain_type=TerrainType[data["terrain_type"].upper()],
            shape_type=data["shape_type"],
            shape_data=data["shape_data"],
            material=material,
        )


@dataclass
class Terrain:
    """Complete terrain configuration.

    Combines elevation map with terrain type regions to define
    a complete playing surface.

    Attributes:
        name: Terrain identifier
        elevation: Elevation/height map
        patches: List of rectangular terrain patches
        regions: List of complex-shaped terrain regions
        default_type: Default terrain type for uncovered areas
    """

    name: str
    elevation: ElevationMap
    patches: list[TerrainPatch] = field(default_factory=list)
    regions: list[TerrainRegion] = field(default_factory=list)
    default_type: TerrainType = TerrainType.ROUGH

    def get_elevation(self, x: float, y: float) -> float:
        """Get interpolated elevation at a position (delegate to elevation map).

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Elevation at the point (meters)
        """
        return self.elevation.get_elevation(x, y)

    def get_normal(self, x: float, y: float) -> np.ndarray:
        """Get surface normal vector at a position (delegate to elevation map).

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Unit normal vector (3,)
        """
        return self.elevation.get_normal(x, y)

    def get_terrain_type(self, x: float, y: float) -> TerrainType:
        """Get terrain type at a position.

        Regions and patches defined later take priority.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Terrain type at the position
        """
        result = self.default_type

        # Check patches (later patches override earlier)
        for patch in self.patches:
            if patch.contains(x, y):
                result = patch.terrain_type

        # Check regions (later regions override)
        for region in self.regions:
            if region.contains(x, y):
                result = region.terrain_type

        return result

    def get_material(self, x: float, y: float) -> SurfaceMaterial:
        """Get surface material at a position."""
        # Check regions first (they override patches)
        for region in reversed(self.regions):
            if region.contains(x, y):
                return region.get_material()

        # Check patches
        for patch in reversed(self.patches):
            if patch.contains(x, y):
                return patch.get_material()

        # Default
        material_name = TERRAIN_MATERIAL_MAP.get(self.default_type, "rough")
        return MATERIALS[material_name]

    def get_properties_at(self, x: float, y: float) -> dict[str, Any]:
        """Get all terrain properties at a position.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Dictionary with elevation, gradient, normal, terrain_type, material
        """
        return {
            "elevation": self.elevation.get_elevation(x, y),
            "gradient": self.elevation.get_gradient(x, y),
            "normal": self.elevation.get_normal(x, y),
            "slope_angle": self.elevation.get_slope_angle(x, y),
            "terrain_type": self.get_terrain_type(x, y),
            "material": self.get_material(x, y),
        }

    def get_contact_params(self, x: float, y: float) -> dict[str, float]:
        """Get physics contact parameters for simulation engines.

        Args:
            x: X coordinate (meters)
            y: Y coordinate (meters)

        Returns:
            Dictionary with friction, restitution, stiffness, damping
        """
        material = self.get_material(x, y)

        # Calculate stiffness and damping from material properties
        # These are tuned for typical physics engine contact models
        base_stiffness = 1e5  # N/m
        stiffness = base_stiffness * material.hardness
        damping = 2.0 * math.sqrt(stiffness * 0.05)  # Critical damping factor

        return {
            "friction": material.friction_coefficient,
            "restitution": material.restitution,
            "stiffness": stiffness,
            "damping": damping,
            "rolling_resistance": material.rolling_resistance,
        }


@dataclass
class TerrainConfig:
    """Configuration for terrain serialization/deserialization."""

    name: str
    elevation_config: dict[str, Any]
    patches_config: list[dict[str, Any]]
    regions_config: list[dict[str, Any]] = field(default_factory=list)
    default_type: str = "rough"

    @classmethod
    def from_terrain(cls, terrain: Terrain) -> TerrainConfig:
        """Create config from terrain object."""
        return cls(
            name=terrain.name,
            elevation_config=terrain.elevation.to_dict(),
            patches_config=[p.to_dict() for p in terrain.patches],
            regions_config=[r.to_dict() for r in terrain.regions],
            default_type=terrain.default_type.name.lower(),
        )

    def to_terrain(self) -> Terrain:
        """Create terrain from config."""
        elevation = ElevationMap.from_dict(self.elevation_config)
        patches = [TerrainPatch.from_dict(p) for p in self.patches_config]
        regions = [TerrainRegion.from_dict(r) for r in self.regions_config]
        default_type = TerrainType[self.default_type.upper()]

        return Terrain(
            name=self.name,
            elevation=elevation,
            patches=patches,
            regions=regions,
            default_type=default_type,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "name": self.name,
            "elevation": self.elevation_config,
            "patches": self.patches_config,
            "regions": self.regions_config,
            "default_type": self.default_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TerrainConfig:
        """Create config from dictionary."""
        return cls(
            name=data["name"],
            elevation_config=data["elevation"],
            patches_config=data.get("patches", []),
            regions_config=data.get("regions", []),
            default_type=data.get("default_type", "rough"),
        )

    def save(self, path: Path | str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> TerrainConfig:
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Handle elevation config format
        elev_data = data.get("elevation", {})
        if "type" in elev_data:
            # Simplified format - convert to full format
            elev_type = elev_data["type"]
            if elev_type == "flat":
                elev_map = ElevationMap.flat(
                    width=elev_data["width"],
                    length=elev_data["length"],
                    resolution=elev_data["resolution"],
                )
            elif elev_type == "sloped":
                elev_map = ElevationMap.sloped(
                    width=elev_data["width"],
                    length=elev_data["length"],
                    resolution=elev_data["resolution"],
                    slope_angle_deg=elev_data.get("slope_angle_deg", 0.0),
                    slope_direction_deg=elev_data.get("slope_direction_deg", 0.0),
                )
            else:
                raise ValueError(f"Unknown elevation type: {elev_type}")
            data["elevation"] = elev_map.to_dict()

        return cls.from_dict(data)


# Factory functions


def create_flat_terrain(
    name: str,
    width: float,
    length: float,
    terrain_type: TerrainType = TerrainType.FAIRWAY,
    resolution: float = 1.0,
) -> Terrain:
    """Create a simple flat terrain.

    Args:
        name: Terrain identifier
        width: Width in X direction (meters)
        length: Length in Y direction (meters)
        terrain_type: Terrain type for the entire surface
        resolution: Grid resolution (meters)

    Returns:
        Flat terrain
    """
    elevation = ElevationMap.flat(width=width, length=length, resolution=resolution)
    patches = [TerrainPatch(terrain_type, 0.0, width, 0.0, length)]

    return Terrain(name=name, elevation=elevation, patches=patches)


def create_sloped_terrain(
    name: str,
    width: float,
    length: float,
    slope_angle_deg: float,
    slope_direction_deg: float,
    terrain_type: TerrainType = TerrainType.FAIRWAY,
    resolution: float = 1.0,
) -> Terrain:
    """Create a uniformly sloped terrain.

    Args:
        name: Terrain identifier
        width: Width in X direction (meters)
        length: Length in Y direction (meters)
        slope_angle_deg: Slope angle in degrees
        slope_direction_deg: Direction of uphill slope (0=+X, 90=+Y)
        terrain_type: Terrain type for the entire surface
        resolution: Grid resolution (meters)

    Returns:
        Sloped terrain
    """
    elevation = ElevationMap.sloped(
        width=width,
        length=length,
        resolution=resolution,
        slope_angle_deg=slope_angle_deg,
        slope_direction_deg=slope_direction_deg,
    )
    patches = [TerrainPatch(terrain_type, 0.0, width, 0.0, length)]

    return Terrain(name=name, elevation=elevation, patches=patches)


def create_terrain_from_config(config_path: Path | str) -> Terrain:
    """Create terrain from configuration file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Configured terrain
    """
    config = TerrainConfig.load(config_path)
    return config.to_terrain()


# Physics integration functions


@functools.lru_cache(maxsize=256)
def compute_gravity_on_slope(
    slope_angle_deg: float,
    gravity: float = float(GRAVITY_M_S2),
) -> tuple[float, float]:
    """Compute gravity components on a slope. Cached for performance.

    Args:
        slope_angle_deg: Slope angle in degrees
        gravity: Gravitational acceleration (m/s^2)

    Returns:
        Tuple of (g_parallel, g_perpendicular) components
    """
    slope_rad = math.radians(slope_angle_deg)
    g_parallel = gravity * math.sin(slope_rad)
    g_perpendicular = gravity * math.cos(slope_rad)

    return g_parallel, g_perpendicular


def compute_roll_direction(
    elevation: ElevationMap,
    x: float,
    y: float,
) -> np.ndarray:
    """Compute ball roll direction on terrain (downhill).

    Args:
        elevation: Elevation map
        x: X coordinate (meters)
        y: Y coordinate (meters)

    Returns:
        Unit vector in roll direction (2D: x, y)
    """
    dzdx, dzdy = elevation.get_gradient(x, y)

    # Roll direction is opposite to gradient (downhill)
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
    """Get contact normal for physics engine.

    Args:
        elevation: Elevation map
        x: X coordinate (meters)
        y: Y coordinate (meters)

    Returns:
        Unit normal vector (3,)
    """
    return elevation.get_normal(x, y)
