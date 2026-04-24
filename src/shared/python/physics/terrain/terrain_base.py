import functools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.physics_constants import GRAVITY_M_S2

from .elevation import ElevationMap
from .materials import MATERIALS, TERRAIN_MATERIAL_MAP, SurfaceMaterial, TerrainType
from .regions import TerrainPatch, TerrainRegion


@dataclass
class Terrain:
    """Complete terrain configuration."""

    name: str
    elevation: ElevationMap
    patches: list[TerrainPatch] = field(default_factory=list)
    regions: list[TerrainRegion] = field(default_factory=list)
    default_type: TerrainType = TerrainType.ROUGH

    def get_elevation(self, x: float, y: float) -> float:
        """Get interpolated elevation at a position."""
        if x is None:
            raise ValueError("x must be provided")
        return self.elevation.get_elevation(x, y)

    def get_normal(self, x: float, y: float) -> np.ndarray:
        """Get surface normal vector at a position."""
        if x is None:
            raise ValueError("x must be provided")
        return self.elevation.get_normal(x, y)

    def get_terrain_type(self, x: float, y: float) -> TerrainType:
        """Get terrain type at a position."""
        if x is None:
            raise ValueError("x must be provided")
        result = self.default_type

        for patch in self.patches:
            if patch.contains(x, y):
                result = patch.terrain_type

        for region in self.regions:
            if region.contains(x, y):
                result = region.terrain_type

        return result

    def get_material(self, x: float, y: float) -> SurfaceMaterial:
        """Get surface material at a position."""
        if x is None:
            raise ValueError("x must be provided")
        for region in reversed(self.regions):
            if region.contains(x, y):
                return region.get_material()

        for patch in reversed(self.patches):
            if patch.contains(x, y):
                return patch.get_material()

        material_name = TERRAIN_MATERIAL_MAP.get(self.default_type, "rough")
        return MATERIALS[material_name]

    def get_properties_at(self, x: float, y: float) -> dict[str, Any]:
        """Get all terrain properties at a position."""
        return {
            "elevation": self.elevation.get_elevation(x, y),
            "gradient": self.elevation.get_gradient(x, y),
            "normal": self.elevation.get_normal(x, y),
            "slope_angle": self.elevation.get_slope_angle(x, y),
            "terrain_type": self.get_terrain_type(x, y),
            "material": self.get_material(x, y),
        }

    def get_contact_params(self, x: float, y: float) -> dict[str, float]:
        """Get physics contact parameters for simulation engines."""
        if x is None:
            raise ValueError("x must be provided")
        material = self.get_material(x, y)

        base_stiffness = 1e5  # N/m
        stiffness = base_stiffness * material.hardness
        damping = 2.0 * math.sqrt(stiffness * 0.05)

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
    def from_terrain(cls, terrain: Terrain) -> "TerrainConfig":
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
    def from_dict(cls, data: dict[str, Any]) -> "TerrainConfig":
        """Create config from dictionary."""
        if data is None:
            raise ValueError("data must be provided")
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
    def load(cls, path: Path | str) -> "TerrainConfig":
        """Load config from JSON file."""
        if path is None:
            raise ValueError("path must be provided")
        with open(path) as f:
            data = json.load(f)

        elev_data = data.get("elevation", {})
        if "type" in elev_data:
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


def create_flat_terrain(
    name: str,
    width: float,
    length: float,
    terrain_type: TerrainType = TerrainType.FAIRWAY,
    resolution: float = 1.0,
) -> Terrain:
    """Create a simple flat terrain."""
    if name is None:
        raise ValueError("name must be provided")
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
    """Create a uniformly sloped terrain."""
    if name is None:
        raise ValueError("name must be provided")
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
    """Create terrain from configuration file."""
    config = TerrainConfig.load(config_path)
    return config.to_terrain()


@functools.lru_cache(maxsize=256)
def compute_gravity_on_slope(
    slope_angle_deg: float,
    gravity: float = float(GRAVITY_M_S2),
) -> tuple[float, float]:
    """Compute gravity components on a slope. Cached for performance."""
    if slope_angle_deg is None:
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
    """Compute ball roll direction on terrain (downhill)."""
    if elevation is None:
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
    """Get contact normal for physics engine."""
    return elevation.get_normal(x, y)
