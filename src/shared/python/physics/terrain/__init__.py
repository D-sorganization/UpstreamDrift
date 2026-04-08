"""Terrain package facade for shared elevation, material, and region models."""

from .elevation import ElevationMap
from .materials import MATERIALS, TERRAIN_MATERIAL_MAP, SurfaceMaterial, TerrainType
from .regions import TerrainPatch, TerrainRegion
from .terrain_base import (
    Terrain,
    TerrainConfig,
    compute_gravity_on_slope,
    compute_roll_direction,
    create_flat_terrain,
    create_sloped_terrain,
    create_terrain_from_config,
    get_contact_normal,
)

__all__ = [
    "TerrainType",
    "SurfaceMaterial",
    "MATERIALS",
    "TERRAIN_MATERIAL_MAP",
    "ElevationMap",
    "TerrainPatch",
    "TerrainRegion",
    "Terrain",
    "TerrainConfig",
    "create_flat_terrain",
    "create_sloped_terrain",
    "create_terrain_from_config",
    "compute_gravity_on_slope",
    "compute_roll_direction",
    "get_contact_normal",
]
