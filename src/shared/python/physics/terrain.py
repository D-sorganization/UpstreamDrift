"""Thin facade for terrain representation, loading, and physics helpers."""

from __future__ import annotations

from .terrain_loading import (
    create_flat_terrain,
    create_sloped_terrain,
    create_terrain_from_config,
)
from .terrain_physics import (
    compute_gravity_on_slope,
    compute_roll_direction,
    get_contact_normal,
)
from .terrain_representation import (
    MATERIALS,
    TERRAIN_MATERIAL_MAP,
    ElevationMap,
    SurfaceMaterial,
    Terrain,
    TerrainConfig,
    TerrainPatch,
    TerrainRegion,
    TerrainType,
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
