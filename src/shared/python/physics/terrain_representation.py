"""Compatibility facade for the decomposed terrain package.

The terrain implementation now lives under ``src.shared.python.physics.terrain``.
This module remains as a stable import path for older callers while keeping the
legacy surface byte-for-byte compatible at the API level.
"""

from __future__ import annotations

from .terrain import (
    MATERIALS,
    TERRAIN_MATERIAL_MAP,
    ElevationMap,
    SurfaceMaterial,
    Terrain,
    TerrainConfig,
    TerrainPatch,
    TerrainRegion,
    TerrainType,
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
