"""Terrain loading and factory helpers."""

from __future__ import annotations

from pathlib import Path

from .terrain import (
    ElevationMap,
    Terrain,
    TerrainConfig,
    TerrainPatch,
    TerrainType,
)


def create_flat_terrain(
    name: str,
    width: float,
    length: float,
    terrain_type: TerrainType = TerrainType.FAIRWAY,
    resolution: float = 1.0,
) -> Terrain:
    """Create a simple flat terrain."""
    if not (name is not None):
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
    if not (name is not None):
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
