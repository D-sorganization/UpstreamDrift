"""Regression tests for the terrain compatibility facade.

These tests keep the legacy ``terrain_representation`` import path working while
the implementation lives in the decomposed ``physics/terrain`` package.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.physics import terrain_representation as legacy_terrain
from src.shared.python.physics.terrain import (
    ElevationMap,
    Terrain,
    TerrainType,
    create_flat_terrain,
)
from src.shared.python.physics.terrain_loading import create_flat_terrain as loader_flat
from src.shared.python.physics.terrain_physics import compute_roll_direction


def test_terrain_representation_reexports_package_symbols() -> None:
    """Legacy module path should point at the package implementation."""
    assert legacy_terrain.ElevationMap is ElevationMap
    assert legacy_terrain.Terrain is Terrain
    assert legacy_terrain.TerrainType is TerrainType
    assert legacy_terrain.create_flat_terrain is create_flat_terrain


def test_terrain_loading_and_physics_use_package_surface() -> None:
    """Helper modules should operate on the decomposed terrain package."""
    terrain = loader_flat("compat", 10.0, 10.0, terrain_type=TerrainType.FAIRWAY)

    assert isinstance(terrain, Terrain)
    assert terrain.get_terrain_type(5.0, 5.0) == TerrainType.FAIRWAY

    roll_dir = compute_roll_direction(terrain.elevation, 5.0, 5.0)
    assert roll_dir.shape == (2,)
    assert np.allclose(roll_dir, np.zeros(2))
