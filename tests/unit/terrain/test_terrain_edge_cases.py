"""Unit tests for terrain modeling system (TDD: Tests First).

Following the Pragmatic Programmer principles:
- DRY (Don't Repeat Yourself)
- Design by Contract
- Orthogonality
- Test-Driven Development

These tests define the expected behavior BEFORE implementation.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

# Tests written first - imports will fail until implementation exists
from src.shared.python.physics.terrain import (
    ElevationMap,
    SurfaceMaterial,
    Terrain,
    TerrainConfig,
    TerrainPatch,
    TerrainRegion,
    TerrainType,
    create_flat_terrain,
    create_sloped_terrain,
    create_terrain_from_config,
)


class TestTerrainEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        "width,length,resolution",
        [
            (0.0, 100.0, 1.0),
            (100.0, 100.0, -1.0),
        ],
        ids=["zero_width", "negative_resolution"],
    )
    def test_elevation_invalid_params(self, width, length, resolution) -> None:
        """Invalid elevation map parameters should raise errors."""
        with pytest.raises(ValueError):
            ElevationMap.flat(width=width, length=length, resolution=resolution)

    def test_steep_slope(self) -> None:
        """Very steep slopes (>45°) should work but warn."""
        elev = ElevationMap.sloped(
            width=100.0,
            length=100.0,
            resolution=1.0,
            slope_angle_deg=60.0,
            slope_direction_deg=0.0,
        )

        # Should still work
        assert elev.get_elevation(50, 50) > 0

    def test_multiple_overlapping_patches(self) -> None:
        """Later patches should override earlier ones."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [
            TerrainPatch(TerrainType.ROUGH, 0.0, 100.0, 0.0, 100.0),
            TerrainPatch(TerrainType.FAIRWAY, 20.0, 80.0, 20.0, 80.0),
            TerrainPatch(TerrainType.GREEN, 60.0, 80.0, 40.0, 60.0),
        ]

        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        # Check priority (last defined wins)
        assert terrain.get_terrain_type(70.0, 50.0) == TerrainType.GREEN
        assert terrain.get_terrain_type(50.0, 50.0) == TerrainType.FAIRWAY
        assert terrain.get_terrain_type(10.0, 10.0) == TerrainType.ROUGH

    def test_point_outside_all_patches(self) -> None:
        """Point outside all patches returns default terrain type."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [TerrainPatch(TerrainType.GREEN, 40.0, 60.0, 40.0, 60.0)]

        terrain = Terrain(
            name="Test",
            elevation=elevation,
            patches=patches,
            default_type=TerrainType.ROUGH,
        )

        assert terrain.get_terrain_type(10.0, 10.0) == TerrainType.ROUGH
