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


class TestTerrain:
    """Test complete terrain configuration."""

    def test_terrain_creation(self) -> None:
        """Create a complete terrain with elevation and patches."""
        elevation = ElevationMap.flat(width=200.0, length=400.0, resolution=1.0)

        patches = [
            TerrainPatch(TerrainType.TEE, 10.0, 30.0, 190.0, 210.0),
            TerrainPatch(TerrainType.FAIRWAY, 30.0, 180.0, 150.0, 250.0),
            TerrainPatch(TerrainType.GREEN, 180.0, 200.0, 190.0, 210.0),
        ]

        terrain = Terrain(
            name="Test Hole",
            elevation=elevation,
            patches=patches,
        )

        assert terrain.name == "Test Hole"
        assert terrain.elevation.width == 200.0
        assert len(terrain.patches) == 3

    def test_terrain_type_at_point(self) -> None:
        """Query terrain type at a given position."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [
            TerrainPatch(TerrainType.FAIRWAY, 0.0, 100.0, 0.0, 100.0),
            TerrainPatch(
                TerrainType.GREEN, 80.0, 100.0, 40.0, 60.0
            ),  # Overlaps fairway
        ]

        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        # Green patch should take priority (defined last)
        assert terrain.get_terrain_type(90.0, 50.0) == TerrainType.GREEN

        # Fairway elsewhere
        assert terrain.get_terrain_type(50.0, 50.0) == TerrainType.FAIRWAY

    @pytest.mark.parametrize(
        "prop_key",
        ["elevation", "gradient", "normal", "terrain_type", "material"],
        ids=["elevation", "gradient", "normal", "terrain_type", "material"],
    )
    def test_terrain_properties_at_point(self, prop_key) -> None:
        """Get all terrain properties at a point."""
        elevation = ElevationMap.sloped(
            width=100.0,
            length=100.0,
            resolution=1.0,
            slope_angle_deg=5.0,
            slope_direction_deg=0.0,
        )
        patches = [TerrainPatch(TerrainType.FAIRWAY, 0.0, 100.0, 0.0, 100.0)]

        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        props = terrain.get_properties_at(50.0, 50.0)
        assert prop_key in props

    @pytest.mark.parametrize(
        "key",
        ["friction", "restitution", "stiffness", "damping"],
        ids=["friction", "restitution", "stiffness", "damping"],
    )
    def test_terrain_contact_parameter_keys(self, key) -> None:
        """Get physics-engine-ready contact parameters contain expected keys."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [TerrainPatch(TerrainType.BUNKER, 0.0, 100.0, 0.0, 100.0)]

        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        contact = terrain.get_contact_params(50.0, 50.0)
        assert key in contact

    def test_terrain_contact_bunker_friction(self) -> None:
        """Bunker should have higher friction."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [TerrainPatch(TerrainType.BUNKER, 0.0, 100.0, 0.0, 100.0)]

        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        contact = terrain.get_contact_params(50.0, 50.0)
        assert contact["friction"] > 0.5
