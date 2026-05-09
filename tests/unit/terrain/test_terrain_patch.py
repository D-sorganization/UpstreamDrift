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


class TestTerrainPatch:
    """Test individual terrain patches (regions with uniform properties)."""

    def test_terrain_patch_creation(self) -> None:
        """Create a terrain patch with specified properties."""
        patch = TerrainPatch(
            terrain_type=TerrainType.FAIRWAY,
            x_min=0.0,
            x_max=100.0,
            y_min=0.0,
            y_max=50.0,
        )

        assert patch.terrain_type == TerrainType.FAIRWAY
        assert patch.x_min == 0.0
        assert patch.x_max == 100.0
        assert patch.y_min == 0.0
        assert patch.y_max == 50.0

    def test_terrain_patch_contains(self) -> None:
        """Check if a point is within the patch."""
        patch = TerrainPatch(
            terrain_type=TerrainType.GREEN,
            x_min=50.0,
            x_max=70.0,
            y_min=20.0,
            y_max=40.0,
        )

        assert patch.contains(60.0, 30.0)
        assert not patch.contains(0.0, 0.0)
        assert not patch.contains(60.0, 50.0)

    def test_terrain_patch_with_custom_material(self) -> None:
        """Patch can override default material for its terrain type."""
        custom_material = SurfaceMaterial(
            name="wet_fairway",
            friction_coefficient=0.4,  # Lower due to moisture
            rolling_resistance=0.15,
            restitution=0.5,
            hardness=0.6,
            grass_height_m=0.02,
        )

        patch = TerrainPatch(
            terrain_type=TerrainType.FAIRWAY,
            x_min=0.0,
            x_max=100.0,
            y_min=0.0,
            y_max=50.0,
            material=custom_material,
        )

        assert patch.material is not None
        assert patch.material.name == "wet_fairway"
        assert patch.material.friction_coefficient == 0.4
