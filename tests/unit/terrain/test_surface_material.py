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


class TestSurfaceMaterial:
    """Test surface material properties."""

    def test_surface_material_creation(self) -> None:
        """Create a surface material with all properties."""
        material = SurfaceMaterial(
            name="fairway_grass",
            friction_coefficient=0.5,
            rolling_resistance=0.1,
            restitution=0.6,
            hardness=0.7,
            grass_height_m=0.015,
        )

        assert material.name == "fairway_grass"
        assert material.friction_coefficient == 0.5
        assert material.rolling_resistance == 0.1
        assert material.restitution == 0.6
        assert material.hardness == 0.7
        assert material.grass_height_m == 0.015

    def test_surface_material_default_values(self) -> None:
        """Surface material should have sensible defaults."""
        material = SurfaceMaterial(name="test")

        # Friction should be positive
        assert material.friction_coefficient > 0
        # Rolling resistance should be non-negative
        assert material.rolling_resistance >= 0
        # Restitution should be between 0 and 1
        assert 0 <= material.restitution <= 1
        # Hardness should be between 0 and 1
        assert 0 <= material.hardness <= 1
        # Grass height defaults to 0
        assert material.grass_height_m >= 0

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"friction_coefficient": -0.5}, "friction"),
            ({"restitution": 1.5}, "restitution"),
            ({"hardness": -0.1}, "hardness"),
        ],
        ids=["negative_friction", "restitution_over_1", "negative_hardness"],
    )
    def test_surface_material_validation(self, kwargs, match) -> None:
        """Invalid material properties should raise errors."""
        with pytest.raises(ValueError, match=match):
            SurfaceMaterial(name="invalid", **kwargs)

    @pytest.mark.parametrize(
        "material_name",
        ["fairway", "rough", "green", "bunker", "tee"],
        ids=["fairway", "rough", "green", "bunker", "tee"],
    )
    def test_predefined_material_exists(self, material_name) -> None:
        """Predefined materials for common terrain types should exist."""
        from src.shared.python.physics.terrain import MATERIALS

        assert material_name in MATERIALS

    def test_predefined_material_relationships(self) -> None:
        """Verify physical relationships between predefined materials."""
        from src.shared.python.physics.terrain import MATERIALS

        # Bunker should have higher friction (sand)
        assert (
            MATERIALS["bunker"].friction_coefficient
            > MATERIALS["fairway"].friction_coefficient
        )

        # Green should have lower grass
        assert MATERIALS["green"].grass_height_m < MATERIALS["fairway"].grass_height_m
