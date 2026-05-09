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


class TestTerrainType:
    """Test terrain type enumeration."""

    @pytest.mark.parametrize(
        "terrain_type",
        [
            TerrainType.FAIRWAY,
            TerrainType.ROUGH,
            TerrainType.GREEN,
            TerrainType.BUNKER,
            TerrainType.TEE,
            TerrainType.FRINGE,
            TerrainType.WATER,
            TerrainType.CART_PATH,
        ],
        ids=[
            "fairway",
            "rough",
            "green",
            "bunker",
            "tee",
            "fringe",
            "water",
            "cart_path",
        ],
    )
    def test_terrain_types_exist(self, terrain_type) -> None:
        """Verify expected terrain type is defined."""
        assert terrain_type is not None

    def test_terrain_type_values_unique(self) -> None:
        """Each terrain type should have a unique value."""
        values = [t.value for t in TerrainType]
        assert len(values) == len(set(values))
