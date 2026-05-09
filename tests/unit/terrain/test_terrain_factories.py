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


class TestTerrainFactories:
    """Test factory functions for common terrain configurations."""

    def test_create_flat_terrain(self) -> None:
        """Create a simple flat terrain."""
        terrain = create_flat_terrain(
            name="Flat Test",
            width=100.0,
            length=200.0,
            terrain_type=TerrainType.FAIRWAY,
        )

        assert terrain.name == "Flat Test"
        assert terrain.elevation.width == 100.0
        assert terrain.elevation.length == 200.0
        assert np.allclose(terrain.elevation.data, 0.0)

    def test_create_sloped_terrain(self) -> None:
        """Create a uniformly sloped terrain."""
        terrain = create_sloped_terrain(
            name="Sloped Test",
            width=100.0,
            length=200.0,
            slope_angle_deg=3.0,
            slope_direction_deg=90.0,  # Slope in Y direction
            terrain_type=TerrainType.FAIRWAY,
        )

        # Grid: 200 nodes at 1m resolution → valid Y range [0, 199].
        h1 = terrain.elevation.get_elevation(50.0, 0.0)
        h2 = terrain.elevation.get_elevation(50.0, 199.0)
        assert h2 > h1

    def test_create_terrain_from_config(self, tmp_path: Path) -> None:
        """Create terrain from config file."""
        config_data = {
            "name": "ConfigTest",
            "elevation": {
                "type": "sloped",
                "width": 50.0,
                "length": 100.0,
                "resolution": 0.5,
                "slope_angle_deg": 2.0,
                "slope_direction_deg": 0.0,
            },
            "patches": [
                {
                    "terrain_type": "tee",
                    "x_min": 0.0,
                    "x_max": 10.0,
                    "y_min": 20.0,
                    "y_max": 30.0,
                },
                {
                    "terrain_type": "fairway",
                    "x_min": 10.0,
                    "x_max": 50.0,
                    "y_min": 0.0,
                    "y_max": 100.0,
                },
            ],
        }
        config_path = tmp_path / "terrain_config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        terrain = create_terrain_from_config(config_path)

        assert terrain.name == "ConfigTest"
        assert len(terrain.patches) == 2
