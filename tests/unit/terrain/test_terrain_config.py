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


class TestTerrainConfig:
    """Test terrain configuration loading/saving."""

    def test_terrain_config_to_dict(self) -> None:
        """Terrain config should be serializable to dict."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [TerrainPatch(TerrainType.FAIRWAY, 0.0, 100.0, 0.0, 100.0)]
        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        config = TerrainConfig.from_terrain(terrain)
        data = config.to_dict()

        assert "name" in data
        assert "elevation" in data
        assert "patches" in data
        assert data["name"] == "Test"

    def test_terrain_config_to_json(self, tmp_path: Path) -> None:
        """Terrain config should save to JSON."""
        elevation = ElevationMap.flat(width=100.0, length=100.0, resolution=1.0)
        patches = [TerrainPatch(TerrainType.FAIRWAY, 0.0, 100.0, 0.0, 100.0)]
        terrain = Terrain(name="Test", elevation=elevation, patches=patches)

        config = TerrainConfig.from_terrain(terrain)
        json_path = tmp_path / "terrain.json"
        config.save(json_path)

        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert data["name"] == "Test"

    def test_terrain_config_from_json(self, tmp_path: Path) -> None:
        """Terrain config should load from JSON."""
        # Create JSON file
        json_data = {
            "name": "LoadedTerrain",
            "elevation": {
                "type": "flat",
                "width": 100.0,
                "length": 100.0,
                "resolution": 1.0,
            },
            "patches": [
                {
                    "terrain_type": "fairway",
                    "x_min": 0.0,
                    "x_max": 100.0,
                    "y_min": 0.0,
                    "y_max": 100.0,
                }
            ],
        }
        json_path = tmp_path / "terrain.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        config = TerrainConfig.load(json_path)
        terrain = config.to_terrain()

        assert terrain.name == "LoadedTerrain"
        assert len(terrain.patches) == 1
