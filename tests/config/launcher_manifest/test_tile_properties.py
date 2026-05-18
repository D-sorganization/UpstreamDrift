"""TDD Tests for Launcher Manifest Loader.

Tests the shared launcher manifest system that ensures parity between
PyQt and Tauri/React launchers.

Test Categories:
    1. Manifest Loading — validate JSON parsing and DBC contracts
    2. Tile Properties — verify all tiles have required fields
    3. Logo Validation — check logo files exist on disk
    4. Ordering — verify Model Explorer is first tile
    5. Parity — verify all tiles can be consumed by both launchers
    6. Categories — verify physics_engine, tool, external groupings
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.config.launcher_manifest_loader import (
    ASSETS_DIR,
    LAUNCHER_CATEGORIES,
    MANIFEST_PATH,
    LauncherManifest,
    LauncherTile,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def manifest() -> LauncherManifest:
    """Load the production manifest."""
    return LauncherManifest.load()


@pytest.fixture
def sample_tile_dict() -> dict:
    """A minimal valid tile dictionary."""
    return {
        "id": "test_tile",
        "name": "Test Tile",
        "description": "A test tile",
        "category": "tool",
        "type": "special_app",
        "path": "src/test.py",
        "logo": "test.png",
        "status": "utility",
        "capabilities": ["test_cap"],
        "order": 1,
    }


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    """A minimal local registry file for provider-manifest tests."""
    config_path = tmp_path / "models.yaml"
    config_path.write_text("models: []\n", encoding="utf-8")
    return config_path


# =============================================================================
# 1. Manifest Loading
# =============================================================================


# =============================================================================
# 2. Tile Properties
# =============================================================================


class TestTileProperties:
    """Test individual tile data integrity."""

    def test_all_tiles_have_required_fields(self, manifest: LauncherManifest) -> None:
        """Every tile must have all required fields."""
        for tile in manifest.tiles:
            assert tile.id, f"Tile missing id: {tile}"
            assert tile.name, f"Tile missing name: {tile.id}"
            assert tile.description, f"Tile missing description: {tile.id}"
            assert tile.category, f"Tile missing category: {tile.id}"
            assert tile.type, f"Tile missing type: {tile.id}"
            assert tile.path, f"Tile missing path: {tile.id}"
            assert tile.logo, f"Tile missing logo: {tile.id}"

    def test_all_tiles_have_valid_category(self, manifest: LauncherManifest) -> None:
        """Category must be one of the allowed values."""
        for tile in manifest.tiles:
            assert (
                tile.category in LAUNCHER_CATEGORIES
            ), f"Tile '{tile.id}' has invalid category: '{tile.category}'"

    def test_physics_engines_have_engine_type(self, manifest: LauncherManifest) -> None:
        """All physics_engine tiles must have an engine_type."""
        for tile in manifest.physics_engines:
            assert tile.engine_type, f"Physics engine '{tile.id}' missing engine_type"

    def test_all_tiles_have_capabilities(self, manifest: LauncherManifest) -> None:
        """Every tile should declare at least one capability."""
        for tile in manifest.tiles:
            assert len(tile.capabilities) > 0, f"Tile '{tile.id}' has no capabilities"

    def test_tile_from_dict_missing_field_raises(self) -> None:
        """DBC: creating tile with missing required field raises ValueError."""
        with pytest.raises(ValueError, match="missing required"):
            LauncherTile.from_dict({"id": "test", "name": "Test"})

    def test_tile_to_dict_roundtrip(self, sample_tile_dict: dict) -> None:
        """Tile can roundtrip through dict serialization."""
        tile = LauncherTile.from_dict(sample_tile_dict)
        result = tile.to_dict()
        assert (
            result["id"] == sample_tile_dict["id"]
        ), "Assertion failed: result[id] == sample_tile_dict[id]"
        assert (
            result["name"] == sample_tile_dict["name"]
        ), "Assertion failed: result[name] == sample_tile_dict[name]"
        assert (
            result["capabilities"] == sample_tile_dict["capabilities"]
        ), "Assertion failed: result[capabilities] == sample_tile_dict[capabilities]"


# =============================================================================
# 3. Logo Validation
# =============================================================================


# =============================================================================
# 4. Ordering
# =============================================================================


# =============================================================================
# 5. Parity (PyQt ↔ Tauri)
# =============================================================================


# =============================================================================
# 6. Category Queries
# =============================================================================
