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


# =============================================================================
# 3. Logo Validation
# =============================================================================


class TestLogoValidation:
    """Test that logo files exist for all tiles."""

    def test_assets_dir_exists(self) -> None:
        """The launcher assets directory must exist."""
        assert ASSETS_DIR.exists(), f"Assets dir missing: {ASSETS_DIR}"

    def test_all_tiles_have_logo_files(self, manifest: LauncherManifest) -> None:
        """Every tile's logo file must exist in the assets directory.

        All SVG logos were created in Phase 3 (closes #1164).
        """
        missing = manifest.validate_logos()
        assert (
            not missing
        ), f"Missing logo files for tiles: {missing}. Expected in: {ASSETS_DIR}"

    def test_logo_path_property(self, sample_tile_dict: dict) -> None:
        """Tile logo_path property returns absolute path."""
        tile = LauncherTile.from_dict(sample_tile_dict)
        assert (
            tile.logo_path.is_absolute()
        ), "Assertion failed: tile.logo_path.is_absolute()"
        assert str(tile.logo_path).endswith(
            sample_tile_dict["logo"]
        ), "Assertion failed: str(tile.logo_path).endswith(sample_tile_dict[logo])"


# =============================================================================
# 4. Ordering
# =============================================================================


# =============================================================================
# 5. Parity (PyQt ↔ Tauri)
# =============================================================================


# =============================================================================
# 6. Category Queries
# =============================================================================
