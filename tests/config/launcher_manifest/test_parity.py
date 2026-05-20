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


# =============================================================================
# 4. Ordering
# =============================================================================


# =============================================================================
# 5. Parity (PyQt ↔ Tauri)
# =============================================================================


class TestParity:
    """Test that manifest covers all tiles needed by both launchers."""

    # The canonical tile IDs that must be present
    REQUIRED_PYQT_IDS = {
        "mujoco_unified",
        "drake_golf",
        "pinocchio_golf",
        "opensim_golf",
        "myosim_suite",
        "matlab_unified",
        "motion_capture",
        "model_explorer",
        "putting_green",
        "video_analyzer",
        "video_processor",
        "data_explorer",
        "data_processor",
    }

    REQUIRED_TAURI_IDS = {
        "mujoco_unified",
        "drake_golf",
        "pinocchio_golf",
        "opensim_golf",
        "myosim_suite",
        "putting_green",
        "video_analyzer",
        "video_processor",
        "data_explorer",
        "data_processor",
    }

    def test_manifest_covers_all_pyqt_tiles(self, manifest: LauncherManifest) -> None:
        """All PyQt launcher tiles must be in the manifest."""
        manifest_ids = set(manifest.tile_ids)
        missing = self.REQUIRED_PYQT_IDS - manifest_ids
        assert not missing, f"PyQt tiles missing from manifest: {missing}"

    def test_manifest_covers_all_tauri_tiles(self, manifest: LauncherManifest) -> None:
        """All Tauri launcher tiles must be in the manifest."""
        manifest_ids = set(manifest.tile_ids)
        missing = self.REQUIRED_TAURI_IDS - manifest_ids
        assert not missing, f"Tauri tiles missing from manifest: {missing}"

    def test_shared_tools_live_in_tools_repo(self, manifest: LauncherManifest) -> None:
        """Video/data surfaces exposed in UpstreamDrift resolve to Tools."""
        shared_ids = {
            "video_analyzer",
            "video_processor",
            "data_explorer",
            "data_processor",
        }

        for tile_id in shared_ids:
            tile = manifest.get_tile(tile_id)
            assert tile is not None, f"Missing shared Tools tile: {tile_id}"
            assert tile.provider == "tools", f"{tile_id} must declare Tools as provider"
            assert (
                tile.source_root == "../Tools"
            ), f"{tile_id} must resolve from the sibling Tools repo"
            assert not tile.path.startswith(
                "src/tools/"
            ), f"{tile_id} must not point at UpstreamDrift-local tool source"

    def test_manifest_serializes_for_api(self, manifest: LauncherManifest) -> None:
        """Manifest can be serialized to JSON for the API endpoint."""
        data = manifest.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert len(parsed["tiles"]) == len(
            manifest.visible_tiles
        ), "Assertion failed: len(parsed[tiles]) == len(manifest.visible_tiles)"


# =============================================================================
# 6. Category Queries
# =============================================================================
