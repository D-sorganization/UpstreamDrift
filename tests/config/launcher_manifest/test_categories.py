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


# =============================================================================
# 6. Category Queries
# =============================================================================


class TestCategories:
    """Test category-based tile queries."""

    def test_physics_engines_not_empty(self, manifest: LauncherManifest) -> None:
        """There must be at least one physics engine."""
        assert (
            len(manifest.physics_engines) > 0
        ), "Assertion failed: len(manifest.physics_engines) > 0"

    def test_tools_not_empty(self, manifest: LauncherManifest) -> None:
        """There must be at least one tool."""
        assert len(manifest.tools) > 0, "Assertion failed: len(manifest.tools) > 0"

    def test_get_tile_by_id(self, manifest: LauncherManifest) -> None:
        """get_tile returns correct tile for valid ID."""
        tile = manifest.get_tile("mujoco_unified")
        assert tile is not None, "Assertion failed: tile is not None"
        assert tile.name == "MuJoCo", "Assertion failed: tile.name == MuJoCo"

    def test_get_tile_returns_none_for_invalid(
        self, manifest: LauncherManifest
    ) -> None:
        """get_tile returns None for nonexistent ID."""
        assert (
            manifest.get_tile("nonexistent") is None
        ), "Assertion failed: manifest.get_tile(nonexistent) is None"

    def test_is_physics_engine_property(self, manifest: LauncherManifest) -> None:
        """is_physics_engine correctly identifies engines."""
        mujoco = manifest.get_tile("mujoco_unified")
        assert mujoco is not None, "Assertion failed: mujoco is not None"
        assert mujoco.is_physics_engine, "Assertion failed: mujoco.is_physics_engine"

        model_explorer = manifest.get_tile("model_explorer")
        assert (
            model_explorer is not None
        ), "Assertion failed: model_explorer is not None"
        assert (
            not model_explorer.is_physics_engine
        ), "Assertion failed: not model_explorer.is_physics_engine"

    def test_motion_capture_is_tool(self, manifest: LauncherManifest) -> None:
        """Motion Capture (C3D + OpenPose + MediaPipe) is categorized as a tool."""
        mc = manifest.get_tile("motion_capture")
        assert mc is not None, "Assertion failed: mc is not None"
        assert mc.is_tool, "Assertion failed: mc.is_tool"
        assert (
            "openpose" in mc.capabilities
        ), "Assertion failed: openpose in mc.capabilities"
        assert (
            "mediapipe" in mc.capabilities
        ), "Assertion failed: mediapipe in mc.capabilities"
        assert (
            "c3d_viewer" in mc.capabilities
        ), "Assertion failed: c3d_viewer in mc.capabilities"

    @pytest.mark.parametrize(
        "tile_id,expected_path",
        [
            ("mujoco_dashboard", "src/launchers/mujoco_dashboard.py"),
            ("drake_dashboard", "src/launchers/drake_dashboard.py"),
            ("pinocchio_dashboard", "src/launchers/pinocchio_dashboard.py"),
        ],
    )
    def test_engine_dashboard_tiles_exist(
        self,
        manifest: LauncherManifest,
        tile_id: str,
        expected_path: str,
    ) -> None:
        """Each engine-specific dashboard has a dedicated tile in the manifest.

        Fixes #5515: engine dashboards were reachable only via source code,
        not from the launcher sidebar.
        """
        tile = manifest.get_tile(tile_id)
        assert (
            tile is not None
        ), f"Tile '{tile_id}' must exist in launcher_manifest.json"
        assert (
            tile.is_physics_engine
        ), f"'{tile_id}' must be in the physics_engine category"
        assert tile.path == expected_path, f"'{tile_id}' must point to {expected_path}"
