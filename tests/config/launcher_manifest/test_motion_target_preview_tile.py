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


class TestMotionTargetPreviewTile:
    """Closes #4486 — multi-source motion-target preview tile + legacy fix."""

    def test_motion_target_preview_tile_present_and_valid(
        self, manifest: LauncherManifest
    ) -> None:
        """The new generic Motion-Match Preview tile must be in the manifest."""
        tile = manifest.get_tile("motion_target_preview")
        assert tile is not None, "motion_target_preview tile missing"
        assert (
            tile.name == "Motion-Match Preview"
        ), "Assertion failed: tile.name == Motion-Match Preview"
        assert tile.category == "tool", "Assertion failed: tile.category == tool"
        assert (
            tile.logo == "motion_target_preview.svg"
        ), "Assertion failed: tile.logo == motion_target_preview.svg"
        assert tile.logo_path.exists(), "Assertion failed: tile.logo_path.exists()"
        assert (
            tile.path == "src.tools.starting_pose_matcher.__main__"
        ), "Assertion failed: tile.path == src.tools.starting_pose_matcher.__main__"
        assert not tile.hidden, "Assertion failed: not tile.hidden"
        # Tags must be source-neutral and cover the issue's required set.
        for required_tag in ("c3d", "mocap", "club", "body", "preview"):
            assert (
                required_tag in tile.tags or required_tag in tile.capabilities
            ), "Assertion failed: required_tag in tile.tags or required_tag in tile.capabilities"

    def test_legacy_starting_pose_matcher_validates_with_logo(
        self, manifest: LauncherManifest
    ) -> None:
        """Legacy alias entry validates (logo present) and is hidden by default."""
        # Search the full tile list (visible defaults exclude hidden tiles).
        legacy = next(
            (t for t in manifest.tiles if t.id == "starting_pose_matcher"),
            None,
        )
        assert legacy is not None, "Legacy starting_pose_matcher tile missing"
        assert legacy.logo, "Legacy tile must have a non-empty logo (#4486)"
        assert legacy.logo_path.exists(), "Assertion failed: legacy.logo_path.exists()"
        assert legacy.hidden is True, "Assertion failed: legacy.hidden is True"

    def test_visible_tiles_excludes_hidden_legacy_alias(
        self, manifest: LauncherManifest
    ) -> None:
        """`visible_tiles` and `tools` must skip hidden legacy aliases."""
        visible_ids = {t.id for t in manifest.visible_tiles}
        assert (
            "motion_target_preview" in visible_ids
        ), "Assertion failed: motion_target_preview in visible_ids"
        assert (
            "starting_pose_matcher" not in visible_ids
        ), "Assertion failed: starting_pose_matcher not in visible_ids"
        tool_ids = {t.id for t in manifest.tools}
        assert (
            "starting_pose_matcher" not in tool_ids
        ), "Assertion failed: starting_pose_matcher not in tool_ids"
