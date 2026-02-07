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


# =============================================================================
# 1. Manifest Loading
# =============================================================================


class TestManifestLoading:
    """Test manifest file loading and DBC contracts."""

    def test_manifest_file_exists(self) -> None:
        """DBC Precondition: manifest file must exist."""
        assert MANIFEST_PATH.exists(), f"Manifest file missing at {MANIFEST_PATH}"

    def test_manifest_loads_successfully(self, manifest: LauncherManifest) -> None:
        """Manifest loads without errors."""
        assert manifest is not None
        assert len(manifest.tiles) > 0

    def test_manifest_has_version(self, manifest: LauncherManifest) -> None:
        """Manifest includes a version string."""
        assert manifest.version
        assert isinstance(manifest.version, str)

    def test_manifest_has_no_duplicate_ids(self, manifest: LauncherManifest) -> None:
        """DBC Postcondition: all tile IDs must be unique."""
        ids = [t.id for t in manifest.tiles]
        assert len(ids) == len(set(ids)), (
            f"Duplicate IDs found: {[x for x in ids if ids.count(x) > 1]}"
        )

    def test_manifest_file_not_found_raises(self) -> None:
        """DBC Precondition: missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            LauncherManifest.load(Path("/nonexistent/manifest.json"))

    def test_manifest_missing_tiles_raises(self, tmp_path: Path) -> None:
        """DBC: manifest without 'tiles' key raises ValueError."""
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"version": "1.0.0"}))
        with pytest.raises(ValueError, match="missing 'tiles'"):
            LauncherManifest.load(bad)

    def test_manifest_duplicate_ids_raises(self, tmp_path: Path) -> None:
        """DBC Postcondition: duplicate IDs raise ValueError."""
        bad = tmp_path / "dup.json"
        tile = {
            "id": "dup",
            "name": "Dup",
            "description": "d",
            "category": "tool",
            "type": "t",
            "path": "p",
            "logo": "l",
        }
        bad.write_text(json.dumps({"tiles": [tile, tile]}))
        with pytest.raises(ValueError, match="Duplicate"):
            LauncherManifest.load(bad)


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
        valid_categories = {"physics_engine", "tool", "external"}
        for tile in manifest.tiles:
            assert tile.category in valid_categories, (
                f"Tile '{tile.id}' has invalid category: '{tile.category}'"
            )

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
        assert result["id"] == sample_tile_dict["id"]
        assert result["name"] == sample_tile_dict["name"]
        assert result["capabilities"] == sample_tile_dict["capabilities"]


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
        assert not missing, (
            f"Missing logo files for tiles: {missing}. Expected in: {ASSETS_DIR}"
        )

    def test_logo_path_property(self, sample_tile_dict: dict) -> None:
        """Tile logo_path property returns absolute path."""
        tile = LauncherTile.from_dict(sample_tile_dict)
        assert tile.logo_path.is_absolute()
        assert str(tile.logo_path).endswith(sample_tile_dict["logo"])


# =============================================================================
# 4. Ordering
# =============================================================================


class TestOrdering:
    """Test tile ordering — Model Explorer must be first."""

    def test_tiles_sorted_by_order(self, manifest: LauncherManifest) -> None:
        """Tiles are returned sorted by their order field."""
        orders = [t.order for t in manifest.tiles]
        assert orders == sorted(orders), f"Tiles not sorted by order: {orders}"

    def test_model_explorer_is_first(self, manifest: LauncherManifest) -> None:
        """Model Explorer must be the first tile (order=1)."""
        first = manifest.tiles[0]
        assert first.id == "model_explorer", (
            f"First tile should be model_explorer, got: {first.id}"
        )

    def test_ordered_ids_returns_deterministic_list(
        self, manifest: LauncherManifest
    ) -> None:
        """ordered_ids is deterministic across loads."""
        ids1 = manifest.ordered_ids
        ids2 = LauncherManifest.load().ordered_ids
        assert ids1 == ids2


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
    }

    REQUIRED_TAURI_IDS = {
        "mujoco_unified",
        "drake_golf",
        "pinocchio_golf",
        "opensim_golf",
        "myosim_suite",
        "putting_green",
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

    def test_manifest_serializes_for_api(self, manifest: LauncherManifest) -> None:
        """Manifest can be serialized to JSON for the API endpoint."""
        data = manifest.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert len(parsed["tiles"]) == len(manifest.tiles)


# =============================================================================
# 6. Category Queries
# =============================================================================


class TestCategories:
    """Test category-based tile queries."""

    def test_physics_engines_not_empty(self, manifest: LauncherManifest) -> None:
        """There must be at least one physics engine."""
        assert len(manifest.physics_engines) > 0

    def test_tools_not_empty(self, manifest: LauncherManifest) -> None:
        """There must be at least one tool."""
        assert len(manifest.tools) > 0

    def test_get_tile_by_id(self, manifest: LauncherManifest) -> None:
        """get_tile returns correct tile for valid ID."""
        tile = manifest.get_tile("mujoco_unified")
        assert tile is not None
        assert tile.name == "MuJoCo"

    def test_get_tile_returns_none_for_invalid(
        self, manifest: LauncherManifest
    ) -> None:
        """get_tile returns None for nonexistent ID."""
        assert manifest.get_tile("nonexistent") is None

    def test_is_physics_engine_property(self, manifest: LauncherManifest) -> None:
        """is_physics_engine correctly identifies engines."""
        mujoco = manifest.get_tile("mujoco_unified")
        assert mujoco is not None
        assert mujoco.is_physics_engine

        model_explorer = manifest.get_tile("model_explorer")
        assert model_explorer is not None
        assert not model_explorer.is_physics_engine

    def test_motion_capture_is_tool(self, manifest: LauncherManifest) -> None:
        """Motion Capture (C3D + OpenPose + MediaPipe) is categorized as a tool."""
        mc = manifest.get_tile("motion_capture")
        assert mc is not None
        assert mc.is_tool
        assert "openpose" in mc.capabilities
        assert "mediapipe" in mc.capabilities
        assert "c3d_viewer" in mc.capabilities
