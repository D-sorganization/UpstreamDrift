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


class TestManifestLoading:
    """Test manifest file loading and DBC contracts."""

    def test_manifest_file_exists(self) -> None:
        """DBC Precondition: manifest file must exist."""
        assert MANIFEST_PATH.exists(), f"Manifest file missing at {MANIFEST_PATH}"

    def test_manifest_loads_successfully(self, manifest: LauncherManifest) -> None:
        """Manifest loads without errors."""
        assert manifest is not None, "Assertion failed: manifest is not None"
        assert len(manifest.tiles) > 0, "Assertion failed: len(manifest.tiles) > 0"

    def test_manifest_has_version(self, manifest: LauncherManifest) -> None:
        """Manifest includes a version string."""
        assert manifest.version, "Assertion failed: manifest.version"
        assert isinstance(manifest.version, str), (
            "Assertion failed: isinstance(manifest.version, str)"
        )

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

    def test_hidden_tile_without_owner_or_reason_raises(self, tmp_path: Path) -> None:
        """Hidden launcher entries must explain ownership and unblock criteria."""
        bad = tmp_path / "hidden.json"
        tile = {
            "id": "hidden_tool",
            "name": "Hidden Tool",
            "description": "A hidden tool",
            "category": "tool",
            "type": "special_app",
            "path": "src.tools.hidden",
            "logo": "golf_logo.svg",
            "hidden": True,
        }
        bad.write_text(json.dumps({"tiles": [tile]}), encoding="utf-8")

        with pytest.raises(ValueError, match="hidden_reason"):
            LauncherManifest.load(bad)

    def test_manifest_loads_provider_tiles_from_configured_roots(
        self,
        tmp_path: Path,
        registry_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Provider manifests augment the launcher tile list through the registry."""
        manifest_path = tmp_path / "launcher_manifest.json"
        manifest_path.write_text(
            json.dumps({"version": "1.0.0", "tiles": []}),
            encoding="utf-8",
        )

        provider_root = tmp_path / "providers" / "mujoco-models"
        provider_root.mkdir(parents=True)
        provider_manifest = provider_root / "model_pack.yaml"
        provider_manifest.write_text(
            """
manifest_version: "1.0.0"
pack_id: "mujoco-pack"
pack_name: "MuJoCo Models"
provider: "mujoco_models"
models:
  - id: "external_mujoco"
    name: "External MuJoCo"
    description: "Provider-backed MuJoCo model"
    type: "custom_humanoid"
    path: "apps/mujoco_launcher.py"
    engine_type: "mujoco"
    capabilities: ["rigid_body", "contact"]
    order: 4
""".strip(),
            encoding="utf-8",
        )
        monkeypatch.setenv("UPSTREAM_DRIFT_PROVIDER_ROOTS", str(provider_root))

        manifest = LauncherManifest.load(
            manifest_path,
            registry_path=registry_path,
        )

        tile = manifest.get_tile("external_mujoco")
        assert tile is not None, "Assertion failed: tile is not None"
        assert tile.category == "physics_engine", (
            "Assertion failed: tile.category == physics_engine"
        )
        assert tile.provider == "mujoco_models", (
            "Assertion failed: tile.provider == mujoco_models"
        )
        assert tile.source_root == str(provider_root), (
            "Assertion failed: tile.source_root == str(provider_root)"
        )
        assert tile.logo == "mujoco_humanoid.svg", (
            "Assertion failed: tile.logo == mujoco_humanoid.svg"
        )
        assert tile.capabilities == (
            "rigid_body",
            "contact",
        ), "Assertion failed: tile.capabilities == (rigid_body, contact)"

    def test_manifest_prefers_explicit_provider_launcher_metadata(
        self,
        tmp_path: Path,
        registry_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Provider packs can define launcher presentation without loader inference."""
        manifest_path = tmp_path / "launcher_manifest.json"
        manifest_path.write_text(
            json.dumps({"version": "1.0.0", "tiles": []}),
            encoding="utf-8",
        )

        provider_root = tmp_path / "providers" / "drake-models"
        provider_root.mkdir(parents=True)
        (provider_root / "model_pack.yaml").write_text(
            """
manifest_version: "1.0.0"
pack_id: "drake-pack"
pack_name: "Drake Models"
provider: "drake_models"
models:
  - id: "external_drake"
    name: "External Drake"
    description: "Provider-backed Drake model"
    type: "drake"
    path: "apps/drake_launcher.py"
    engine_type: "drake"
    capabilities: ["rigid_body"]
    launcher:
      category: "physics_engine"
      logo: "drake.svg"
      status: "experimental"
      web_route: "/providers/drake"
""".strip(),
            encoding="utf-8",
        )
        monkeypatch.setenv("UPSTREAM_DRIFT_PROVIDER_ROOTS", str(provider_root))

        manifest = LauncherManifest.load(
            manifest_path,
            registry_path=registry_path,
        )

        tile = manifest.get_tile("external_drake")
        assert tile is not None, "Assertion failed: tile is not None"
        assert tile.category == "physics_engine", (
            "Assertion failed: tile.category == physics_engine"
        )
        assert tile.logo == "drake.svg", "Assertion failed: tile.logo == drake.svg"
        assert tile.status == "experimental", (
            "Assertion failed: tile.status == experimental"
        )
        assert tile.web_route == "/providers/drake", (
            "Assertion failed: tile.web_route == /providers/drake"
        )

    def test_manifest_marks_provider_tile_runtime_unavailable(
        self,
        tmp_path: Path,
        registry_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Provider tiles should surface runtime-unavailable status cleanly."""
        manifest_path = tmp_path / "launcher_manifest.json"
        manifest_path.write_text(
            json.dumps({"version": "1.0.0", "tiles": []}),
            encoding="utf-8",
        )

        provider_root = tmp_path / "providers" / "pinocchio-models"
        (provider_root / "models").mkdir(parents=True)
        (provider_root / "models" / "pinocchio.urdf").write_text(
            "<robot />",
            encoding="utf-8",
        )
        (provider_root / "model_pack.yaml").write_text(
            """
manifest_version: "1.0.0"
pack_id: "pinocchio-pack"
pack_name: "Pinocchio Models"
provider: "pinocchio_models"
models:
  - id: "external_pinocchio"
    name: "External Pinocchio"
    description: "Provider-backed Pinocchio model"
    type: "urdf"
    path: "models/pinocchio.urdf"
    engine_type: "pinocchio"
    capabilities: ["rigid_body"]
    identity:
      canonical_id: "demo.external.pinocchio"
      motion_family: "demo"
      exercise: "pinocchio"
      humanoid: "athlete"
""".strip(),
            encoding="utf-8",
        )
        monkeypatch.setenv("UPSTREAM_DRIFT_PROVIDER_ROOTS", str(provider_root))
        monkeypatch.setattr(
            "src.config.launcher_manifest_loader.is_engine_runtime_available",
            lambda engine_type: False,
        )

        manifest = LauncherManifest.load(
            manifest_path,
            registry_path=registry_path,
        )

        tile = manifest.get_tile("external_pinocchio")
        assert tile is not None, "Assertion failed: tile is not None"
        assert tile.status == "runtime_unavailable", (
            "Assertion failed: tile.status == runtime_unavailable"
        )

    def test_manifest_loads_utility_provider_tiles_from_known_roots_without_env(
        self,
        tmp_path: Path,
    ) -> None:
        """Utility repos should be discoverable through the shared launch path."""
        workspace_root = tmp_path
        repo_root = workspace_root / "UpstreamDrift"
        manifest_path = repo_root / "src" / "config" / "launcher_manifest.json"
        registry_path = repo_root / "src" / "config" / "models.yaml"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps({"version": "1.0.0", "tiles": []}),
            encoding="utf-8",
        )
        registry_path.write_text("models: []\n", encoding="utf-8")

        tools_root = workspace_root / "Tools"
        tools_root.mkdir(parents=True)
        (tools_root / "model_pack.yaml").write_text(
            """
manifest_version: "1.0.0"
pack_id: "tools-pack"
pack_name: "Tools"
provider: "tools"
models:
  - id: "pendulum_suite"
    name: "Pendulum Suite"
    description: "Pendulum workflows"
    type: "special_app"
    path: "src/pendulum_launcher.py"
    capabilities: ["pendulum", "simulation"]
    launcher:
      category: "tool"
      logo: "golf_logo.svg"
      status: "utility"
      web_route: "/tools/pendulum-suite"
""".strip(),
            encoding="utf-8",
        )

        manifest = LauncherManifest.load(
            manifest_path,
            registry_path=registry_path,
        )

        tile = manifest.get_tile("pendulum_suite")
        assert tile is not None, "Assertion failed: tile is not None"
        assert tile.category == "tool", "Assertion failed: tile.category == tool"
        assert tile.status == "utility", "Assertion failed: tile.status == utility"
        assert tile.web_route == "/tools/pendulum-suite", (
            "Assertion failed: tile.web_route == /tools/pendulum-suite"
        )

    def test_manifest_ignores_provider_tiles_when_disabled(
        self,
        tmp_path: Path,
        registry_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Provider discovery is opt-out for callers that need static-only tiles."""
        manifest_path = tmp_path / "launcher_manifest.json"
        manifest_path.write_text(
            json.dumps({"version": "1.0.0", "tiles": []}),
            encoding="utf-8",
        )

        provider_root = tmp_path / "providers" / "opensim-models"
        provider_root.mkdir(parents=True)
        (provider_root / "model_pack.yaml").write_text(
            """
manifest_version: "1.0.0"
pack_id: "opensim-pack"
pack_name: "OpenSim Models"
provider: "opensim_models"
models:
  - id: "external_opensim"
    name: "External OpenSim"
    description: "Provider-backed OpenSim model"
    type: "opensim"
    path: "apps/opensim_gui.py"
    engine_type: "opensim"
""".strip(),
            encoding="utf-8",
        )
        monkeypatch.setenv("UPSTREAM_DRIFT_PROVIDER_ROOTS", str(provider_root))

        manifest = LauncherManifest.load(
            manifest_path,
            include_provider_tiles=False,
            registry_path=registry_path,
        )

        assert manifest.get_tile("external_opensim") is None, (
            "Assertion failed: manifest.get_tile(external_opensim) is None"
        )


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
            assert tile.category in LAUNCHER_CATEGORIES, (
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
        assert result["id"] == sample_tile_dict["id"], (
            "Assertion failed: result[id] == sample_tile_dict[id]"
        )
        assert result["name"] == sample_tile_dict["name"], (
            "Assertion failed: result[name] == sample_tile_dict[name]"
        )
        assert result["capabilities"] == sample_tile_dict["capabilities"], (
            "Assertion failed: result[capabilities] == sample_tile_dict[capabilities]"
        )


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
        assert tile.logo_path.is_absolute(), (
            "Assertion failed: tile.logo_path.is_absolute()"
        )
        assert str(tile.logo_path).endswith(sample_tile_dict["logo"]), (
            "Assertion failed: str(tile.logo_path).endswith(sample_tile_dict[logo])"
        )


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
        assert ids1 == ids2, "Assertion failed: ids1 == ids2"

    def test_mixed_static_and_provider_tiles_sort_by_order_then_id(
        self,
        tmp_path: Path,
        registry_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mixed tile sources preserve deterministic ordering across migration."""
        manifest_path = tmp_path / "launcher_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "version": "1.0.0",
                    "tiles": [
                        {
                            "id": "z_static",
                            "name": "Z Static",
                            "description": "Static tile",
                            "category": "tool",
                            "type": "special_app",
                            "path": "src/z_static.py",
                            "logo": "golf_logo.svg",
                            "status": "utility",
                            "capabilities": ["docs"],
                            "order": 3,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        provider_root = tmp_path / "providers" / "mujoco-models"
        provider_root.mkdir(parents=True)
        (provider_root / "model_pack.yaml").write_text(
            """
manifest_version: "1.0.0"
pack_id: "mujoco-pack"
pack_name: "MuJoCo Models"
provider: "mujoco_models"
models:
  - id: "a_provider"
    name: "A Provider"
    description: "Provider tile"
    type: "custom_humanoid"
    path: "apps/mujoco_launcher.py"
    engine_type: "mujoco"
    order: 3
""".strip(),
            encoding="utf-8",
        )
        monkeypatch.setenv("UPSTREAM_DRIFT_PROVIDER_ROOTS", str(provider_root))

        manifest = LauncherManifest.load(
            manifest_path,
            registry_path=registry_path,
        )

        assert manifest.ordered_ids == [
            "a_provider",
            "z_static",
        ], "Assertion failed: manifest.ordered_ids == [a_provider, z_static]"


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
            assert tile.source_root == "../Tools", (
                f"{tile_id} must resolve from the sibling Tools repo"
            )
            assert not tile.path.startswith("src/tools/"), (
                f"{tile_id} must not point at UpstreamDrift-local tool source"
            )

    def test_manifest_serializes_for_api(self, manifest: LauncherManifest) -> None:
        """Manifest can be serialized to JSON for the API endpoint."""
        data = manifest.to_dict()
        # Should be JSON-serializable
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert len(parsed["tiles"]) == len(manifest.visible_tiles), (
            "Assertion failed: len(parsed[tiles]) == len(manifest.visible_tiles)"
        )


# =============================================================================
# 6. Category Queries
# =============================================================================


class TestCategories:
    """Test category-based tile queries."""

    def test_physics_engines_not_empty(self, manifest: LauncherManifest) -> None:
        """There must be at least one physics engine."""
        assert len(manifest.physics_engines) > 0, (
            "Assertion failed: len(manifest.physics_engines) > 0"
        )

    def test_tools_not_empty(self, manifest: LauncherManifest) -> None:
        """There must be at least one tool."""
        assert len(manifest.tools) > 0, "Assertion failed: len(manifest.tools) > 0"

    def test_biomechanics_category_is_discoverable(
        self, manifest: LauncherManifest
    ) -> None:
        """Provider-backed biomechanics tools must be discoverable by category."""
        biomechanics = manifest.get_tiles_by_category("biomechanics")
        assert {tile.id for tile in biomechanics} == {"movement_optimizer"}
        assert manifest.categories["biomechanics"] == biomechanics

    def test_unknown_category_raises(self, manifest: LauncherManifest) -> None:
        """DBC: category queries reject unknown categories."""
        with pytest.raises(ValueError, match="Unknown launcher category"):
            manifest.get_tiles_by_category("misc")

    def test_get_tile_by_id(self, manifest: LauncherManifest) -> None:
        """get_tile returns correct tile for valid ID."""
        tile = manifest.get_tile("mujoco_unified")
        assert tile is not None, "Assertion failed: tile is not None"
        assert tile.name == "MuJoCo", "Assertion failed: tile.name == MuJoCo"

    def test_get_tile_returns_none_for_invalid(
        self, manifest: LauncherManifest
    ) -> None:
        """get_tile returns None for nonexistent ID."""
        assert manifest.get_tile("nonexistent") is None, (
            "Assertion failed: manifest.get_tile(nonexistent) is None"
        )

    def test_is_physics_engine_property(self, manifest: LauncherManifest) -> None:
        """is_physics_engine correctly identifies engines."""
        mujoco = manifest.get_tile("mujoco_unified")
        assert mujoco is not None, "Assertion failed: mujoco is not None"
        assert mujoco.is_physics_engine, "Assertion failed: mujoco.is_physics_engine"

        model_explorer = manifest.get_tile("model_explorer")
        assert model_explorer is not None, (
            "Assertion failed: model_explorer is not None"
        )
        assert not model_explorer.is_physics_engine, (
            "Assertion failed: not model_explorer.is_physics_engine"
        )

    def test_motion_capture_is_tool(self, manifest: LauncherManifest) -> None:
        """Motion Capture (C3D + OpenPose + MediaPipe) is categorized as a tool."""
        mc = manifest.get_tile("motion_capture")
        assert mc is not None, "Assertion failed: mc is not None"
        assert mc.is_tool, "Assertion failed: mc.is_tool"
        assert "openpose" in mc.capabilities, (
            "Assertion failed: openpose in mc.capabilities"
        )
        assert "mediapipe" in mc.capabilities, (
            "Assertion failed: mediapipe in mc.capabilities"
        )
        assert "c3d_viewer" in mc.capabilities, (
            "Assertion failed: c3d_viewer in mc.capabilities"
        )


class TestMotionTargetPreviewTile:
    """Closes #4486 — multi-source motion-target preview tile + legacy fix."""

    def test_motion_target_preview_tile_present_and_valid(
        self, manifest: LauncherManifest
    ) -> None:
        """The new generic Motion-Match Preview tile must be in the manifest."""
        tile = manifest.get_tile("motion_target_preview")
        assert tile is not None, "motion_target_preview tile missing"
        assert tile.name == "Motion-Match Preview", (
            "Assertion failed: tile.name == Motion-Match Preview"
        )
        assert tile.category == "tool", "Assertion failed: tile.category == tool"
        assert tile.logo == "motion_target_preview.svg", (
            "Assertion failed: tile.logo == motion_target_preview.svg"
        )
        assert tile.logo_path.exists(), "Assertion failed: tile.logo_path.exists()"
        assert tile.path == "src.tools.starting_pose_matcher.__main__", (
            "Assertion failed: tile.path == src.tools.starting_pose_matcher.__main__"
        )
        assert not tile.hidden, "Assertion failed: not tile.hidden"
        # Tags must be source-neutral and cover the issue's required set.
        for required_tag in ("c3d", "mocap", "club", "body", "preview"):
            assert required_tag in tile.tags or required_tag in tile.capabilities, (
                "Assertion failed: required_tag in tile.tags or required_tag in tile.capabilities"
            )

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
        assert legacy.hidden_reason, "Hidden tiles must document why they are hidden"
        assert legacy.hidden_owner, "Hidden tiles must document an owning team"

    def test_visible_tiles_excludes_hidden_legacy_alias(
        self, manifest: LauncherManifest
    ) -> None:
        """`visible_tiles` and `tools` must skip hidden legacy aliases."""
        visible_ids = {t.id for t in manifest.visible_tiles}
        assert "motion_target_preview" in visible_ids, (
            "Assertion failed: motion_target_preview in visible_ids"
        )
        assert "starting_pose_matcher" not in visible_ids, (
            "Assertion failed: starting_pose_matcher not in visible_ids"
        )
        tool_ids = {t.id for t in manifest.tools}
        assert "starting_pose_matcher" not in tool_ids, (
            "Assertion failed: starting_pose_matcher not in tool_ids"
        )


class TestWebRouteFieldRoundTrip:
    """Tests for web_route round-trip preservation (issue #2494)."""

    def test_from_dict_preserves_web_route(self) -> None:
        """from_dict() must read web_route from the manifest dict."""
        data = {
            "id": "test_tile",
            "name": "Test",
            "description": "A test tile",
            "category": "tool",
            "type": "web",
            "path": "/some/path",
            "logo": "logo.png",
            "status": "gui_ready",
            "web_route": "/tools/test",
        }
        tile = LauncherTile.from_dict(data)
        assert tile.web_route == "/tools/test", (
            "Assertion failed: tile.web_route == /tools/test"
        )

    def test_to_dict_includes_web_route(self) -> None:
        """to_dict() must serialize web_route so it survives a round-trip."""
        data = {
            "id": "test_tile",
            "name": "Test",
            "description": "A test tile",
            "category": "tool",
            "type": "web",
            "path": "/some/path",
            "logo": "logo.png",
            "status": "gui_ready",
            "web_route": "/tools/test",
        }
        tile = LauncherTile.from_dict(data)
        serialized = tile.to_dict()
        assert "web_route" in serialized, "Assertion failed: web_route in serialized"
        assert serialized["web_route"] == "/tools/test", (
            "Assertion failed: serialized[web_route] == /tools/test"
        )

    def test_web_route_none_by_default(self) -> None:
        """web_route defaults to None when absent from the manifest dict."""
        data = {
            "id": "test_tile",
            "name": "Test",
            "description": "A test tile",
            "category": "physics_engine",
            "type": "mujoco",
            "path": "/some/path",
            "logo": "logo.png",
            "status": "engine_ready",
        }
        tile = LauncherTile.from_dict(data)
        assert tile.web_route is None, "Assertion failed: tile.web_route is None"

    def test_to_dict_omits_web_route_when_none(self) -> None:
        """to_dict() must not include web_route key when it is None."""
        data = {
            "id": "test_tile",
            "name": "Test",
            "description": "A test tile",
            "category": "physics_engine",
            "type": "mujoco",
            "path": "/some/path",
            "logo": "logo.png",
        }
        tile = LauncherTile.from_dict(data)
        serialized = tile.to_dict()
        assert "web_route" not in serialized, (
            "Assertion failed: web_route not in serialized"
        )
