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
        assert tile.capabilities == ("rigid_body", "contact"), (
            "Assertion failed: tile.capabilities == (rigid_body, contact)"
        )

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
