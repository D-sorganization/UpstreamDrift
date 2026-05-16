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


# =============================================================================
# 6. Category Queries
# =============================================================================
