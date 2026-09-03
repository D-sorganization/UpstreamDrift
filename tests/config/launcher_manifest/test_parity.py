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
import re
from pathlib import Path

import pytest
from src.config.launcher_manifest_loader import (
    MANIFEST_PATH,
    WEB_LAUNCH_MODES,
    LauncherManifest,
    WebLaunchContract,
)
from src.shared.python.config.model_registry import ModelRegistry

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_TSX = REPO_ROOT / "ui" / "src" / "App.tsx"

pytestmark = pytest.mark.parity

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODELS_YAML = _REPO_ROOT / "src" / "config" / "models.yaml"
_MANIFEST_JSON = _REPO_ROOT / "src" / "config" / "launcher_manifest.json"

# ``models.yaml`` is the primary PyQt6 launch contract.  Shared tiles must
# expose the same native launch semantics through the manifest API.
# ``web_route`` and ``capabilities`` are deliberately excluded: they are
# web-catalog hints with no native-launch counterpart, not replacements for
# the native target.
_SHARED_SEMANTIC_FIELDS = (
    "category",
    "status",
    "type",
    "path",
    "engine_type",
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
        "matlab_suite",
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
            "rate_of_closure",
        }

        for tile_id in shared_ids:
            tile = manifest.get_tile(tile_id)
            assert tile is not None, f"Missing shared Tools tile: {tile_id}"
            assert tile.provider == "tools", f"{tile_id} must declare Tools as provider"
            assert tile.source_root is None, (
                f"{tile_id} must not serialize a mutable Tools checkout path"
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

    def test_shared_tiles_match_the_native_pyqt6_semantic_contract(self) -> None:
        """Shared IDs retain the native category, target, and engine semantics.

        ``web_route`` and capability tags intentionally remain web-only
        catalog metadata; neither may substitute for ``path``.
        """
        native_by_id = {
            model.id: model
            for model in ModelRegistry(config_path=_MODELS_YAML).get_all_models()
        }
        manifest = LauncherManifest.load(
            _MANIFEST_JSON, include_provider_tiles=False, registry_path=_MODELS_YAML
        )
        shared_by_id = {
            tile.id: tile for tile in manifest.tiles if tile.id in native_by_id
        }

        mismatches: dict[str, dict[str, tuple[object, object]]] = {}
        for tile_id, tile in shared_by_id.items():
            native = native_by_id[tile_id]
            native_launcher = native.launcher
            assert native_launcher is not None
            native_status = native_launcher.status
            if (
                native.provider == "tools"
                and not (_REPO_ROOT / "vendor" / "ud-tools" / "src").is_dir()
            ):
                native_status = "provider_unavailable"
            elif (
                native.provider != "tools"
                and isinstance(native.source_root, str)
                and not Path(native.source_root).exists()
            ):
                # Sibling-checkout tiles (e.g. movement_optimizer) degrade to
                # provider_unavailable when the sibling is absent - the same
                # rule the loader applies (issue #9412 surfaces every
                # registry tile on the web, not only the former manifest set).
                native_status = "provider_unavailable"
            native_fields = {
                "category": native_launcher.category,
                "status": native_status,
                "type": native.type,
                "path": native.path,
                "engine_type": native.engine_type,
            }
            field_differences = {
                field: (native_fields[field], getattr(tile, field))
                for field in _SHARED_SEMANTIC_FIELDS
                if native_fields[field] != getattr(tile, field)
            }
            if field_differences:
                mismatches[tile_id] = field_differences

        assert not mismatches, (
            f"Shared launcher tiles drift from the primary PyQt6 contract: {mismatches}"
        )


# =============================================================================
# 6. Category Queries
# =============================================================================


# =============================================================================
# 7. Web reachability contract (issue #7461)
# =============================================================================


def _raw_manifest_tiles() -> list[dict]:
    """Tiles exactly as declared in launcher_manifest.json (no augmentation)."""
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)["tiles"]


def _react_router_paths() -> set[str]:
    """Extract the route table from ui/src/App.tsx via regex."""
    content = APP_TSX.read_text(encoding="utf-8")
    return set(re.findall(r'path="([^"]+)"', content))


class TestWebReachabilityContract:
    """Every manifest tile must declare an honest web launch contract.

    A new tile without a ``web`` declaration (or with a route that does not
    exist in the React router) fails CI, so the web dashboard can always
    render honest affordances instead of dead buttons (issue #7461).
    """

    def test_every_tile_declares_web_contract(self) -> None:
        """Every tile in launcher_manifest.json has an explicit `web` mapping."""
        missing = [t["id"] for t in _raw_manifest_tiles() if "web" not in t]
        assert not missing, (
            f"Tiles missing explicit 'web' launch contract: {missing}. "
            "Declare web: {mode: route|native-window|unavailable, ...} "
            "for each new tile (issue #7461)."
        )

    def test_every_web_contract_is_valid(self) -> None:
        """Every declared contract validates against WebLaunchContract."""
        for tile in _raw_manifest_tiles():
            contract = WebLaunchContract.from_dict(tile["web"], tile_id=tile["id"])
            assert contract.mode in WEB_LAUNCH_MODES

    def test_route_mode_routes_start_with_slash(self) -> None:
        """Every route-mode tile declares a route string starting with '/'."""
        for tile in _raw_manifest_tiles():
            web = tile["web"]
            if web["mode"] == "route":
                assert web.get("route", "").startswith("/"), (
                    f"Tile '{tile['id']}' route must start with '/': {web!r}"
                )

    def test_route_mode_routes_exist_in_react_router(self) -> None:
        """Every route-mode route resolves to a route in ui/src/App.tsx."""
        assert APP_TSX.exists(), f"React router file not found: {APP_TSX}"
        router_paths = _react_router_paths()
        assert router_paths, "No route table extracted from App.tsx"
        bad = [
            (tile["id"], tile["web"]["route"])
            for tile in _raw_manifest_tiles()
            if tile["web"]["mode"] == "route"
            and tile["web"]["route"] not in router_paths
        ]
        assert not bad, (
            f"route-mode tiles whose route is not in the React router: {bad}. "
            f"Known routes: {sorted(router_paths)}"
        )

    def test_loader_exposes_contract_on_all_tiles(self) -> None:
        """The typed loader attaches a contract to every tile (incl. derived)."""
        manifest = LauncherManifest.load()
        for tile in manifest.tiles:
            assert tile.web is not None, f"Tile '{tile.id}' has no web contract"
            assert tile.web.mode in WEB_LAUNCH_MODES

    def test_to_dict_serializes_web_contract(self) -> None:
        """API serialization includes the web contract for every tile."""
        manifest = LauncherManifest.load()
        for tile_dict in manifest.to_dict(include_hidden=True)["tiles"]:
            assert "web" in tile_dict, f"{tile_dict['id']} missing 'web' in to_dict"
            assert tile_dict["web"]["mode"] in WEB_LAUNCH_MODES

    def test_unavailable_mode_requires_reason(self) -> None:
        """unavailable contracts must carry a non-empty reason."""
        for tile in _raw_manifest_tiles():
            web = tile["web"]
            if web["mode"] == "unavailable":
                assert isinstance(web.get("reason"), str) and web["reason"].strip(), (
                    f"Tile '{tile['id']}' unavailable mode requires a reason"
                )


class TestWebLaunchContractValidation:
    """DbC validation of the WebLaunchContract schema itself."""

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="web.mode"):
            WebLaunchContract(mode="popup")

    def test_route_mode_requires_route(self) -> None:
        with pytest.raises(ValueError, match="web.route"):
            WebLaunchContract(mode="route")

    def test_route_must_start_with_slash(self) -> None:
        with pytest.raises(ValueError, match="web.route"):
            WebLaunchContract(mode="route", route="tools/foo")

    def test_route_disallowed_for_native_window(self) -> None:
        with pytest.raises(ValueError, match="only valid for mode 'route'"):
            WebLaunchContract(mode="native-window", route="/tools/foo")

    def test_unavailable_requires_reason(self) -> None:
        with pytest.raises(ValueError, match="web.reason"):
            WebLaunchContract(mode="unavailable")

    def test_from_dict_includes_tile_id_in_error(self) -> None:
        with pytest.raises(ValueError, match="my_tile"):
            WebLaunchContract.from_dict({"mode": "bogus"}, tile_id="my_tile")

    def test_derive_prefers_web_route(self) -> None:
        contract = WebLaunchContract.derive(web_route="/tools/x", path="a.py")
        assert contract.mode == "route"
        assert contract.route == "/tools/x"

    def test_derive_falls_back_to_native_window(self) -> None:
        contract = WebLaunchContract.derive(web_route=None, path="a.py")
        assert contract.mode == "native-window"

    def test_derive_unavailable_when_nothing_declared(self) -> None:
        contract = WebLaunchContract.derive(web_route=None, path="")
        assert contract.mode == "unavailable"
        assert contract.reason

    def test_to_dict_round_trip(self) -> None:
        contract = WebLaunchContract(mode="route", route="/chat")
        assert WebLaunchContract.from_dict(contract.to_dict()) == contract
