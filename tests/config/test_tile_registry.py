"""The single tile registry contract (issue #9412; supersedes #8853's allowlist).

``src/config/models.yaml`` is the one registry both launchers read. These
tests pin that contract without any test-side allowlist:

- both launchers see exactly the tile set the registry declares for their
  surface (desktop = ``surfaces`` containing ``pyqt``, web = ``web``);
- a tile missing from either surface must say why (``surface_reason``) in
  the data file, not in a test;
- readiness is modelled (``maturity``) and honest (no ``ready`` tile opens a
  known placeholder; hidden tiles are never leaked by ``GET /launcher/tiles``);
- the derived registries (``GUIRegistry``, embed adapters, feature names)
  are projections of the same file;
- ``src.tools.video_analyzer`` exports its full documented API (#8863).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from src.config.launcher_manifest_loader import LauncherManifest
from src.config.registry_adapters import (
    embed_adapter_modules,
    feature_names_by_tile,
    filter_desktop_models,
    populate_gui_registry,
)
from src.config.tile_registry import (
    MATURITY_LEVELS,
    REGISTRY_PATH,
    SURFACES,
    TileRegistry,
    TileRegistryError,
    load_tile_registry,
    parse_tile_registry,
)
from src.shared.python.config.model_registry import ModelRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def registry() -> TileRegistry:
    return load_tile_registry()


@pytest.fixture(scope="module")
def desktop_models() -> list:
    """Models the PyQt launcher builds after the surface filter."""
    models = ModelRegistry(config_path=REGISTRY_PATH, discovery_mode="local-only")
    return filter_desktop_models(models.get_all_models())


@pytest.fixture(scope="module")
def manifest() -> LauncherManifest:
    return LauncherManifest.load(include_provider_tiles=False)


class TestOneTileSet:
    def test_registry_file_is_the_single_source(self, registry: TileRegistry) -> None:
        assert registry.path == REGISTRY_PATH
        assert registry.native_tiles(), "models: list is empty"
        assert registry.web_catalog_tiles(), "web_catalog: list is empty"

    def test_desktop_launcher_sees_exactly_the_pyqt_surface(
        self, registry: TileRegistry, desktop_models: list
    ) -> None:
        desktop_visible = {m.id for m in desktop_models if not m.hidden}
        assert desktop_visible == registry.surface_ids("pyqt")

    def test_web_launcher_sees_exactly_the_web_surface(
        self, registry: TileRegistry, manifest: LauncherManifest
    ) -> None:
        web_visible = {t.id for t in manifest.visible_tiles}
        assert web_visible == registry.surface_ids("web")

    def test_both_launchers_see_the_same_tile_set_unless_justified(
        self, registry: TileRegistry, desktop_models: list, manifest: LauncherManifest
    ) -> None:
        desktop = {m.id for m in desktop_models}
        web = {t.id for t in manifest.tiles}
        justified = {t.id for t in registry.tiles if t.surface_reason}
        assert (desktop ^ web) <= justified, (
            "tiles on only one launcher without a surface_reason in models.yaml: "
            f"{sorted((desktop ^ web) - justified)}"
        )
        assert desktop | web == set(registry.ids)

    def test_every_partial_surface_carries_a_substantive_reason(
        self, registry: TileRegistry
    ) -> None:
        for tile in registry.tiles:
            if set(tile.surfaces) != set(SURFACES):
                assert tile.surface_reason and len(tile.surface_reason) >= 15, (
                    f"{tile.id}: surfaces {tile.surfaces} need a surface_reason"
                )

    def test_web_catalog_entries_never_reach_the_desktop(
        self, registry: TileRegistry, desktop_models: list
    ) -> None:
        desktop = {m.id for m in desktop_models}
        leaked = {t.id for t in registry.web_catalog_tiles()} & desktop
        assert not leaked, f"web-only catalog entries on the desktop: {sorted(leaked)}"

    def test_retired_duplicate_aliases_stay_retired(
        self, registry: TileRegistry
    ) -> None:
        assert not {"cross_engine", "matlab_unified"} & set(registry.ids)


class TestMaturity:
    def test_every_tile_declares_a_valid_maturity(self, registry: TileRegistry) -> None:
        assert {t.maturity for t in registry.tiles} <= MATURITY_LEVELS

    def test_hidden_and_maturity_agree(self, registry: TileRegistry) -> None:
        for tile in registry.tiles:
            assert tile.hidden == (tile.maturity == "hidden"), tile.id

    def test_no_ready_tile_opens_a_known_placeholder(
        self, registry: TileRegistry
    ) -> None:
        """#8883: the Video Analyzer fallback is a bare placeholder label."""
        video = registry.get("video_analyzer")
        assert video is not None and video.maturity != "ready"
        for tile in registry.tiles:
            if tile.maturity != "ready" or not tile.path.startswith("src/tools/"):
                continue
            target = REPO_ROOT / tile.path
            if target.is_file():
                assert "GUI placeholder" not in target.read_text(encoding="utf-8"), (
                    f"ready tile {tile.id} points at a placeholder GUI: {tile.path}"
                )

    def test_jaxsim_is_registered_but_hidden_with_a_reason(
        self, registry: TileRegistry
    ) -> None:
        """#8870: JaxSim is half-integrated; the tile exists but cannot mislead."""
        tile = registry.get("jaxsim_dashboard")
        assert tile is not None
        assert tile.maturity == "hidden" and tile.hidden
        assert "8870" in (tile.hidden_reason or "")

    def test_maturity_distribution_is_reported(self, registry: TileRegistry) -> None:
        counts = registry.maturity_counts()
        assert sum(counts.values()) == len(registry.tiles)
        assert counts["ready"] > 0 and counts["hidden"] >= 3


class TestDerivedRegistries:
    def test_gui_registry_is_derived_from_the_tile_registry(
        self, registry: TileRegistry
    ) -> None:
        from src.shared.python.gui_launcher.registry import GUIRegistry

        scratch = GUIRegistry()
        registered = populate_gui_registry(registry, gui_registry=scratch)
        assert set(registered) == registry.surface_ids("pyqt")
        assert {r.tool_name for r in scratch.list_tools()} == set(registered)

    def test_embed_adapters_come_from_the_registry(
        self, registry: TileRegistry
    ) -> None:
        from src.launchers.embedded_tool_bootstrap import _adapter_modules_for_bootstrap

        declared = embed_adapter_modules(registry)
        assert declared, "models.yaml declares no embed_adapter"
        assert set(declared) <= set(_adapter_modules_for_bootstrap())

    def test_engine_tiles_map_onto_runtime_features(
        self, registry: TileRegistry
    ) -> None:
        names = feature_names_by_tile(registry)
        assert names["mujoco_unified"] == "mujoco"
        assert names["drake_golf"] == "drake"
        assert "jaxsim_dashboard" in names

    def test_feature_bindings_match_the_parity_ledger(
        self, registry: TileRegistry
    ) -> None:
        from src.config.feature_parity_loader import FeatureParityRegistry

        parity = FeatureParityRegistry.load()
        by_feature = {e.feature_id: set(e.tiles) for e in parity.entries}
        for feature_id, tiles in registry.feature_bindings().items():
            assert by_feature.get(feature_id) == set(tiles), feature_id
        assert parity.covered_tile_ids == set(registry.ids)


class TestRegistryHygiene:
    def test_tiles_endpoint_excludes_hidden_aliases(self) -> None:
        """GET /launcher/tiles must not leak hidden tiles (issue #8863)."""
        from src.api.launcher_manifest_cache import invalidate_manifest_cache
        from src.api.routes import launcher as launcher_routes

        invalidate_manifest_cache()
        try:
            tiles = asyncio.run(launcher_routes.get_tiles())
        finally:
            invalidate_manifest_cache()
        leaked = [t["id"] for t in tiles if t.get("hidden")]
        assert not leaked, f"hidden tiles leaked by /launcher/tiles: {leaked}"
        ids = {t["id"] for t in tiles}
        assert "starting_pose_matcher" not in ids
        assert "jaxsim_dashboard" not in ids

    def test_video_analyzer_public_api_is_not_silently_truncated(self) -> None:
        """__all__ keeps the documented analysis API (issue #8863)."""
        import src.tools.video_analyzer as va

        assert {
            "SwingAnalyzer",
            "Landmark",
            "PoseFrame",
            "PostureMetrics",
            "VideoAnalyzerAdapter",
        } <= set(va.__all__)

    def test_help_paths_exist_when_declared(self, registry: TileRegistry) -> None:
        for tile in registry.tiles:
            if tile.help:
                assert (REPO_ROOT / tile.help).is_file(), f"{tile.id}: {tile.help}"


class TestContracts:
    @staticmethod
    def _entry(**overrides: object) -> dict:
        base: dict = {
            "id": "t",
            "name": "T",
            "description": "d",
            "type": "special_app",
            "path": "src/x.py",
            "order": 1,
            "maturity": "ready",
            "surfaces": ["pyqt", "web"],
            "help": None,
            "feature_id": None,
            "launcher": {"category": "tool", "logo": "x.svg", "status": "ready"},
        }
        base.update(overrides)
        return base

    def test_invalid_maturity_rejected(self) -> None:
        with pytest.raises(TileRegistryError, match="maturity"):
            parse_tile_registry({"models": [self._entry(maturity="done")]})

    def test_partial_surface_without_reason_rejected(self) -> None:
        with pytest.raises(TileRegistryError, match="surface_reason"):
            parse_tile_registry({"models": [self._entry(surfaces=["pyqt"])]})

    def test_hidden_without_hidden_maturity_rejected(self) -> None:
        with pytest.raises(TileRegistryError, match="must agree"):
            parse_tile_registry(
                {
                    "models": [
                        self._entry(hidden=True, hidden_reason="r", hidden_owner="o")
                    ]
                }
            )

    def test_duplicate_order_rejected(self) -> None:
        with pytest.raises(TileRegistryError, match="duplicate order"):
            parse_tile_registry({"models": [self._entry(id="a"), self._entry(id="b")]})

    def test_web_catalog_cannot_claim_pyqt(self) -> None:
        with pytest.raises(TileRegistryError, match="pyqt surface"):
            parse_tile_registry(
                {
                    "models": [],
                    "web_catalog": [self._entry(surfaces=["pyqt", "web"], path="")],
                }
            )

    def test_route_mode_requires_leading_slash(self) -> None:
        with pytest.raises(TileRegistryError, match="web.route"):
            parse_tile_registry(
                {"models": [self._entry(web={"mode": "route", "route": "x"})]}
            )
