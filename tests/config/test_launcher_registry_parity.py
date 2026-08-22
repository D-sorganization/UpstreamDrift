"""Desktop/web launcher registry parity (issue #8853) and hygiene (#8863).

``src/config/models.yaml`` is the canonical launcher registry. The PyQt
desktop launcher reads it directly through ``ModelRegistry``; the web/API
surface reads the same registry through ``LauncherManifest.load()``, which
overlays web-only metadata from ``launcher_manifest.json``.

These tests pin that contract:

- every desktop (registry) tile ID appears on the web surface — no more
  structural exclusion of repo-local tools;
- the only web-surface extras are the explicitly declared
  ``WEB_CATALOG_ONLY_TILES``, each carrying an honest web contract;
- retired duplicate-identity aliases never come back;
- ``GET /launcher/tiles`` does not leak hidden alias tiles (#8863);
- ``src.tools.video_analyzer`` exports its full documented API (#8863).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from src.config.launcher_manifest_loader import LauncherManifest
from src.shared.python.config.model_registry import ModelRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_YAML = REPO_ROOT / "src" / "config" / "models.yaml"

pytestmark = pytest.mark.unit

# Tiles that exist only on the web/API catalog. Each entry documents why it
# has no desktop counterpart. Adding a manifest tile that is neither in
# models.yaml nor listed (and justified) here fails the parity test.
WEB_CATALOG_ONLY_TILES: dict[str, str] = {
    "chat_assistant": "web chat page (/chat); desktop equivalent is the sidekick dock",
    "dataset_generator": "web page (/tools/dataset); native path is the MATLAB chooser",
    "character_builder": "web page (/tools/character-builder); native side is a CLI",
    "analysis_tools_api": "web page (/tools/analysis) over REST endpoints",
    "motion_pipeline": "REST pipeline service; no desktop tile",
    "perturbation_analysis": "API-backed catalog entry; no launchable surface",
    "force_overlays": "API-backed catalog entry; no launchable surface",
    "realtime_ws": "WebSocket endpoint catalog entry; no launchable surface",
    "aip": "API-backed catalog entry; no launchable surface",
    "actuator_controls": "API-backed catalog entry; no launchable surface",
    "unreal_integration": "integration library catalog entry; no launchable surface",
    "robotics_module": "python module catalog entry; no launchable surface",
    "tools_calculator_hub": "alias surface over the Tools data processor",
    "pid_generator": "Tools-ported CLI (generate-pid); no GUI yet",
}

# Duplicate-identity aliases retired by #8853; they must never reappear.
RETIRED_ALIASES = frozenset({"cross_engine", "matlab_unified"})


@pytest.fixture(scope="module")
def desktop_ids() -> set[str]:
    """Tile IDs the PyQt desktop launcher builds from the canonical registry."""
    registry = ModelRegistry(config_path=MODELS_YAML)
    return {model.id for model in registry.get_all_models()}


@pytest.fixture(scope="module")
def manifest() -> LauncherManifest:
    return LauncherManifest.load()


class TestRegistryParity:
    def test_every_desktop_tile_reaches_the_web_surface(
        self, desktop_ids: set[str], manifest: LauncherManifest
    ) -> None:
        web_ids = set(manifest.tile_ids)
        missing = desktop_ids - web_ids
        assert not missing, (
            f"Desktop registry tiles structurally excluded from the web "
            f"surface: {sorted(missing)}"
        )

    def test_web_only_extras_are_exactly_the_declared_set(
        self, desktop_ids: set[str], manifest: LauncherManifest
    ) -> None:
        extras = set(manifest.tile_ids) - desktop_ids
        declared = set(WEB_CATALOG_ONLY_TILES)
        assert extras == declared, (
            "Web-surface tile IDs diverged from the declared web-only set.\n"
            f"Undeclared extras: {sorted(extras - declared)}\n"
            f"Stale declarations: {sorted(declared - extras)}\n"
            "Either add the tile to src/config/models.yaml (canonical) or "
            "declare and justify it in WEB_CATALOG_ONLY_TILES."
        )

    def test_web_only_tiles_carry_an_honest_web_contract(
        self, manifest: LauncherManifest
    ) -> None:
        for tile_id in WEB_CATALOG_ONLY_TILES:
            tile = manifest.get_tile(tile_id)
            assert tile is not None, f"declared web-only tile missing: {tile_id}"
            assert tile.web is not None, f"{tile_id} lacks a web contract"
            # native-window is honest only when a real provider backs the
            # launch (e.g. the Tools vendor gitlink).
            allowed = {"route", "unavailable"}
            if tile.provider == "tools":
                allowed.add("native-window")
            assert tile.web.mode in allowed, (
                f"web-only tile {tile_id} claims a web mode it cannot "
                f"honestly back ({tile.web.mode})"
            )

    def test_retired_duplicate_aliases_stay_retired(
        self, manifest: LauncherManifest
    ) -> None:
        leaked = RETIRED_ALIASES & set(manifest.tile_ids)
        assert not leaked, (
            f"Retired duplicate tile aliases resurfaced: {sorted(leaked)} "
            "(use cross_engine_dashboard / matlab_suite)"
        )


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
