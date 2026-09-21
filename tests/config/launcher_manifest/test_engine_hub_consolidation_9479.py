"""TDD tests for issue #9479: consolidate the six engine meta-tiles.

Eleven of the 49 launcher tiles carried ``category: physics_engine``: five
real engines plus six dashboards/meta-tiles layered on top of them. Three of
the six (``bunkershot3d``, ``pendulum_simulator``, ``swing_objective_lab``)
were miscategorized -- they are specialized golf tools, not "pick an engine"
tiles -- and are reclassified to ``simulation``. The remaining three
(``drake_dashboard``, ``mujoco_dashboard``, ``pinocchio_dashboard``) are
genuine per-engine duplicates of the engine tiles themselves (one,
``mujoco_dashboard``, even shares its ``path`` with ``mujoco_unified``) and
are hidden, with ``cross_engine_dashboard`` remaining as the one
comparison/hub entry point. No capability is lost: hidden tiles keep their
path/route and stay resolvable.
"""

from __future__ import annotations

import pytest

from src.config.launcher_manifest_loader import LauncherManifest

CANONICAL_ENGINE_TILE_IDS = frozenset(
    {
        "mujoco_unified",
        "drake_golf",
        "pinocchio_golf",
        "opensim_golf",
        "myosim_suite",
        # matlab_suite is a distinct MATLAB/Simscape provider, not a
        # duplicate meta-tile of one of the five engines above; issue #9479
        # did not identify it (its evidence read the raw manifest JSON,
        # which tags it "external" -- the merged/served category from
        # models.yaml is "physics_engine"). It is out of this
        # consolidation's scope and stays visible.
        "matlab_suite",
    }
)

RETIRED_ENGINE_DASHBOARD_IDS = (
    "drake_dashboard",
    "mujoco_dashboard",
    "pinocchio_dashboard",
)

RECLASSIFIED_TOOL_IDS = ("bunkershot3d", "pendulum_simulator", "swing_objective_lab")


@pytest.fixture
def manifest() -> LauncherManifest:
    """Load the production manifest."""
    return LauncherManifest.load()


class TestPhysicsEngineCategoryIsExactlyTheEngineSet:
    """The physics_engine category is the single, unambiguous engine picker."""

    def test_visible_physics_engine_tiles_are_exactly_the_five_engines(
        self, manifest: LauncherManifest
    ) -> None:
        visible_ids = {t.id for t in manifest.physics_engines}
        assert visible_ids == CANONICAL_ENGINE_TILE_IDS, (
            "Visible physics_engine tiles must be exactly the five engines; "
            f"got {sorted(visible_ids)}"
        )

    @pytest.mark.parametrize("tile_id", sorted(CANONICAL_ENGINE_TILE_IDS))
    def test_every_engine_is_reachable(
        self, manifest: LauncherManifest, tile_id: str
    ) -> None:
        """Every engine reachable before the consolidation is reachable after it."""
        tile = manifest.get_tile(tile_id)
        assert tile is not None, f"Engine tile {tile_id!r} must exist"
        assert not tile.hidden, f"Engine tile {tile_id!r} must not be hidden"
        assert tile.is_physics_engine, (
            f"Engine tile {tile_id!r} must stay physics_engine"
        )
        assert tile.path or tile.web_route, (
            f"Engine tile {tile_id!r} must have a resolvable entry point"
        )


class TestDuplicateEngineDashboardsAreConsolidated:
    """The three per-engine dashboards are hidden, not deleted."""

    @pytest.mark.parametrize("tile_id", RETIRED_ENGINE_DASHBOARD_IDS)
    def test_dashboard_tile_is_hidden_with_reason(
        self, manifest: LauncherManifest, tile_id: str
    ) -> None:
        tile = manifest.get_tile(tile_id)
        assert tile is not None, f"{tile_id!r} must still exist in the manifest"
        assert tile.hidden, f"{tile_id!r} must be hidden after consolidation"
        assert tile.hidden_reason, f"{tile_id!r} must document why it is hidden"
        assert "9479" in tile.hidden_reason or "Engine Hub" in tile.hidden_reason

    @pytest.mark.parametrize("tile_id", RETIRED_ENGINE_DASHBOARD_IDS)
    def test_dashboard_entry_point_still_resolves(
        self, manifest: LauncherManifest, tile_id: str
    ) -> None:
        """Hiding a tile must not remove its capability -- it stays launchable."""
        tile = manifest.get_tile(tile_id)
        assert tile is not None
        assert tile.path or tile.web_route

    def test_cross_engine_dashboard_remains_the_single_comparison_entry_point(
        self, manifest: LauncherManifest
    ) -> None:
        hub = manifest.get_tile("cross_engine_dashboard")
        assert hub is not None
        assert not hub.hidden, "The comparison/hub tile must stay visible"


class TestMiscategorizedToolsReclassified:
    """Specialized tools that are not engine pickers move out of physics_engine."""

    @pytest.mark.parametrize("tile_id", RECLASSIFIED_TOOL_IDS)
    def test_tile_is_no_longer_physics_engine(
        self, manifest: LauncherManifest, tile_id: str
    ) -> None:
        tile = manifest.get_tile(tile_id)
        assert tile is not None
        assert tile.category == "simulation", (
            f"{tile_id!r} should be categorized as 'simulation', not {tile.category!r}"
        )
        assert not tile.is_physics_engine
