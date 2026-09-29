"""Manifest coverage for canonical-core shell tiles."""

from __future__ import annotations

import pytest

from src.config.launcher_manifest_loader import LauncherManifest

pytestmark = [pytest.mark.unit]


@pytest.fixture
def manifest() -> LauncherManifest:
    return LauncherManifest.load()


@pytest.mark.parametrize(
    ("tile_id", "route"),
    [
        ("canonical_core_estimation", "/tools/canonical-core/estimation"),
        ("canonical_core_comparison", "/tools/canonical-core/comparison"),
    ],
)
def test_canonical_core_tiles_are_dual_shell_tools(
    manifest: LauncherManifest,
    tile_id: str,
    route: str,
) -> None:
    tile = manifest.get_tile(tile_id)

    assert tile is not None
    assert tile.category == "biomechanics"
    assert tile.is_tool
    assert tile.web_route == route
    assert tile.default_launch == "tab"
    assert tile.shell_surfaces == ("pyqt6", "react")


def test_api_manifest_serializes_canonical_core_shell_metadata(
    manifest: LauncherManifest,
) -> None:
    tiles = {
        tile["id"]: tile
        for tile in manifest.to_dict()["tiles"]
        if tile["id"].startswith("canonical_core_")
    }

    assert set(tiles) == {"canonical_core_estimation", "canonical_core_comparison"}
    assert tiles["canonical_core_estimation"]["shell_surfaces"] == ["pyqt6", "react"]
