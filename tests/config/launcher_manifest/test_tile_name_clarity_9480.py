"""TDD tests for issue #9480: confusable tile name pairs.

``data_explorer``/``data_processor`` and ``video_analyzer``/``video_processor``
had names that did not tell a user which one to click. Each tile's ``name``
now states its distinguishing outcome, and each ``description`` calls out
what the *other* tile in the pair is for.
"""

from __future__ import annotations

import pytest

from src.config.launcher_manifest_loader import LauncherManifest

pytestmark = pytest.mark.unit

CONFUSABLE_PAIRS = [
    ("data_explorer", "data_processor"),
    ("video_analyzer", "video_processor"),
]


@pytest.fixture
def manifest() -> LauncherManifest:
    """Load the production manifest."""
    return LauncherManifest.load()


class TestConfusablePairsAreDistinguishable:
    @pytest.mark.parametrize("tile_id_a,tile_id_b", CONFUSABLE_PAIRS)
    def test_names_differ_beyond_the_shared_prefix(
        self, manifest: LauncherManifest, tile_id_a: str, tile_id_b: str
    ) -> None:
        tile_a = manifest.get_tile(tile_id_a)
        tile_b = manifest.get_tile(tile_id_b)
        assert tile_a is not None and tile_b is not None
        assert tile_a.name != tile_b.name
        # Each name must carry a parenthetical (or equivalent) qualifier that
        # states the outcome, not just the bare noun.
        assert "(" in tile_a.name, f"{tile_id_a!r} name lacks a clarifying qualifier"
        assert "(" in tile_b.name, f"{tile_id_b!r} name lacks a clarifying qualifier"

    @pytest.mark.parametrize("tile_id_a,tile_id_b", CONFUSABLE_PAIRS)
    def test_each_description_points_to_its_sibling(
        self, manifest: LauncherManifest, tile_id_a: str, tile_id_b: str
    ) -> None:
        tile_a = manifest.get_tile(tile_id_a)
        tile_b = manifest.get_tile(tile_id_b)
        assert tile_a is not None and tile_b is not None
        assert tile_b.name.split(" (")[0] in tile_a.description, (
            f"{tile_id_a!r} description should reference {tile_id_b!r} by name"
        )
        assert tile_a.name.split(" (")[0] in tile_b.description, (
            f"{tile_id_b!r} description should reference {tile_id_a!r} by name"
        )
