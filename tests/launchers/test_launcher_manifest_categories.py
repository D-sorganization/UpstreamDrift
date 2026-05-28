"""Tests for launcher manifest category coverage — issues #5509, #5510, #5511, #5514, #5538.

Verifies that:
- Every sidebar category (biomechanics, simulation, motion_matching) has at
  least one visible tile in the combined manifest.
- Specific tiles added/fixed by this PR exist with the correct category.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "config" / "launcher_manifest.json"
)

# Categories that the sidebar exposes as filter buttons (issues #5509).
SIDEBAR_CATEGORIES = {"biomechanics", "simulation", "motion_matching"}


def _load_manifest_tiles() -> list[dict]:
    """Load only the base JSON manifest tiles (no provider augmentation)."""
    assert MANIFEST_PATH.exists(), f"Manifest not found: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [t for t in raw["tiles"] if not t.get("hidden", False)]


def _tiles_by_category(tiles: list[dict], category: str) -> list[dict]:
    return [t for t in tiles if t.get("category") == category]


# ---------------------------------------------------------------------------
# Issue #5509 — every sidebar category must have >= 1 tile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("category", sorted(SIDEBAR_CATEGORIES))
def test_sidebar_category_has_at_least_one_tile(category: str) -> None:
    """Each sidebar filter category must resolve to at least one visible tile.

    Fixes #5509.
    """
    tiles = _load_manifest_tiles()
    matching = _tiles_by_category(tiles, category)
    assert matching, (
        f"Sidebar category '{category}' has no visible tiles in the manifest. "
        f"Available categories: {sorted({str(t.get('category')) for t in tiles if t.get('category') is not None})}"
    )


# ---------------------------------------------------------------------------
# Issue #5510 — cross_engine_dashboard tile exists and is in 'simulation'
# ---------------------------------------------------------------------------


def test_cross_engine_dashboard_tile_exists() -> None:
    """cross_engine tile must exist in the manifest.

    Fixes #5510.
    """
    tiles = _load_manifest_tiles()
    ids = [t["id"] for t in tiles]
    assert "cross_engine" in ids, f"Tile 'cross_engine' not found. Present tiles: {ids}"


def test_cross_engine_dashboard_category_is_simulation() -> None:
    """cross_engine tile must have category 'simulation'.

    Fixes #5510.
    """
    tiles = _load_manifest_tiles()
    tile = next((t for t in tiles if t["id"] == "cross_engine"), None)
    assert tile is not None, "Tile 'cross_engine' not found in manifest"
    assert tile["category"] == "simulation", (
        f"Expected category 'simulation', got '{tile['category']}'"
    )


# ---------------------------------------------------------------------------
# Issue #5511 — biomech_exercise tile exists and is in 'biomechanics'
# ---------------------------------------------------------------------------


def test_exercise_dashboard_tile_exists() -> None:
    """biomech_exercise tile must exist in the manifest.

    Fixes #5511.
    """
    tiles = _load_manifest_tiles()
    ids = [t["id"] for t in tiles]
    assert "biomech_exercise" in ids, (
        f"Tile 'biomech_exercise' not found. Present tiles: {ids}"
    )


def test_exercise_dashboard_category_is_biomechanics() -> None:
    """biomech_exercise tile must have category 'biomechanics'.

    Fixes #5511.
    """
    tiles = _load_manifest_tiles()
    tile = next((t for t in tiles if t["id"] == "biomech_exercise"), None)
    assert tile is not None, "Tile 'biomech_exercise' not found in manifest"
    assert tile["category"] == "biomechanics", (
        f"Expected category 'biomechanics', got '{tile['category']}'"
    )


# ---------------------------------------------------------------------------
# Issue #5514 — golf_simulation_suite tile exists and is in 'simulation'
# ---------------------------------------------------------------------------


def test_golf_simulation_suite_tile_exists() -> None:
    """golf_simulation_suite tile must exist in the manifest.

    Fixes #5514.
    """
    tiles = _load_manifest_tiles()
    ids = [t["id"] for t in tiles]
    assert "golf_simulation_suite" in ids, (
        f"Tile 'golf_simulation_suite' not found. Present tiles: {ids}"
    )


def test_golf_simulation_suite_category_is_simulation() -> None:
    """golf_simulation_suite tile must have category 'simulation'.

    Fixes #5514.
    """
    tiles = _load_manifest_tiles()
    tile = next((t for t in tiles if t["id"] == "golf_simulation_suite"), None)
    assert tile is not None, "Tile 'golf_simulation_suite' not found in manifest"
    assert tile["category"] == "simulation", (
        f"Expected category 'simulation', got '{tile['category']}'"
    )


# ---------------------------------------------------------------------------
# Issue #5538 — putting_green must be in 'simulation', not 'physics_engine'
# ---------------------------------------------------------------------------


def test_putting_green_category_is_simulation() -> None:
    """putting_green tile must have category 'simulation', not 'physics_engine'.

    Fixes #5538.
    """
    tiles = _load_manifest_tiles()
    tile = next((t for t in tiles if t["id"] == "putting_green"), None)
    assert tile is not None, "Tile 'putting_green' not found in manifest"
    assert tile["category"] == "simulation", (
        f"Expected category 'simulation', got '{tile['category']}'"
    )


# ---------------------------------------------------------------------------
# Bonus: motion_target_preview must be in 'motion_matching'
# ---------------------------------------------------------------------------


def test_motion_target_preview_category_is_motion_matching() -> None:
    """motion_target_preview tile must have category 'motion_matching'."""
    tiles = _load_manifest_tiles()
    tile = next((t for t in tiles if t["id"] == "motion_target_preview"), None)
    assert tile is not None, "Tile 'motion_target_preview' not found in manifest"
    assert tile["category"] == "motion_matching", (
        f"Expected category 'motion_matching', got '{tile['category']}'"
    )
