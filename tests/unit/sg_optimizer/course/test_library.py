"""Unit tests for sg_optimizer.course.library (Phase 2).

Tests the classic-holes loader: slug validation, enumeration, and data
integrity of each backing GeoJSON file.
"""

from __future__ import annotations

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.course.library import (
    list_classics,
    load_classic,
)


# ---------------------------------------------------------------------------
# list_classics
# ---------------------------------------------------------------------------


def test_list_classics_returns_sorted():
    slugs = list_classics()
    assert slugs == sorted(slugs)


def test_list_classics_contains_expected_holes():
    slugs = set(list_classics())
    required = {
        "sawgrass_17",
        "augusta_13",
        "pebble_7",
        "road_hole_17",
        "cypress_16",
    }
    assert required <= slugs, f"missing slugs: {required - slugs}"


def test_list_classics_non_empty():
    assert len(list_classics()) >= 5


# ---------------------------------------------------------------------------
# load_classic — invalid slug
# ---------------------------------------------------------------------------


def test_load_unknown_slug_raises():
    with pytest.raises((ContractViolationError, ValueError)):
        load_classic("nonexistent_hole")


def test_load_empty_slug_raises():
    with pytest.raises((ContractViolationError, ValueError)):
        load_classic("")


# ---------------------------------------------------------------------------
# load_classic — each known hole
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", list_classics())
def test_load_classic_returns_valid_geometry(slug):
    from src.shared.python.sg_optimizer.course.course_io import HoleGeometry

    hole = load_classic(slug)
    assert isinstance(hole, HoleGeometry)
    assert hole.par in (3, 4, 5)
    assert hole.yardage > 0
    assert hole.name
    # Tee and green_center must be distinct lat/lon points.
    assert not (
        hole.tee.lat == hole.green_center.lat and hole.tee.lon == hole.green_center.lon
    ), "tee and green_center must differ"


def test_sawgrass_17_is_par3():
    hole = load_classic("sawgrass_17")
    assert hole.par == 3
    # Island green — should have water polygon.
    assert len(hole.water) >= 1


def test_augusta_13_is_par5():
    hole = load_classic("augusta_13")
    assert hole.par == 5


def test_pebble_7_is_par3():
    hole = load_classic("pebble_7")
    assert hole.par == 3


def test_road_hole_17_is_par4():
    hole = load_classic("road_hole_17")
    assert hole.par == 4


def test_cypress_16_is_par3():
    hole = load_classic("cypress_16")
    assert hole.par == 3
