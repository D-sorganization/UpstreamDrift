"""Unit tests for sg_optimizer.course.library — classic holes library."""

from __future__ import annotations

import pytest

from src.shared.python.sg_optimizer.course.library import (
    list_classics,
    load_classic,
)


# ---------------------------------------------------------------------------
# list_classics
# ---------------------------------------------------------------------------


def test_list_classics_returns_all_five_slugs():
    slugs = list_classics()
    assert set(slugs) == {
        "sawgrass_17",
        "augusta_13",
        "pebble_7",
        "road_hole_17",
        "cypress_16",
    }


def test_list_classics_is_sorted():
    slugs = list_classics()
    assert slugs == sorted(slugs)


def test_list_classics_returns_list_of_strings():
    slugs = list_classics()
    assert isinstance(slugs, list)
    for slug in slugs:
        assert isinstance(slug, str)


# ---------------------------------------------------------------------------
# load_classic
# ---------------------------------------------------------------------------


def test_load_classic_sawgrass_17():
    hole = load_classic("sawgrass_17")
    assert hole.hole_number == 17
    assert hole.par == 3
    assert hole.yardage == 137
    assert (
        "Sawgrass" in hole.name or "sawgrass" in hole.name.lower() or "17" in hole.name
    )


def test_load_classic_augusta_13():
    hole = load_classic("augusta_13")
    assert hole.hole_number == 13
    assert hole.par == 5
    assert hole.yardage == 510


def test_load_classic_pebble_7():
    hole = load_classic("pebble_7")
    assert hole.hole_number == 7
    assert hole.par == 3
    assert hole.yardage == 106


def test_load_classic_road_hole_17():
    hole = load_classic("road_hole_17")
    assert hole.hole_number == 17
    assert hole.par == 4
    assert hole.yardage == 455


def test_load_classic_cypress_16():
    hole = load_classic("cypress_16")
    assert hole.hole_number == 16
    assert hole.par == 3
    assert hole.yardage == 231


def test_load_classic_all_have_tee_and_green():
    for slug in list_classics():
        hole = load_classic(slug)
        assert hole.tee is not None, f"{slug}: missing tee"
        assert hole.green_center is not None, f"{slug}: missing green_center"
        # Tee and green should be different points.
        assert (
            hole.tee.lat != hole.green_center.lat
            or hole.tee.lon != hole.green_center.lon
        ), f"{slug}: tee == green_center"


def test_load_classic_unknown_slug_raises():
    from src.shared.python.contracts import ContractViolationError

    with pytest.raises(ContractViolationError):
        load_classic("nonexistent_hole_99")


def test_load_classic_returns_hole_geometry():
    from src.shared.python.sg_optimizer.course.course_io import HoleGeometry

    hole = load_classic("sawgrass_17")
    assert isinstance(hole, HoleGeometry)
