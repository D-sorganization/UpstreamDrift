"""Unit tests for sg_optimizer.course.features (Phase 2).

Tests StateFeatures construction, contract checks, and factory method.
"""

from __future__ import annotations

import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.course.features import StateFeatures
from src.shared.python.sg_optimizer.course.rasterize import (
    CircleFeature,
    LIE_CODES,
    RectFeature,
    SyntheticHole,
    rasterize_synthetic,
)
from src.shared.python.sg_optimizer.mdp.state import State


# ---------------------------------------------------------------------------
# Contract checks
# ---------------------------------------------------------------------------


def test_negative_distance_to_pin_raises():
    with pytest.raises((ContractViolationError, ValueError)):
        StateFeatures(
            distance_to_pin_m=-1.0,
            distance_to_center_m=50.0,
            lie="fairway",
        )


def test_invalid_stimp_raises():
    with pytest.raises((ContractViolationError, ValueError)):
        StateFeatures(
            distance_to_pin_m=50.0,
            distance_to_center_m=50.0,
            lie="green",
            stimp=15.0,  # > 14
        )


def test_negative_wind_raises():
    with pytest.raises((ContractViolationError, ValueError)):
        StateFeatures(
            distance_to_pin_m=50.0,
            distance_to_center_m=50.0,
            lie="fairway",
            wind_mph=-1.0,
        )


def test_invalid_wind_dir_raises():
    with pytest.raises((ContractViolationError, ValueError)):
        StateFeatures(
            distance_to_pin_m=50.0,
            distance_to_center_m=50.0,
            lie="fairway",
            wind_dir_deg=360.0,  # must be < 360
        )


def test_slope_out_of_range_raises():
    with pytest.raises((ContractViolationError, ValueError)):
        StateFeatures(
            distance_to_pin_m=50.0,
            distance_to_center_m=50.0,
            lie="fairway",
            slope_deg=91.0,
        )


# ---------------------------------------------------------------------------
# Valid construction
# ---------------------------------------------------------------------------


def test_valid_state_features():
    sf = StateFeatures(
        distance_to_pin_m=100.0,
        distance_to_center_m=100.0,
        lie="rough",
    )
    assert sf.lie == "rough"
    assert sf.stimp == pytest.approx(10.5)
    assert sf.wind_mph == pytest.approx(0.0)


def test_all_lie_values_accepted():
    for lie in ("fairway", "rough", "bunker", "green", "tee", "ob", "trees", "water"):
        sf = StateFeatures(
            distance_to_pin_m=50.0,
            distance_to_center_m=50.0,
            lie=lie,  # type: ignore[arg-type]
        )
        assert sf.lie == lie


# ---------------------------------------------------------------------------
# Factory: from_state_and_course
# ---------------------------------------------------------------------------


def _make_simple_raster():
    hole = SyntheticHole(
        name="test",
        par=3,
        tee=(0.0, 0.0),
        pin=(120.0, 0.0),
        bbox=(-10.0, 140.0, -20.0, 20.0),
        features=(
            RectFeature("fairway", 0.0, 130.0, -10.0, 10.0),
            CircleFeature("green", 120.0, 0.0, 8.0),
        ),
    )
    return rasterize_synthetic(hole, resolution_yd=5.0)


def test_from_state_distance_approximately_correct():
    raster = _make_simple_raster()
    # State at 50 yards from tee, pin at 120 yards.
    state = State(x=50.0, y=0.0, lie=LIE_CODES["fairway"])
    sf = StateFeatures.from_state_and_course(state, raster)
    # Distance to pin: 70 yd * 0.9144 m/yd ≈ 64.0 m
    assert sf.distance_to_pin_m == pytest.approx(70.0 * 0.9144, rel=0.01)


def test_from_state_lie_str_maps_fairway():
    raster = _make_simple_raster()
    state = State(x=50.0, y=0.0, lie=LIE_CODES["fairway"])
    sf = StateFeatures.from_state_and_course(state, raster)
    assert sf.lie == "fairway"


def test_from_state_with_wind():
    raster = _make_simple_raster()
    state = State(x=0.0, y=0.0, lie=LIE_CODES["fairway"])
    sf = StateFeatures.from_state_and_course(
        state, raster, wind_mph=15.0, wind_dir_deg=270.0
    )
    assert sf.wind_mph == pytest.approx(15.0)
    assert sf.wind_dir_deg == pytest.approx(270.0)


def test_from_state_green_center_override():
    raster = _make_simple_raster()
    state = State(x=60.0, y=0.0, lie=LIE_CODES["fairway"])
    sf_default = StateFeatures.from_state_and_course(state, raster)
    sf_override = StateFeatures.from_state_and_course(
        state, raster, green_center=(100.0, 0.0)
    )
    # With the override, distance_to_center should use (100, 0) not pin.
    expected = 40.0 * 0.9144
    assert sf_override.distance_to_center_m == pytest.approx(expected, rel=0.01)
    # distance_to_pin is unaffected.
    assert sf_default.distance_to_pin_m == sf_override.distance_to_pin_m
