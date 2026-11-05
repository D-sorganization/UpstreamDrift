"""Ground bounce/roll model for dispersion tables (#14610)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.sg_optimizer.shot_model.ground_runout import (
    SLOPE_BINS,
    SLOPES,
    SKILL_CLUB,
    rest_position,
)

pytestmark = pytest.mark.unit


def test_slope_bins_change_downrange_rest() -> None:
    carry = np.array([[100.0, 2.0], [100.0, 2.0]])
    speed = np.array([40.0, 40.0])
    flat = rest_position(carry, slope="flat", lie="fairway", landing_speed_ms=speed)
    up = rest_position(carry, slope="uphill", lie="fairway", landing_speed_ms=speed)
    down = rest_position(carry, slope="downhill", lie="fairway", landing_speed_ms=speed)
    assert up[:, 0].mean() < flat[:, 0].mean() < down[:, 0].mean()
    np.testing.assert_allclose(flat[:, 1], carry[:, 1])


def test_rough_stops_shorter_than_fairway() -> None:
    carry = np.array([[120.0, 0.0]])
    speed = np.array([35.0])
    fw = rest_position(carry, slope="flat", lie="fairway", landing_speed_ms=speed)
    ro = rest_position(carry, slope="flat", lie="rough", landing_speed_ms=speed)
    assert ro[0, 0] < fw[0, 0]


def test_slopes_tuple_matches_bins() -> None:
    assert set(SLOPES) == set(SLOPE_BINS)


def test_skill_club_distinct_from_skill_name() -> None:
    assert SKILL_CLUB["mid_iron"] == "7_iron"
    assert SKILL_CLUB["mid_iron"] != "mid_iron"
