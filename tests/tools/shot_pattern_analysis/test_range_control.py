"""Geometric common-range control; no new flight dynamics are implied."""

from __future__ import annotations

import math

import pytest

from src.tools.shot_pattern_analysis.range_control import (
    equal_range_endpoint,
    equal_range_summary,
)

pytestmark = pytest.mark.unit


def test_equal_range_preserves_bearing_and_sets_radius() -> None:
    x, y = equal_range_endpoint(180.0, 24.0, 195.333983)
    assert math.hypot(x, y) == pytest.approx(195.333983)
    assert math.atan2(y, x) == pytest.approx(math.atan2(24.0, 180.0))


@pytest.mark.parametrize(
    "x,y,radius",
    [
        (0.0, 0.0, 195.0),
        (float("nan"), 1.0, 195.0),
        (1.0, float("inf"), 195.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, -1.0),
    ],
)
def test_equal_range_rejects_invalid_geometry(
    x: float, y: float, radius: float
) -> None:
    with pytest.raises(ValueError):
        equal_range_endpoint(x, y, radius)


def test_summary_uses_equal_radius_and_target_circle() -> None:
    data = [(195.0, -10.0), (195.0, 0.0), (195.0, 10.0)]
    summary = equal_range_summary(data, 195.0, 10.0)
    assert summary["n"] == 3
    assert summary["lateral_sd_m"] > 0
    assert summary["lateral_variance_m2"] == pytest.approx(summary["lateral_sd_m"] ** 2)
    assert summary["target_hit_fraction"] == pytest.approx(1.0)
    assert summary["range_min_m"] == pytest.approx(195.0)
    assert summary["range_max_m"] == pytest.approx(195.0)
