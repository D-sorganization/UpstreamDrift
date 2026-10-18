"""Per-world-axis random 1-sigma of the CoP reading (overlay CoP ellipses).

The reading reports the CoP's random 1-sigma for world x and y separately:
``hypot(declared CoP sigma, skew * world-axis rate)``, the world rates being
the stance rates carried by ``|R_xy|`` exactly as for the worst-axis value.
The existing scalar ``cop_sigma_random_m`` is their maximum.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.force_plates import (
    PlacedTrial,
    PlatePlacement,
    PlateTimeAlignment,
    ReadingOptions,
    reading_at,
)
from src.shared.python.force_plates.readings import FootSample
from tests.unit.force_plates import _trial_factory as tf

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_ALIGN = PlateTimeAlignment(
    offset_s=0.0, sigma_offset_s=0.001, kind="manual", note="test clock"
)
# Right CoP of the factory trial in [0, 0.005]: stance x slope 1 m/s, y slope
# 6 m/s (see test_readings.test_cop_sigma_composes_measurement_skew_and_bias);
# declared CoP sigma 0.010 m, skew 1 ms.
_BASE, _SKEW = 0.010, 1.0e-3
_OPTS = ReadingOptions(coverage=2.0, skew_s=_SKEW, cop_curvature_floor=1.0)


def _right(yaw: float) -> FootSample:
    placement = PlatePlacement.from_yaw(yaw, origin_m=[0.0, 0.0, 0.0])
    placed = PlacedTrial(tf.make_trial(), placement)
    return reading_at(placed, 0.0025, _ALIGN, _OPTS).feet["right"]


def test_per_axis_sigma_uses_each_world_axis_rate() -> None:
    right = _right(0.0)
    assert right.cop_sigma_random_xy_m == pytest.approx(
        (math.hypot(_BASE, _SKEW * 1.0), math.hypot(_BASE, _SKEW * 6.0))
    )


def test_a_quarter_turn_swaps_the_world_axes() -> None:
    sx, sy = _right(math.pi / 2).cop_sigma_random_xy_m
    assert (sx, sy) == pytest.approx(
        (math.hypot(_BASE, _SKEW * 6.0), math.hypot(_BASE, _SKEW * 1.0))
    )


@pytest.mark.parametrize("yaw", [0.0, math.pi / 6, math.pi / 4, math.pi / 2])
def test_the_worst_axis_scalar_is_the_larger_axis(yaw: float) -> None:
    right = _right(yaw)
    assert max(right.cop_sigma_random_xy_m) == pytest.approx(right.cop_sigma_random_m)


def test_an_unloaded_cop_reports_the_declared_sigma_on_both_axes() -> None:
    placed = PlacedTrial(tf.make_trial(), PlatePlacement.identity())
    left = reading_at(placed, 0.0075, _ALIGN, ReadingOptions(coverage=2.0)).feet["left"]
    assert left.cop_m is None
    assert left.cop_sigma_random_xy_m == pytest.approx((_BASE, _BASE))


@pytest.mark.parametrize(
    "bad", [(0.01,), (0.01, -0.01), (0.01, math.nan), (0.01, math.inf)]
)
def test_per_axis_sigma_is_validated(bad: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="cop_sigma_random_xy_m"):
        FootSample(
            load_n=1.0,
            sigma_load_n=0.0,
            cop_m=None,
            sigma_cop_m=0.0,
            sigma_sensing_m=0.0,
            cop_sigma_random_xy_m=bad,  # type: ignore[arg-type]
        )
