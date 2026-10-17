"""Plate readings at an arbitrary model time with an uncertainty budget (OF-05).

Oracles are worked out by hand in the comments; none re-derives the formulas
with the implementation's own helpers.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from src.shared.python.force_plates import (
    PlacedTrial,
    PlatePlacement,
    PlateReading,
    PlateTimeAlignment,
    ReadingOptions,
    VerticalPlateTrial,
    loaded_net_cop,
    reading_at,
)
from src.shared.python.force_plates.readings import FootSample
from tests.unit.force_plates import _trial_factory as tf

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _align(offset_s: float = 0.0) -> PlateTimeAlignment:
    return PlateTimeAlignment(
        offset_s=offset_s, sigma_offset_s=0.001, kind="manual", note="test clock"
    )


def _opts(**kw: float) -> ReadingOptions:
    kw.setdefault("coverage", 2.0)
    return ReadingOptions(**kw)


def _placed(
    trial: VerticalPlateTrial | None = None, placement: PlatePlacement | None = None
) -> PlacedTrial:
    return PlacedTrial(
        trial=trial if trial is not None else tf.make_trial(),
        placement=placement if placement is not None else PlatePlacement.identity(),
    )


def _hand_trial(
    times: list[float],
    left: list[float],
    right: list[float],
    left_cop: list[list[float]],
    right_cop: list[list[float]],
) -> VerticalPlateTrial:
    """Fully loaded hand-built trial (every CoP present)."""
    loads = {"left": np.array(left), "right": np.array(right)}
    cops: dict[str, Any] = {"left": np.array(left_cop), "right": np.array(right_cop)}
    cops["net"] = loaded_net_cop(loads, cops)
    return tf.make_trial(times_s=np.array(times), load_n=loads, cop_m=cops, events_s={})


NONUNIFORM = [0.0, 0.004, 0.010, 0.015]
CONST_COP = [[-0.2, 0.0]] * 4
CONST_COP_R = [[0.2, 0.0]] * 4


def _linear_trial() -> VerticalPlateTrial:
    # left = 100 + 1000 t -> 100, 104, 110, 115 N ; right constant 400 N.
    return _hand_trial(
        NONUNIFORM, [100.0, 104.0, 110.0, 115.0], [400.0] * 4, CONST_COP, CONST_COP_R
    )


def _quadratic_trial() -> VerticalPlateTrial:
    # left = 100 + 1e6 t^2 -> 100, 116, 200, 325 N ; the second divided
    # difference of an a*t^2 series is exactly a = 1e6 on every triple.
    return _hand_trial(
        NONUNIFORM, [100.0, 116.0, 200.0, 325.0], [400.0] * 4, CONST_COP, CONST_COP_R
    )


# ------------------------------------------------------- value objects --
def test_options_validation() -> None:
    assert _opts().skew_s == 0.0 and _opts().offset_shift_s == 0.0
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="coverage"):
            ReadingOptions(coverage=bad)
    for bad in (-0.001, math.nan):
        with pytest.raises(ValueError, match="skew_s"):
            ReadingOptions(coverage=2.0, skew_s=bad)
    with pytest.raises(ValueError, match="offset_shift_s"):
        ReadingOptions(coverage=2.0, offset_shift_s=math.inf)
    with pytest.raises(TypeError, match="coverage"):
        ReadingOptions(coverage="2")  # type: ignore[arg-type]


def test_placed_trial_type_checks_and_span() -> None:
    placed = _placed(_linear_trial())
    assert placed.span_s() == (0.0, 0.015)
    with pytest.raises(TypeError, match="trial"):
        PlacedTrial(trial="x", placement=PlatePlacement.identity())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="placement"):
        PlacedTrial(trial=tf.make_trial(), placement=None)  # type: ignore[arg-type]


def test_foot_sample_cop_is_read_only_and_validated() -> None:
    sample = FootSample(1.0, 0.1, np.array([0.0, 0.0, 0.0]), 0.01, 0.002)
    with pytest.raises(ValueError):
        sample.cop_m[0] = 1.0  # type: ignore[index]
    assert FootSample(1.0, 0.1, None, 0.01, 0.002).cop_m is None
    with pytest.raises(ValueError, match="cop_m"):
        FootSample(1.0, 0.1, np.zeros(2), 0.01, 0.002)
    with pytest.raises(ValueError, match="sigma_load_n"):
        FootSample(1.0, -0.1, None, 0.01, 0.002)
    with pytest.raises(ValueError, match="sigma_cop_m"):
        FootSample(1.0, 0.1, None, math.nan, 0.002)
    with pytest.raises(ValueError, match="sigma_sensing_m"):
        FootSample(1.0, 0.1, None, 0.01, -1.0)
    with pytest.raises(ValueError, match="load_n"):
        FootSample(math.nan, 0.1, None, 0.01, 0.002)


def test_foot_sample_copies_cop_input() -> None:
    source = np.array([1.0, 2.0, 3.0])
    sample = FootSample(1.0, 0.1, source, 0.01, 0.002)
    source[0] = 99.0
    assert sample.cop_m[0] == 1.0  # type: ignore[index]


def test_reading_feet_must_be_keyed_by_feet_and_read_only() -> None:
    foot = FootSample(1.0, 0.1, None, 0.01, 0.002)
    reading = PlateReading(0.0, 0.0, {"left": foot, "right": foot})
    with pytest.raises(TypeError):
        reading.feet["left"] = foot  # type: ignore[index]
    with pytest.raises(ValueError, match="feet"):
        PlateReading(0.0, 0.0, {"left": foot})
    with pytest.raises(TypeError, match="feet"):
        PlateReading(0.0, 0.0, {"left": foot, "right": 1.0})  # type: ignore[dict-item]


# --------------------------------------------------------- interpolation --
def test_exact_sample_hit_returns_the_sample_with_no_bias() -> None:
    reading = reading_at(_placed(_quadratic_trial()), 0.010, _align(), _opts())
    left = reading.feet["left"]
    assert left.load_n == pytest.approx(200.0)
    # hypot(2, 0.01 * 200) = hypot(2, 2); zero skew; bias zero at a sample.
    assert left.sigma_load_n == pytest.approx(math.hypot(2.0, 2.0))
    assert reading.trial_time_s == pytest.approx(0.010)
    assert reading.model_time_s == 0.010


def test_exact_hit_at_both_ends_of_the_span() -> None:
    placed = _placed(_linear_trial())
    assert reading_at(placed, 0.0, _align(), _opts()).feet["left"].load_n == 100.0
    last = reading_at(placed, 0.015, _align(), _opts())
    assert last.feet["left"].load_n == pytest.approx(115.0)


def test_linear_between_nonuniform_samples_is_exact_with_zero_bias() -> None:
    reading = reading_at(_placed(_linear_trial()), 0.007, _align(), _opts())
    left = reading.feet["left"]
    assert left.load_n == pytest.approx(107.0)
    # hypot(2, 0.01 * 107): no bias because every second difference is 0.
    assert left.sigma_load_n == pytest.approx(math.hypot(2.0, 1.07), abs=1e-9)
    assert reading.feet["right"].load_n == pytest.approx(400.0)


def test_nonuniform_first_interval_uses_its_own_width() -> None:
    # t = 0.001 in [0, 0.004]: 100 + 4 * 0.25 = 101.
    left = reading_at(_placed(_linear_trial()), 0.001, _align(), _opts()).feet["left"]
    assert left.load_n == pytest.approx(101.0)


def test_quadratic_bias_bound_matches_the_true_interpolation_error() -> None:
    # Interval [0.004, 0.010], h = 0.006, midpoint 0.007.  Linear interpolation
    # gives 116 + (84 / 0.006) * 0.003 = 158 N; truth is 100 + 1e6 * 4.9e-5 =
    # 149 N, an error of 9 N.  M2 = 2 * 1e6, so
    # bias = (0.003 * 0.003) / 2 * 2e6 = 9 N: the bound is tight here.
    k = 2.0
    left = reading_at(
        _placed(_quadratic_trial()), 0.007, _align(), _opts(coverage=k)
    ).feet["left"]
    assert left.load_n == pytest.approx(158.0)
    expected = math.hypot(2.0, 0.01 * 158.0) + 9.0 / k
    assert left.sigma_load_n == pytest.approx(expected)


def test_coverage_band_contains_the_bias_exactly() -> None:
    # k * sigma_eff = k * hypot(...) + bias, for two different k.
    base = math.hypot(2.0, 1.58)
    for k in (1.0, 3.0):
        left = reading_at(
            _placed(_quadratic_trial()), 0.007, _align(), _opts(coverage=k)
        ).feet["left"]
        assert k * left.sigma_load_n == pytest.approx(k * base + 9.0)


def test_bias_is_smaller_nearer_a_sample() -> None:
    # t = 0.005: (0.001)(0.005)/2 * 2e6 = 5 N ; load 116 + 14000 * 0.001 = 130.
    left = reading_at(
        _placed(_quadratic_trial()), 0.005, _align(), _opts(coverage=1.0)
    ).feet["left"]
    assert left.load_n == pytest.approx(130.0)
    assert left.sigma_load_n == pytest.approx(math.hypot(2.0, 1.3) + 5.0)


def test_end_interval_uses_only_the_available_second_difference() -> None:
    # Three-sample factory trial, right load (400, 500, 795) at (0, .005, .010):
    # f[0,1]=2e4, f[1,2]=5.9e4 -> dd=(5.9e4-2e4)/0.010=3.9e6, M2=7.8e6.
    # At t=0.0025 bias = (0.0025*0.0025)/2*7.8e6 = 24.375 N.
    right = reading_at(_placed(), 0.0025, _align(), _opts(coverage=3.0)).feet["right"]
    assert right.load_n == pytest.approx(450.0)
    expected = math.hypot(2.0, 4.5) + 24.375 / 3.0
    assert right.sigma_load_n == pytest.approx(expected)


def test_two_sample_trial_with_declared_floor_has_floor_bias() -> None:
    trial = _hand_trial(
        [0.0, 0.010],
        [100.0, 200.0],
        [400.0, 400.0],
        [[-0.2, 0.0]] * 2,
        [[0.2, 0.0]] * 2,
    )
    opts = _opts(load_curvature_floor=1e5, cop_curvature_floor=100.0)
    left = reading_at(_placed(trial), 0.004, _align(), opts).feet["left"]
    assert left.load_n == pytest.approx(140.0)
    # bias = (0.004 * 0.006) / 2 * 1e5 = 1.2 N ; coverage 2.
    assert left.load_bias_n == pytest.approx(1.2)
    assert left.sigma_load_n == pytest.approx(math.hypot(2.0, 1.4) + 0.6)
    # CoP: (0.004 * 0.006) / 2 * 100 = 1.2e-3 m.
    assert left.cop_bias_m == pytest.approx(1.2e-3)


def test_skew_term_is_skew_times_first_difference() -> None:
    # Slope of the [0.004, 0.010] interval is 6 / 0.006 = 1000 N/s; a 2 ms
    # skew therefore adds 2 N in quadrature to hypot(2, 1.07).
    left = reading_at(
        _placed(_linear_trial()), 0.007, _align(), _opts(skew_s=0.002)
    ).feet["left"]
    assert left.sigma_load_n == pytest.approx(
        math.hypot(math.hypot(2.0, 1.07), 2.0), abs=1e-9
    )


def test_skew_and_bias_compose_literally() -> None:
    # Quadratic interval slope is 84 / 0.006 = 14000 N/s; skew 1 ms -> 14 N.
    k = 2.0
    left = reading_at(
        _placed(_quadratic_trial()),
        0.007,
        _align(),
        _opts(coverage=k, skew_s=0.001),
    ).feet["left"]
    expected = math.hypot(math.hypot(2.0, 1.58), 14.0) + 9.0 / k
    assert left.sigma_load_n == pytest.approx(expected)


# ------------------------------------------------------ alignment / span --
def test_offset_maps_model_time_to_trial_time() -> None:
    reading = reading_at(_placed(_linear_trial()), 1.007, _align(1.0), _opts())
    assert reading.trial_time_s == pytest.approx(0.007)
    assert reading.model_time_s == 1.007
    assert reading.feet["left"].load_n == pytest.approx(107.0)


def test_offset_shift_profiles_the_common_mode_offset() -> None:
    reading = reading_at(
        _placed(_linear_trial()), 1.007, _align(1.0), _opts(offset_shift_s=0.001)
    )
    assert reading.trial_time_s == pytest.approx(0.006)
    assert reading.feet["left"].load_n == pytest.approx(106.0)


@pytest.mark.parametrize("model_time", [-0.0001, 0.0151])
def test_out_of_range_time_is_refused_not_extrapolated(model_time: float) -> None:
    with pytest.raises(ValueError, match="outside the trial"):
        reading_at(_placed(_linear_trial()), model_time, _align(), _opts())


def test_shift_can_push_a_valid_time_out_of_range() -> None:
    with pytest.raises(ValueError, match="outside the trial"):
        reading_at(_placed(_linear_trial()), 0.0, _align(), _opts(offset_shift_s=0.001))


def test_reading_at_input_validation() -> None:
    placed, al, op = _placed(), _align(), _opts()
    with pytest.raises(TypeError, match="placed"):
        reading_at(tf.make_trial(), 0.0, al, op)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="alignment"):
        reading_at(placed, 0.0, None, op)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="options"):
        reading_at(placed, 0.0, al, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model_time_s"):
        reading_at(placed, math.nan, al, op)
    with pytest.raises(TypeError, match="model_time_s"):
        reading_at(placed, "0", al, op)  # type: ignore[arg-type]


# ------------------------------------------------------------------ CoP --
def test_cop_is_none_when_a_bracketing_sample_is_unloaded() -> None:
    # Left load is 5 N (< 20 N) at sample 2, so its CoP is NaN there.
    placed = _placed()
    between = reading_at(placed, 0.0075, _align(), _opts())
    assert between.feet["left"].cop_m is None
    assert between.feet["right"].cop_m is not None
    at_unloaded = reading_at(placed, 0.010, _align(), _opts())
    assert at_unloaded.feet["left"].cop_m is None
    # The load itself is still reported (5 N is a real, if small, reading).
    assert at_unloaded.feet["left"].load_n == pytest.approx(5.0)
    loaded = reading_at(placed, 0.004, _align(), _opts(cop_curvature_floor=1.0))
    assert loaded.feet["left"].cop_m is not None


def test_cop_world_point_lies_on_the_sensing_plane() -> None:
    # Identity placement, surface z = 0, sensing depth 0.015 m.
    # Right CoP at t=0 is stance (0.185, -0.02).
    right = reading_at(_placed(), 0.0, _align(), _opts()).feet["right"]
    np.testing.assert_allclose(right.cop_m, [0.185, -0.02, -0.015])
    # Midway to sample 1 (0.19, 0.01): (0.1875, -0.005).
    mid = reading_at(_placed(), 0.0025, _align(), _opts()).feet["right"]
    np.testing.assert_allclose(mid.cop_m, [0.1875, -0.005, -0.015])


def test_cop_follows_the_placement_and_surface_height() -> None:
    # Yaw 90 deg: stance (x, y) -> world (-y, x); origin (1, 2, 0.5).
    placement = PlatePlacement.from_yaw(math.pi / 2, origin_m=[1.0, 2.0, 0.5])
    reading = reading_at(_placed(placement=placement), 0.0, _align(), _opts())
    np.testing.assert_allclose(
        reading.feet["right"].cop_m, [1.02, 2.185, 0.485], atol=1e-12
    )


def test_cop_sigma_composes_measurement_skew_and_bias() -> None:
    # Right CoP, factory trial, t = 0.0025, samples at 0, .005, .010:
    #   x: .185, .19, .20 -> slopes 1, 2 -> dd = 100 -> M2 = 200
    #   y: -.02, .01, .03 -> slopes 6, 4 -> dd = -200 -> M2 = 400
    # bias = max((0.0025^2/2)*200, (0.0025^2/2)*400) = 1.25e-3 m
    # rate = max(1, 6) = 6 m/s ; skew 1 ms -> 6e-3 m ; measurement 0.010 m.
    k = 2.0
    right = reading_at(
        _placed(), 0.0025, _align(), _opts(coverage=k, skew_s=0.001)
    ).feet["right"]
    expected = math.hypot(0.010, 0.006) + 1.25e-3 / k
    assert right.sigma_cop_m == pytest.approx(expected)


def test_unavailable_cop_still_reports_a_sigma_budget() -> None:
    left = reading_at(_placed(), 0.0075, _align(), _opts()).feet["left"]
    assert left.cop_m is None
    assert left.sigma_cop_m == pytest.approx(0.010)


def test_sensing_depth_sigma_is_carried_through() -> None:
    reading = reading_at(_placed(), 0.0, _align(), _opts())
    assert reading.feet["left"].sigma_sensing_m == pytest.approx(0.002)
    assert reading.feet["right"].sigma_sensing_m == pytest.approx(0.002)


def test_unloaded_neighbour_does_not_poison_the_other_foot() -> None:
    right = reading_at(_placed(), 0.0075, _align(), _opts()).feet["right"]
    assert math.isfinite(right.sigma_cop_m)
