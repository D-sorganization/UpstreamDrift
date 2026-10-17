"""Uncertainty components, curvature floors and known limits of readings (OF-05).

Hand-worked oracles; helpers are shared with ``test_readings``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from src.shared.python.force_plates import (
    PlateReading,
    PlateTimeAlignment,
    ReadingOptions,
    VerticalPlateTrial,
    reading_at,
)
from src.shared.python.force_plates.readings import FootSample
from tests.unit.force_plates.test_readings import (
    CONST_COP,
    CONST_COP_R,
    NONUNIFORM,
    _align,
    _hand_trial,
    _opts,
    _placed,
    _quadratic_trial,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _pulse_trial() -> VerticalPlateTrial:
    # Baseline 100 N plus a 1 ms-wide 1000 N Gaussian pulse centred at 7 ms,
    # sampled on [0, 4, 10, 15] ms: only 11.1 N of it is ever seen.
    pulse = [1000.0 * math.exp(-0.5 * ((t - 0.007) / 0.001) ** 2) for t in NONUNIFORM]
    return _hand_trial(
        NONUNIFORM, [100.0 + p for p in pulse], [400.0] * 4, CONST_COP, CONST_COP_R
    )


def _two_sample_trial() -> VerticalPlateTrial:
    return _hand_trial(
        [0.0, 0.010], [100.0, 200.0], [400.0] * 2, [[-0.2, 0.0]] * 2, [[0.2, 0.0]] * 2
    )


def test_limitation_bias_estimate_is_blind_to_an_undersampled_impact_pulse() -> None:
    """PINS a known limit: the bias is an estimate, not a bound.

    The pulse is narrower than two sample intervals, so no second divided
    difference sees it: the interpolated value misses ~989 N while the bias
    estimate stays below 0.1 N.  Callers must exclude or separately bound such
    intervals; this test fails if someone starts claiming otherwise.
    """
    left = reading_at(_placed(_pulse_trial()), 0.007, _align(), _opts()).feet["left"]
    true_load = 1100.0
    assert true_load - left.load_n > 900.0
    assert left.load_bias_n < 0.1


def test_components_are_carried_separately() -> None:
    # Quadratic interval, t = 7 ms, skew 1 ms, k = 2: bias 9 N, rate 14000 N/s.
    reading = reading_at(
        _placed(_quadratic_trial()),
        0.007,
        _align(),
        _opts(coverage=2.0, skew_s=0.001),
    )
    left = reading.feet["left"]
    random = math.hypot(math.hypot(2.0, 1.58), 14.0)
    assert left.load_sigma_random_n == pytest.approx(random)
    assert left.load_bias_n == pytest.approx(9.0)
    assert left.sigma_load_n == pytest.approx(random + 4.5)
    assert reading.coverage == 2.0


def test_cop_components_are_carried_separately() -> None:
    right = reading_at(
        _placed(), 0.0025, _align(), _opts(coverage=2.0, skew_s=0.001)
    ).feet["right"]
    assert right.cop_sigma_random_m == pytest.approx(math.hypot(0.010, 0.006))
    assert right.cop_bias_m == pytest.approx(1.25e-3)


def test_unavailable_cop_has_no_bias_component() -> None:
    left = reading_at(_placed(), 0.0075, _align(), _opts()).feet["left"]
    assert left.cop_bias_m == 0.0
    assert left.cop_sigma_random_m == pytest.approx(0.010)


def test_unavailable_load_curvature_without_a_floor_is_refused() -> None:
    with pytest.raises(ValueError, match="load_curvature_floor"):
        reading_at(_placed(_two_sample_trial()), 0.004, _align(), _opts())


def test_unavailable_cop_curvature_without_a_floor_is_refused() -> None:
    opts = _opts(load_curvature_floor=1e5)
    with pytest.raises(ValueError, match="cop_curvature_floor"):
        reading_at(_placed(_two_sample_trial()), 0.004, _align(), opts)


def test_cop_curvature_with_nan_neighbour_triples_needs_a_floor() -> None:
    # Factory left CoP is NaN at sample 2, so the only triple around the
    # [0, 5] ms interval is unusable although both bracket samples are loaded.
    with pytest.raises(ValueError, match="left.*cop_m.*cop_curvature_floor"):
        reading_at(_placed(), 0.0025, _align(), _opts())
    opts = _opts(cop_curvature_floor=100.0)
    left = reading_at(_placed(), 0.0025, _align(), opts).feet["left"]
    # (0.0025 * 0.0025) / 2 * 100 = 3.125e-4 m
    assert left.cop_bias_m == pytest.approx(3.125e-4)


def test_exact_sample_hit_needs_no_curvature() -> None:
    left = reading_at(_placed(_two_sample_trial()), 0.0, _align(), _opts()).feet["left"]
    assert left.load_bias_n == 0.0


def test_floor_acts_as_a_lower_limit_on_the_estimate() -> None:
    # Estimated M2 = 2e6 -> bias 9 N.  A smaller floor changes nothing; a
    # larger floor 4e6 gives (0.003 * 0.003) / 2 * 4e6 = 18 N.
    trial = _placed(_quadratic_trial())
    low = reading_at(trial, 0.007, _align(), _opts(load_curvature_floor=1e6))
    high = reading_at(trial, 0.007, _align(), _opts(load_curvature_floor=4e6))
    assert low.feet["left"].load_bias_n == pytest.approx(9.0)
    assert high.feet["left"].load_bias_n == pytest.approx(18.0)


@pytest.mark.parametrize("name", ["load_curvature_floor", "cop_curvature_floor"])
@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_curvature_floor_validation(name: str, bad: float) -> None:
    with pytest.raises(ValueError, match=name):
        ReadingOptions(coverage=2.0, **{name: bad})  # type: ignore[arg-type]
    assert getattr(ReadingOptions(coverage=2.0), name) is None


def test_cop_bias_and_rate_may_come_from_different_axes() -> None:
    # Left CoP x: -0.2, -0.2, -0.19 -> dd = 200, M2 = 400, bias 1.25e-3 m,
    # rate 0 on the first interval; y: 0, .001, .002 -> no curvature, rate
    # 0.2 m/s.  An axis mix-up cannot reproduce both numbers.
    left_cop = [[-0.2, 0.0], [-0.2, 0.001], [-0.19, 0.002]]
    trial = _hand_trial(
        [0.0, 0.005, 0.010], [400.0] * 3, [400.0] * 3, left_cop, [[0.2, 0.0]] * 3
    )
    left = reading_at(
        _placed(trial), 0.0025, _align(), _opts(coverage=2.0, skew_s=0.001)
    ).feet["left"]
    assert left.cop_bias_m == pytest.approx(1.25e-3)
    assert left.cop_sigma_random_m == pytest.approx(math.hypot(0.010, 2e-4))
    assert left.sigma_cop_m == pytest.approx(math.hypot(0.010, 2e-4) + 6.25e-4)


def test_skew_rate_at_an_interior_kink_uses_the_steeper_side() -> None:
    # Slopes meet at t = 4 ms: 0 N/s before, 1000 N/s after.
    trial = _hand_trial(
        NONUNIFORM, [100.0, 100.0, 106.0, 106.0], [400.0] * 4, CONST_COP, CONST_COP_R
    )
    left = reading_at(_placed(trial), 0.004, _align(), _opts(skew_s=0.002)).feet["left"]
    assert left.load_sigma_random_n == pytest.approx(
        math.hypot(math.hypot(2.0, 1.0), 2.0)
    )


@pytest.mark.parametrize("field", ["coverage", "skew_s", "offset_shift_s"])
def test_options_reject_bool(field: str) -> None:
    kwargs: dict[str, Any] = {"coverage": 2.0, field: True}
    with pytest.raises(TypeError, match=field):
        ReadingOptions(**kwargs)


def test_reading_rejects_nan_times_and_foot_sample_infinite_cop() -> None:
    foot = FootSample(1.0, 0.1, None, 0.01, 0.002)
    with pytest.raises(ValueError, match="model_time_s"):
        PlateReading(math.nan, 0.0, {"left": foot, "right": foot})
    with pytest.raises(ValueError, match="cop_m"):
        FootSample(1.0, 0.1, np.array([0.0, math.inf, 0.0]), 0.01, 0.002)


def test_clock_offset_sigma_is_profiled_not_folded_into_sigmas() -> None:
    tight = PlateTimeAlignment(0.0, 0.0, "manual", "tight")
    loose = PlateTimeAlignment(0.0, 0.5, "manual", "loose")
    opts = _opts(cop_curvature_floor=1.0)
    a = reading_at(_placed(), 0.0025, tight, opts)
    b = reading_at(_placed(), 0.0025, loose, opts)
    for foot in ("left", "right"):
        assert a.feet[foot].sigma_load_n == b.feet[foot].sigma_load_n
        assert a.feet[foot].sigma_cop_m == b.feet[foot].sigma_cop_m
