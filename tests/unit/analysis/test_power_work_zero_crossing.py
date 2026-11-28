"""Positive/negative work and impulse must split at the zero crossing.

The mixin integrates ``max(x, 0)`` and ``min(x, 0)`` with the trapezoid rule.
On a segment whose endpoints straddle zero that clipped trapezoid counts the
whole half-segment instead of the triangle between the sample and the
crossing, so positive and negative totals are inflated (their sum, the net,
is unaffected). Expected values below are closed-form integrals of the
piecewise-linear signal.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.analysis.power_work_metrics import PowerWorkMetricsMixin

pytestmark = pytest.mark.unit


def _make(torque: np.ndarray, velocity: np.ndarray, times: np.ndarray):
    """Concrete mixin instance carrying one joint's torque/velocity/time series."""

    class _Concrete(PowerWorkMetricsMixin):
        pass

    obj = _Concrete()
    obj.times = times
    obj.dt = float(np.mean(np.diff(times)))
    obj.joint_torques = torque.reshape(-1, 1)
    obj.joint_velocities = velocity.reshape(-1, 1)
    obj.joint_positions = np.zeros((len(times), 1))
    obj.ground_forces = None
    obj._work_metrics_cache = {}
    return obj


def test_work_single_sign_change_segment() -> None:
    """A +1 to -1 W segment splits its 1 s span into two 0.25 J triangles."""
    # power goes +1 -> -1 over 1 s: crossing at 0.5 s, each triangle = 0.25 J
    obj = _make(np.array([1.0, -1.0]), np.ones(2), np.array([0.0, 1.0]))
    result = obj.compute_work_metrics(0)
    assert result is not None
    assert result["positive_work"] == pytest.approx(0.25)
    assert result["negative_work"] == pytest.approx(-0.25)
    assert result["net_work"] == pytest.approx(0.0, abs=1e-12)


def test_work_uneven_crossing_position() -> None:
    """Crossing at 1.5 s of a +3 to -1 W ramp yields 2.25 J and -0.25 J."""
    # power +3 -> -1 over 2 s: crossing at 1.5 s
    # positive = 0.5*1.5*3 = 2.25, negative = -(0.5*0.5*1) = -0.25
    obj = _make(np.array([3.0, -1.0]), np.ones(2), np.array([0.0, 2.0]))
    result = obj.compute_work_metrics(0)
    assert result is not None
    assert result["positive_work"] == pytest.approx(2.25)
    assert result["negative_work"] == pytest.approx(-0.25)
    assert result["net_work"] == pytest.approx(2.0)


def test_work_without_sign_change_is_plain_trapezoid() -> None:
    """A same-sign segment keeps the plain trapezoid area with no split."""
    obj = _make(np.array([1.0, 3.0, 2.0]), np.ones(3), np.array([0.0, 1.0, 2.0]))
    result = obj.compute_work_metrics(0)
    assert result is not None
    assert result["positive_work"] == pytest.approx(4.5)
    assert result["negative_work"] == 0.0


def test_impulse_single_sign_change_segment() -> None:
    """Torque +2 to -2 N m over 1 s splits into +/-0.5 N m s impulse triangles."""
    obj = _make(np.array([2.0, -2.0]), np.ones(2), np.array([0.0, 1.0]))
    result = obj.compute_impulse_metrics("torque", 0)
    assert result is not None
    assert result.positive_impulse == pytest.approx(0.5)
    assert result.negative_impulse == pytest.approx(-0.5)
    assert result.net_impulse == pytest.approx(0.0, abs=1e-12)


def test_exact_zero_sample_does_not_split() -> None:
    """A sample landing exactly on zero must not create a spurious split."""
    # power 1, 0, -1 on a 1 s grid: pos = 0.5, neg = -0.5 exactly
    obj = _make(np.array([1.0, 0.0, -1.0]), np.ones(3), np.array([0.0, 1.0, 2.0]))
    result = obj.compute_work_metrics(0)
    assert result is not None
    assert result["positive_work"] == pytest.approx(0.5)
    assert result["negative_work"] == pytest.approx(-0.5)


@pytest.mark.parametrize(
    "left,right,positive,negative",
    [
        (1e308, -1e308, 2.5e307, -2.5e307),
        (-1e308, 1e308, 2.5e307, -2.5e307),
        (1e16, -1.0, 5e15, -5e-17),
        (-1.0, 1e16, 5e15, -5e-17),
    ],
)
@pytest.mark.parametrize("metric", ["work", "impulse"])
def test_crossing_weights_survive_finite_dynamic_range(
    left: float, right: float, positive: float, negative: float, metric: str
) -> None:
    """The small triangle must survive cancellation; opposite peaks stay finite."""
    obj = _make(np.array([left, right]), np.ones(2), np.array([0.0, 1.0]))
    if metric == "work":
        work = obj.compute_work_metrics(0)
        assert work is not None
        actual_positive, actual_negative, actual_net = (
            work["positive_work"],
            work["negative_work"],
            work["net_work"],
        )
    else:
        impulse = obj.compute_impulse_metrics("torque", 0)
        assert impulse is not None
        actual_positive, actual_negative, actual_net = (
            impulse.positive_impulse,
            impulse.negative_impulse,
            impulse.net_impulse,
        )
    assert actual_positive == pytest.approx(positive, rel=1e-14, abs=0.0)
    assert actual_negative == pytest.approx(negative, rel=1e-14, abs=0.0)
    assert actual_net == pytest.approx(positive + negative, rel=1e-14)


@pytest.mark.unit
class TestZeroCrossingDurations15863:
    """FC-012a-1: gen/abs durations split at the interpolated crossing (#15863)."""

    def test_probe_straddle_splits_durations_at_the_crossing(self) -> None:
        """``t=[0,1,2,3], P=[0,0,+900,-100]``: 1.9 s + 0.1 s, one second idle.

        Whole-segment OR reporting (the #15343-era behaviour) claimed
        ``[2.0 s, 1.0 s]`` with zero idle — a 10x absorption error.
        """
        obj = _make(
            np.array([0.0, 0.0, 900.0, -100.0]),
            np.ones(4),
            np.array([0.0, 1.0, 2.0, 3.0]),
        )
        metrics = obj.compute_joint_power_metrics(0)
        assert metrics is not None
        assert metrics.generation_duration == pytest.approx(1.9, abs=1e-9)
        assert metrics.absorption_duration == pytest.approx(0.1, abs=1e-9)

    def test_net_work_follows_sample_times_not_the_scalar_dt(self) -> None:
        """On a non-uniform grid net work integrates over ``np.diff(times)``."""
        obj = _make(
            np.array([1.0, 3.0, 3.0]),
            np.ones(3),
            np.array([0.0, 1.0, 3.0]),
        )
        metrics = obj.compute_joint_power_metrics(0)
        assert metrics is not None
        assert metrics.net_work == pytest.approx(8.0, abs=1e-9)
