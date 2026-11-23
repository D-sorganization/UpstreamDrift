"""Torque-history margin validation for feasible strategies (#15494)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.loop_nullspace.strategy_margins import (
    torque_history_margins,
)

pytestmark = pytest.mark.unit

TIMES = np.array([0.0, 0.1, 0.2, 0.3])
LO = np.array([-10.0, -20.0])
HI = np.array([10.0, 20.0])


def _call(torque, **kw):
    kw.setdefault("tau_lo", LO)
    kw.setdefault("tau_hi", HI)
    return torque_history_margins(TIMES, torque, **kw)


def test_margins_and_narrowest_sample_are_exact() -> None:
    tau = np.array([[0.0, 0.0], [5.0, -10.0], [8.0, 15.0], [-2.0, 0.0]])
    r = _call(tau)
    assert r["within_limits"] is True
    assert r["margin_nm"] == pytest.approx([2.0, 5.0])
    assert r["narrowest"] == {"joint": 0, "sample": 2, "margin_nm": 2.0}
    assert r["violations"] == []


def test_violation_is_reported_not_clipped() -> None:
    tau = np.zeros((4, 2))
    tau[1, 1] = -25.0
    r = _call(tau)
    assert r["within_limits"] is False
    assert r["violations"] == [{"joint": 1, "sample": 1, "excess_nm": 5.0}]
    assert r["narrowest"]["margin_nm"] == pytest.approx(-5.0)


def test_safety_fraction_shrinks_the_envelope() -> None:
    tau = np.zeros((4, 2))
    tau[2, 0] = 8.0
    assert _call(tau)["within_limits"] is True
    r = _call(tau, safety_fraction=0.5)
    assert r["within_limits"] is False
    assert r["violations"][0]["excess_nm"] == pytest.approx(3.0)


def test_slew_is_peak_finite_difference_and_can_fail() -> None:
    tau = np.zeros((4, 2))
    tau[1, 0] = 4.0  # 40 N m/s up, then 40 N m/s down
    r = _call(tau, max_slew_nm_s=30.0)
    assert r["peak_slew_nm_s"] == pytest.approx([40.0, 0.0])
    assert r["slew_ok"] is False
    assert _call(tau, max_slew_nm_s=50.0)["slew_ok"] is True
    assert _call(tau)["slew_ok"] is None  # no slew limit declared


def test_signed_and_positive_work_are_separate() -> None:
    tau = np.array([[2.0, 0.0]] * 4)
    omega = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]])
    r = _call(tau, velocity_rad_s=omega)
    # power per sample: +2,+2,-2,-2; signed trapezoid = 0.2 + 0.0 - 0.2 = 0;
    # positive part [2,2,0,0] integrates to 0.2 + 0.1 = 0.3
    assert r["signed_work_j"] == pytest.approx([0.0, 0.0])
    assert r["positive_work_j"][0] == pytest.approx(0.3)
    assert _call(tau)["signed_work_j"] is None  # unavailable stays unavailable


def test_result_is_json_serialisable() -> None:
    r = _call(np.zeros((4, 2)), max_slew_nm_s=1.0)
    assert json.loads(json.dumps(r)) == r


@pytest.mark.parametrize(
    "mutate",
    [
        lambda kw: kw.update(torque=np.full((4, 2), np.nan)),
        lambda kw: kw.update(torque=np.zeros((3, 2))),
        lambda kw: kw.update(torque=np.zeros((4, 3))),
        lambda kw: kw.update(tau_lo=np.array([1.0, -1.0])),
        lambda kw: kw.update(safety_fraction=0.0),
        lambda kw: kw.update(safety_fraction=1.5),
        lambda kw: kw.update(max_slew_nm_s=-1.0),
        lambda kw: kw.update(velocity_rad_s=np.zeros((4, 3))),
        lambda kw: kw.update(times=np.array([0.0, 0.1, 0.1, 0.3])),
    ],
)
def test_invalid_inputs_are_rejected(mutate) -> None:
    kw = {
        "times": TIMES,
        "torque": np.zeros((4, 2)),
        "tau_lo": LO,
        "tau_hi": HI,
    }
    mutate(kw)
    with pytest.raises(ValueError):
        torque_history_margins(
            kw.pop("times"),
            kw.pop("torque"),
            **kw,
        )
