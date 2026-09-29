"""Unit tests for ``cost.compute_total_work``."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.motion_matching.final_cost import (
    SimOutput,
    compute_total_work,
)


def _sim(time: np.ndarray, tau: np.ndarray, omega: np.ndarray) -> SimOutput:
    return SimOutput(
        butt=np.zeros((time.size, 3)),
        clubhead=np.zeros((time.size, 3)),
        club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (time.size, 1)),
        time=time,
        tau=tau,
        omega=omega,
    )


def test_zero_torque_yields_zero_work() -> None:
    time = np.linspace(0.0, 1.0, 11)
    tau = np.zeros((11, 3))
    omega = np.random.default_rng(0).standard_normal((11, 3))
    assert compute_total_work(_sim(time, tau, omega)) == 0.0


def test_constant_torque_constant_omega_handcalc() -> None:
    # tau=2 N*m, omega=3 rad/s, two joints, T=0.5 s
    # power per joint = |2*3| = 6, summed across 2 joints = 12, integrated 0.5 = 6.0
    time = np.linspace(0.0, 0.5, 1001)
    tau = np.full((1001, 2), 2.0)
    omega = np.full((1001, 2), 3.0)
    w = compute_total_work(_sim(time, tau, omega))
    assert abs(w - 6.0) < 1e-12


def test_eccentric_and_concentric_count_equally() -> None:
    time = np.linspace(0.0, 1.0, 1001)
    tau = np.full((1001, 1), 1.0)
    # Sign-flipped omega should integrate to the same magnitude
    omega_pos = np.full((1001, 1), 4.0)
    omega_neg = np.full((1001, 1), -4.0)
    w_pos = compute_total_work(_sim(time, tau, omega_pos))
    w_neg = compute_total_work(_sim(time, tau, omega_neg))
    assert abs(w_pos - w_neg) < 1e-12


def test_compute_total_work_missing_field_raises() -> None:
    time = np.linspace(0.0, 1.0, 11)
    sim = SimOutput(
        butt=np.zeros((11, 3)),
        clubhead=np.zeros((11, 3)),
        club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (11, 1)),
        time=time,
        tau=None,
        omega=None,
    )
    with pytest.raises(ValueError, match="tau"):
        compute_total_work(sim)


def test_compute_total_work_shape_mismatch_raises() -> None:
    time = np.linspace(0.0, 1.0, 11)
    tau = np.zeros((10, 2))
    omega = np.zeros((11, 2))
    with pytest.raises(ValueError):
        compute_total_work(_sim(time, tau, omega))
