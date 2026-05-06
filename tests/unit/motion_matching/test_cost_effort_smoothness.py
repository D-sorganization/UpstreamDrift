"""Unit tests for the ``effort_l2`` and ``smoothness_l2`` regularizers.

Mirrors the MATLAB tests in
``src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/
shared/tests/test_compute_cost_effort_smoothness.m`` and locks in cross-language
numeric equivalence to ``1e-9`` on a fixture vector. Added for parity with
``MachineLearning/optimize_torque_sequence_for_club.py`` (PR #3966).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from src.shared.python.motion_matching.cost import (
    CostOptions,
    SimOutput,
    compute_cost,
)

from ._fixtures import make_target


def _matching_sim(target, tau: np.ndarray) -> SimOutput:
    """SimOutput identical to ``target`` kinematically, with provided tau."""
    n = target.time.shape[0]
    return SimOutput(
        butt=target.butt.copy(),
        clubhead=target.clubhead.copy(),
        club_quat=target.club_quat.copy(),
        time=target.time.copy(),
        tau=tau,
        omega=np.zeros((n, tau.shape[1])),
    )


def _kinematic_zero_opts(**overrides) -> CostOptions:
    """CostOptions with kinematic terms disabled and lambda=1."""
    base = {
        "w_position": 0.0,
        "w_orientation": 0.0,
        "w_anchor_impact": 0.0,
        "lambda_": 1.0,
    }
    base.update(overrides)
    return CostOptions(**base)


def test_effort_l2_zero_reference_matches_mean_tau_squared() -> None:
    target = make_target(n=31)
    n = target.time.shape[0]
    tau = np.linspace(-1.0, 2.0, n * 3).reshape(n, 3)
    sim = _matching_sim(target, tau)
    opts = _kinematic_zero_opts(regularizer="effort_l2")
    _, terms = compute_cost(np.zeros(7), target, lambda _t: sim, opts)
    expected = float(np.mean(tau * tau))
    assert abs(terms.regularizer - expected) < 1e-12


def test_smoothness_l2_zero_for_constant_torque() -> None:
    target = make_target(n=31)
    n = target.time.shape[0]
    tau = 0.7 * np.ones((n, 3))
    sim = _matching_sim(target, tau)
    opts = _kinematic_zero_opts(regularizer="smoothness_l2")
    _, terms = compute_cost(np.zeros(7), target, lambda _t: sim, opts)
    assert terms.regularizer == 0.0


def test_smoothness_l2_ramp_analytic() -> None:
    """For tau(t)=a*t on uniform grid, mean(diff(tau)^2) = (a*dt)^2."""
    target = make_target(n=31)
    a = 4.5
    dt = float(target.time[1] - target.time[0])
    tau = (a * target.time).reshape(-1, 1)
    sim = _matching_sim(target, tau)
    opts = _kinematic_zero_opts(regularizer="smoothness_l2")
    _, terms = compute_cost(np.zeros(7), target, lambda _t: sim, opts)
    expected = (a * dt) ** 2
    assert terms.regularizer == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_regularizer_weights_honoured() -> None:
    target = make_target(n=31)
    n = target.time.shape[0]
    tau = np.column_stack([np.ones(n), 2.0 * np.ones(n), 3.0 * np.ones(n)])
    sim = _matching_sim(target, tau)
    weights = np.array([0.5, 1.0, 2.0])
    opts = _kinematic_zero_opts(
        regularizer="effort_l2",
        regularizer_weights=weights,
    )
    _, terms = compute_cost(np.zeros(7), target, lambda _t: sim, opts)
    expected = float(np.mean(tau * tau * weights[np.newaxis, :]))
    assert abs(terms.regularizer - expected) < 1e-12


def test_effort_l2_nonzero_reference_zero_residual() -> None:
    target = make_target(n=15)
    n = target.time.shape[0]
    tau = np.ones((n, 3))
    sim = _matching_sim(target, tau)
    opts = _kinematic_zero_opts(
        regularizer="effort_l2",
        tau_reference=np.ones((n, 3)),
    )
    _, terms = compute_cost(np.zeros(7), target, lambda _t: sim, opts)
    assert terms.regularizer == 0.0


# --- MATLAB parity fixture --------------------------------------------------
#
# These two values are computed in closed form from the same fixture as the
# MATLAB test (test_compute_cost_effort_smoothness.m). Cross-language equality
# is locked at 1e-9 to satisfy the PR #3966 parity contract.


def _parity_fixture() -> tuple[np.ndarray, float]:
    """Return (tau, dt) using the MATLAB-side fixture: N=31 over t in [0, 0.3]."""
    n = 31
    t = np.linspace(0.0, 0.3, n)
    dt = float(t[1] - t[0])
    a = 4.5
    tau = (a * t).reshape(-1, 1)
    return tau, dt


def test_matlab_parity_smoothness_l2_value() -> None:
    """The smoothness_l2 ramp value must equal (a*dt)^2 to 1e-9."""
    tau, dt = _parity_fixture()
    a = 4.5
    matlab_value = (a * dt) ** 2  # MATLAB analytic value used in MATLAB test

    target = make_target(n=tau.shape[0])
    sim = _matching_sim(target, tau)
    opts = _kinematic_zero_opts(regularizer="smoothness_l2")
    _, terms = compute_cost(np.zeros(7), target, lambda _t: sim, opts)

    assert abs(terms.regularizer - matlab_value) < 1e-9


def test_matlab_parity_effort_l2_value() -> None:
    """effort_l2 with zero reference equals mean(tau^2) to 1e-9."""
    tau, _ = _parity_fixture()
    matlab_value = float(np.mean(tau * tau))

    target = make_target(n=tau.shape[0])
    sim = _matching_sim(target, tau)
    opts = _kinematic_zero_opts(regularizer="effort_l2")
    _, terms = compute_cost(np.zeros(7), target, lambda _t: sim, opts)

    assert abs(terms.regularizer - matlab_value) < 1e-9


def test_costoptions_new_fields_default_none() -> None:
    opts = CostOptions()
    assert opts.tau_reference is None
    assert opts.regularizer_weights is None
    # frozen dataclass: ensure replace works for the new fields.
    new_opts = dataclasses.replace(opts, regularizer_weights=np.array([1.0]))
    assert new_opts.regularizer_weights is not None
