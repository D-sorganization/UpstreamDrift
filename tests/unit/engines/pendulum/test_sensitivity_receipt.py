"""Tests for pendulum parity sensitivity fields and STM harness (SAP-4, #13793).

Design by Contract
------------------
- Pre: Positive state scales and finite difference step sizes (fail closed).
- Pre: Valid tolerance and epsilon for agreement horizon calculation.
- Post: SensitivityReceipt records sigma_max, FTLE, agreement horizon,
  dominant sensitivity direction, FD step, and state scale conventions.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.pendulum_simulator.sensitivity import (
    SensitivityReceipt,
    StateScales,
    compute_agreement_horizon,
    compute_linear_system_stm,
    finite_time_spectra,
    normalized_transition,
    propagate_state_transition_matrix,
    run_pendulum_sensitivity_study,
)
from src.shared.python.pendulum_simulator.step_convergence import (
    BreakpointTorqueProfile,
    InHouseTripleAdapter,
)

pytestmark = pytest.mark.unit


def test_red_sensitivity_receipt_schema_and_required_fields() -> None:
    """Receipt schema requires sigma_max, ftle, agreement_horizon, dominant_direction, fd_step, state_scales."""
    receipt = SensitivityReceipt(
        sigma_max=124.0,
        ftle_per_s=8.3,
        agreement_horizon_s=0.55,
        dominant_direction=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        fd_step=1e-6,
        state_scales=(1.0, 1.0, 1.0, 10.0, 10.0, 10.0),
        sigma_max_phases={"mid_swing": 10.5, "impact": 124.0},
        metadata={"model": "canonical_3dof"},
    )
    data = receipt.to_dict()
    assert data["sigma_max"] == 124.0
    assert data["ftle_per_s"] == 8.3
    assert data["agreement_horizon_s"] == 0.55
    assert len(data["dominant_direction"]) == 6
    assert data["fd_step"] == 1e-6
    assert data["state_scales"] == [1.0, 1.0, 1.0, 10.0, 10.0, 10.0]
    assert data["sigma_max_phases"]["impact"] == 124.0
    assert data["metadata"]["model"] == "canonical_3dof"


def test_dbc_fails_closed_on_invalid_scales_and_fd_steps() -> None:
    """DbC: nonpositive FD step, nonpositive or nonfinite scales fail closed."""
    with pytest.raises(ValueError, match="fd_step must be finite and positive"):
        SensitivityReceipt(
            sigma_max=1.0,
            ftle_per_s=0.0,
            agreement_horizon_s=1.0,
            dominant_direction=(1.0,),
            fd_step=0.0,
            state_scales=(1.0,),
        )

    with pytest.raises(ValueError, match="fd_step must be finite and positive"):
        SensitivityReceipt(
            sigma_max=1.0,
            ftle_per_s=0.0,
            agreement_horizon_s=1.0,
            dominant_direction=(1.0,),
            fd_step=-1e-6,
            state_scales=(1.0,),
        )

    with pytest.raises(ValueError, match="state scales must contain finite positive"):
        StateScales((1.0, 0.0, 1.0))

    with pytest.raises(ValueError, match="state scales must contain finite positive"):
        StateScales((1.0, float("nan"), 1.0))


def test_agreement_horizon_dbc_and_exact_formula() -> None:
    """Agreement horizon = ln(tol / eps) / lambda; fails closed on invalid inputs."""
    tol = 0.20
    eps = 1e-6
    lam = 8.3
    expected = math.log(tol / eps) / lam
    actual = compute_agreement_horizon(tolerance=tol, epsilon=eps, ftle=lam)
    assert math.isclose(actual, expected, rel_tol=1e-12)

    # When lambda <= 0, horizon is unbounded (inf)
    assert compute_agreement_horizon(tolerance=tol, epsilon=eps, ftle=0.0) == float(
        "inf"
    )
    assert compute_agreement_horizon(tolerance=tol, epsilon=eps, ftle=-2.0) == float(
        "inf"
    )

    # Fail closed on nonpositive or inverted tol/eps
    with pytest.raises(
        ValueError, match="tolerance and epsilon must be finite and positive"
    ):
        compute_agreement_horizon(tolerance=-0.1, epsilon=1e-6, ftle=lam)
    with pytest.raises(
        ValueError, match="tolerance and epsilon must be finite and positive"
    ):
        compute_agreement_horizon(tolerance=0.1, epsilon=0.0, ftle=lam)
    with pytest.raises(ValueError, match="tolerance must be greater than epsilon"):
        compute_agreement_horizon(tolerance=1e-7, epsilon=1e-6, ftle=lam)


def test_stm_analytic_linear_system_accuracy() -> None:
    """STM propagation on an analytic linear oscillator matches matrix exponential <= 1e-8."""
    # Undamped oscillator: q_dot = v, v_dot = -omega^2 * q
    omega = 2.0
    A = np.array([[0.0, 1.0], [-(omega**2), 0.0]], dtype=np.float64)
    t_end = 0.5
    dt = 0.0001  # high resolution integration to reach <= 1e-8

    # Analytic transition matrix Phi(t) = exp(A * t)
    # For harmonic oscillator:
    # Phi = [[cos(w*t), sin(w*t)/w], [-w*sin(w*t), cos(w*t)]]
    wt = omega * t_end
    phi_analytic = np.array(
        [[math.cos(wt), math.sin(wt) / omega], [-omega * math.sin(wt), math.cos(wt)]],
        dtype=np.float64,
    )

    phi_numerical = compute_linear_system_stm(A, t_end=t_end, dt=dt, fd_step=1e-6)
    max_error = float(np.max(np.abs(phi_numerical - phi_analytic)))
    assert max_error <= 1e-8, f"Analytic STM error {max_error} exceeds 1e-8 threshold"


def test_sigma_max_fd_step_insensitivity_on_canonical_swing() -> None:
    """Sigma-max is insensitive across h = 1e-6 and 1e-5 (<= 1% relative difference) on canonical swing."""
    adapter = InHouseTripleAdapter()
    profile = BreakpointTorqueProfile(initial_torque=(80.0, 30.0, 10.0))
    scales = StateScales((1.0, 1.0, 1.0, 10.0, 10.0, 10.0))

    receipt_h1 = run_pendulum_sensitivity_study(
        adapter,
        torque_profile=profile,
        state_scales=scales,
        t_end=0.2,
        dt=0.001,
        fd_step=1e-6,
    )
    receipt_h2 = run_pendulum_sensitivity_study(
        adapter,
        torque_profile=profile,
        state_scales=scales,
        t_end=0.2,
        dt=0.001,
        fd_step=1e-5,
    )

    rel_diff = abs(receipt_h1.sigma_max - receipt_h2.sigma_max) / receipt_h1.sigma_max
    assert rel_diff <= 0.01, (
        f"Sigma-max sensitive to FD step: {receipt_h1.sigma_max} vs {receipt_h2.sigma_max} (rel diff {rel_diff:.4f} > 0.01)"
    )
    assert receipt_h1.ftle_per_s > 0.0
    assert math.isfinite(receipt_h1.agreement_horizon_s)
    assert len(receipt_h1.dominant_direction) == 6
