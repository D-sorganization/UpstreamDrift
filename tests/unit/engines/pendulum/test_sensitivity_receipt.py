"""Tests for pendulum parity sensitivity fields and STM harness (SAP-4, #13793)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.pendulum_simulator.sensitivity import (
    SensitivityReceipt,
    StateScales,
    compute_agreement_horizon,
    compute_linear_system_stm,
    run_pendulum_sensitivity_study,
)
from src.shared.python.pendulum_simulator.step_convergence import (
    BreakpointTorqueProfile,
    InHouseTripleAdapter,
)

pytestmark = pytest.mark.unit


def test_red_sensitivity_receipt_schema_and_required_fields() -> None:
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


def test_dbc_fails_closed_on_invalid_scales_and_fd_steps() -> None:
    with pytest.raises(ValueError, match="fd_step must be finite and positive"):
        SensitivityReceipt(1.0, 0.0, 1.0, (1.0,), 0.0, (1.0,))
    with pytest.raises(ValueError, match="fd_step must be finite and positive"):
        SensitivityReceipt(1.0, 0.0, 1.0, (1.0,), -1e-6, (1.0,))
    with pytest.raises(ValueError, match="state scales must contain finite positive"):
        StateScales((1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="state scales must contain finite positive"):
        StateScales((1.0, float("nan"), 1.0))


def test_agreement_horizon_dbc_and_exact_formula() -> None:
    expected = math.log(0.20 / 1e-6) / 8.3
    assert math.isclose(
        compute_agreement_horizon(0.20, 1e-6, 8.3), expected, rel_tol=1e-12
    )
    assert compute_agreement_horizon(0.20, 1e-6, 0.0) == float("inf")
    assert compute_agreement_horizon(0.20, 1e-6, -2.0) == float("inf")

    with pytest.raises(
        ValueError, match="tolerance and epsilon must be finite and positive"
    ):
        compute_agreement_horizon(-0.1, 1e-6, 8.3)
    with pytest.raises(
        ValueError, match="tolerance and epsilon must be finite and positive"
    ):
        compute_agreement_horizon(0.1, 0.0, 8.3)
    with pytest.raises(ValueError, match="tolerance .* must be greater than"):
        compute_agreement_horizon(1e-7, 1e-6, 8.3)


def test_stm_analytic_linear_system_accuracy() -> None:
    omega, t_end = 2.0, 0.5
    A = np.array([[0.0, 1.0], [-(omega**2), 0.0]], dtype=np.float64)
    wt = omega * t_end
    phi_analytic = np.array(
        [[math.cos(wt), math.sin(wt) / omega], [-omega * math.sin(wt), math.cos(wt)]],
        dtype=np.float64,
    )
    phi_numerical = compute_linear_system_stm(A, t_end=t_end, dt=0.001, fd_step=1e-5)
    max_error = float(np.max(np.abs(phi_numerical - phi_analytic)))
    assert max_error <= 1e-8, f"Analytic STM error {max_error} exceeds 1e-8 threshold"


def test_sigma_max_fd_step_insensitivity_on_canonical_swing() -> None:
    adapter = InHouseTripleAdapter()
    profile = BreakpointTorqueProfile(initial_torque=(80.0, 30.0, 10.0))
    scales = StateScales((1.0, 1.0, 1.0, 10.0, 10.0, 10.0))

    receipt_h1 = run_pendulum_sensitivity_study(
        adapter, profile, scales, t_end=0.2, dt=0.001, fd_step=1e-6
    )
    receipt_h2 = run_pendulum_sensitivity_study(
        adapter, profile, scales, t_end=0.2, dt=0.001, fd_step=1e-5
    )

    rel_diff = abs(receipt_h1.sigma_max - receipt_h2.sigma_max) / receipt_h1.sigma_max
    assert rel_diff <= 0.01
    assert receipt_h1.ftle_per_s > 0.0
    assert math.isfinite(receipt_h1.agreement_horizon_s)
    assert len(receipt_h1.dominant_direction) == 6
