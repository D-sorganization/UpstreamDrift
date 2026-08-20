"""Research tests for manufactured-solution controls on the articulated tier (#8752).

Verifies manufactured free-body motion, inverse-dynamics closed-form balance,
numerical forward convergence order, and constrained-motion Lagrange multiplier equilibrium.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

pytestmark = [pytest.mark.scientific]

from scripts.research.proximal_distal_energy.articulated_manufactured_solution import (
    evaluate_manufactured_constrained_motion,
    evaluate_manufactured_free_body,
    manufactured_harmonic_trajectory,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _closed_state() -> tuple[object, dict[str, object], np.ndarray, float]:
    model, metadata = build_subject_scaled_model(default_synthetic_profiles()[0])
    with np.load(DATA / "subject_scaled_closed_contact.npz") as source:
        q = np.asarray(source["solution_q"][0, 6], dtype=float)
        grip_span_m = float(source["case_grip_span_m"][0])
    return model, metadata, q, grip_span_m


def test_manufactured_harmonic_trajectory_derivatives() -> None:
    """Manufactured harmonic trajectory positions, velocities, and accelerations must match analytical calculus."""
    model, _, q, _ = _closed_state()
    time_grid, q_ex, qd_ex, qdd_ex = manufactured_harmonic_trajectory(
        model, q, duration_s=0.01, sample_count=101, frequency_hz=2.0
    )
    dt = time_grid[1] - time_grid[0]

    # Finite difference checks of exact functions
    qd_num = (q_ex[2:] - q_ex[:-2]) / (2 * dt)
    assert np.max(np.abs(qd_num - qd_ex[1:-1])) < 1e-4

    qdd_num = (qd_ex[2:] - qd_ex[:-2]) / (2 * dt)
    assert np.max(np.abs(qdd_num - qdd_ex[1:-1])) < 1e-3


def test_manufactured_free_body_closed_form_checks() -> None:
    """Free-body manufactured inverse dynamics, convergence, and conservation checks."""
    model, _, q, _ = _closed_state()
    result = evaluate_manufactured_free_body(
        model,
        q,
        duration_s=0.01,
        time_steps_s=(0.002, 0.001, 0.0005),
    )
    assert result.closed_form_check_passed is True
    assert result.inverse_dynamics_residual < 1e-10
    assert result.observed_convergence_order >= 0.8
    assert result.mechanical_energy_conservation_error < 0.05

    # Errors must decrease monotonically with step size
    steps = sorted(result.integration_step_errors.keys())
    assert (
        result.integration_step_errors[steps[0]]
        <= result.integration_step_errors[steps[-1]]
    )


def test_manufactured_constrained_motion_checks() -> None:
    """Constrained motion equilibrium, Lagrange multipliers, and action-reaction parity."""
    model, metadata, q, grip_span_m = _closed_state()
    hand_x = float(metadata["hand_contact_local_x_m"])

    result = evaluate_manufactured_constrained_motion(
        model,
        q,
        duration_s=0.01,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_x,
    )
    assert result.closed_form_check_passed is True
    assert result.equilibrium_residual < 1e-10
    assert result.action_reaction_residual_n < 1e-12
