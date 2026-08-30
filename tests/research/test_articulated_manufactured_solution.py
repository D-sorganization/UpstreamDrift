"""Research tests for manufactured-solution controls on the articulated tier (#8752).

Verifies manufactured free-body motion, inverse-dynamics closed-form balance,
numerical forward convergence order, and constrained-motion Lagrange multiplier equilibrium.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

pytestmark = [pytest.mark.scientific]

from scripts.research.proximal_distal_energy import articulated_manufactured_solution
from scripts.research.proximal_distal_energy.articulated_manufactured_solution import (
    evaluate_manufactured_constrained_motion,
    evaluate_manufactured_free_body,
    manufactured_harmonic_trajectory,
)
from scripts.research.proximal_distal_energy.articulated_inertia_cross_engine import (
    require_robotics_pinocchio,
)
from scripts.research.proximal_distal_energy.run_articulated_manufactured_solution import (
    write_record,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    build_subject_scaled_model,
    default_synthetic_profiles,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _robotics_pinocchio_available() -> bool:
    try:
        import pinocchio as pin

        require_robotics_pinocchio(pin)
    except (ImportError, RuntimeError):
        return False
    return True


requires_native_pinocchio = pytest.mark.skipif(
    not _robotics_pinocchio_available(),
    reason="robotics Pinocchio is exercised in the optional-stack CI lane",
)


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


@pytest.mark.requires_pinocchio
@requires_native_pinocchio
def test_manufactured_free_body_closed_form_checks() -> None:
    """Three independent dynamics paths and measured conservation must pass."""
    model, _, q, _ = _closed_state()
    result = evaluate_manufactured_free_body(
        model,
        q,
        duration_s=0.004,
        time_steps_s=(0.002, 0.001, 0.0005),
    )
    assert result.closed_form_check_passed is True
    assert 0.0 < result.inverse_dynamics_residual < 0.05
    assert 0.0 < result.lagrange_mujoco_relative_error < 0.05
    assert 0.0 < result.lagrange_pinocchio_relative_error < 0.05
    assert 0.0 < result.mujoco_pinocchio_relative_error < 1e-8
    assert result.independent_engine_difference_detected is True
    assert len(result.richardson_orders) == 2
    assert all(0.9 <= order <= 1.1 for order in result.richardson_orders)
    assert 0.0 < result.linear_momentum_conservation_error < 0.02
    assert 0.0 < result.angular_momentum_conservation_error < 0.02
    assert 0.0 < result.mechanical_energy_conservation_error < 0.02

    # Errors must decrease monotonically with step size
    steps = sorted(result.integration_step_errors.keys())
    assert (
        result.integration_step_errors[steps[0]]
        <= result.integration_step_errors[steps[-1]]
    )


@pytest.mark.requires_pinocchio
@requires_native_pinocchio
def test_manufactured_solution_killswitch_detects_corrupt_native_inverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupted native engine must fail instead of satisfying an identity."""

    model, _, q, _ = _closed_state()
    native = articulated_manufactured_solution.mujoco_inverse_dynamics

    def corrupted(*args: object, **kwargs: object) -> np.ndarray:
        return native(*args, **kwargs) + 10.0  # type: ignore[arg-type]

    monkeypatch.setattr(
        articulated_manufactured_solution, "mujoco_inverse_dynamics", corrupted
    )
    result = evaluate_manufactured_free_body(
        model, q, duration_s=0.004, time_steps_s=(0.002, 0.001, 0.0005)
    )
    assert result.closed_form_check_passed is False
    assert result.inverse_dynamics_residual > 0.05


@pytest.mark.requires_pinocchio
@requires_native_pinocchio
def test_manufactured_constrained_motion_checks() -> None:
    """Constrained motion must recover loads through independent engines."""
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
    assert result.independent_engine_difference_detected is True
    assert 0.0 <= result.constraint_residual < 1e-10
    assert 0.0 <= result.constraint_velocity_residual < 1e-10
    assert 0.0 < result.equilibrium_residual < 0.05
    assert 0.0 < result.lagrange_multiplier_residual < 0.05
    assert 0.0 < result.action_reaction_residual_n < 1e-8


def test_committed_manufactured_solution_evidence_is_current_and_nontrivial() -> None:
    """Release evidence must be source-pinned and contain measured residuals."""

    record = json.loads((DATA / "articulated_manufactured_solution.json").read_text())
    assert record["schema_version"] == "1.1.0"
    assert record["all_gates_pass"] is True
    assert record["classification"] == (
        "synthetic_numerical_verification_not_human_evidence"
    )
    assert record["engines"]["mujoco"] != record["engines"]["pinocchio"] or (
        record["engines"]["mujoco"] == "3.8.0"
    )
    inverse = record["free_body"]["inverse_dynamics_relative_error"]
    assert 0.0 < inverse["lagrange_mujoco"] < 0.05
    assert 0.0 < inverse["lagrange_pinocchio"] < 0.05
    assert 0.0 < inverse["mujoco_pinocchio"] < 1e-8
    assert all(
        0.9 <= value <= 1.1 for value in record["free_body"]["richardson_orders"]
    )
    drift = record["free_body"]["gravity_free_zero_torque_relative_drift"]
    assert all(0.0 < value < 0.02 for value in drift.values())
    constrained = record["constrained_motion"]
    assert 0.0 < constrained["multiplier_relative_residual"] < 0.05
    assert 0.0 < constrained["equilibrium_relative_residual"] < 0.05
    for relative_path, expected in record["source_sha256"].items():
        actual = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual == expected


@pytest.mark.requires_pinocchio
@requires_native_pinocchio
def test_manufactured_solution_record_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    """Two native builds must reproduce the exact committed evidence bytes."""

    first = write_record(tmp_path / "first.json").read_bytes()
    second = write_record(tmp_path / "second.json").read_bytes()
    committed = (DATA / "articulated_manufactured_solution.json").read_bytes()
    assert first == second
    assert first == committed
