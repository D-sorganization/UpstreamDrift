"""Trajectory-level contracts for shoulder-velocity strategy search."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.shoulder_velocity_strategy_search import (
    ShoulderVelocityProgram,
    build_controls,
    evaluate_programs,
    pareto_program_indices,
)
from src.shared.python.simulation_backends import GolfModelParams

pytestmark = pytest.mark.scientific


def test_control_program_has_declared_shoulder_cut_and_wrist_release() -> None:
    program = ShoulderVelocityProgram(
        shoulder_cut_s=0.2,
        shoulder_torque_before_nm=60.0,
        shoulder_torque_after_nm=15.0,
        wrist_release_s=0.15,
        wrist_restrain_nm=10.0,
        wrist_drive_nm=15.0,
    )

    controls = build_controls(program, horizon=5, dt_s=0.1)

    np.testing.assert_allclose(controls[:, 0], [60.0, 60.0, 15.0, 15.0, 15.0])
    np.testing.assert_allclose(controls[:, 1], [-10.0, -10.0, 15.0, 15.0, 15.0])


def test_program_rejects_release_after_shoulder_cut_boundary_error() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        ShoulderVelocityProgram(
            shoulder_cut_s=-0.1,
            shoulder_torque_before_nm=60.0,
            shoulder_torque_after_nm=0.0,
            wrist_release_s=0.2,
            wrist_restrain_nm=10.0,
            wrist_drive_nm=15.0,
        )


def test_small_trajectory_search_reports_closed_transfer_metrics() -> None:
    programs = (
        ShoulderVelocityProgram(0.12, 60.0, 0.0, 0.14, 10.0, 15.0),
        ShoulderVelocityProgram(0.24, 60.0, 30.0, 0.18, 10.0, 15.0),
        ShoulderVelocityProgram(0.30, 60.0, 60.0, 0.22, 10.0, 15.0),
    )

    outcomes = evaluate_programs(programs, GolfModelParams.default())

    assert len(outcomes) == len(programs)
    assert any(outcome.valid_impact for outcome in outcomes)
    for outcome in outcomes:
        assert outcome.transfer_work_closure_residual_j == pytest.approx(0.0, abs=1e-8)
        assert outcome.braking_grip_work_j >= 0.0
        assert outcome.peak_grip_force_n >= 0.0


def test_pareto_programs_balance_speed_braking_and_peak_force() -> None:
    values = np.array(
        [
            [40.0, 3.0, 200.0],
            [42.0, 4.0, 220.0],
            [39.0, 5.0, 250.0],
        ]
    )

    assert pareto_program_indices(values).tolist() == [0, 1]
