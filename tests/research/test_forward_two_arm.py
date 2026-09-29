"""Forward constrained two-hand model and killswitch contracts."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.forward_two_arm import (
    ForwardTwoArmConfig,
    branch_zero_command,
    constant_control,
    mechanical_energy,
    rollout_forward_two_arm,
)
from scripts.research.proximal_distal_energy.two_arm_closed_loop import (
    TwoArmControl,
    TwoArmParams,
    constraint_jacobian,
    constraint_vector,
    contact_wrench,
    kinematics,
)

pytestmark = pytest.mark.scientific


def _initial_state(params: TwoArmParams) -> tuple[np.ndarray, np.ndarray]:
    q = params.consistent_configuration(np.array([0.0, -0.50]), 0.16)
    return q, np.zeros_like(q)


def _control() -> TwoArmControl:
    return TwoArmControl(
        right_shoulder_nm=18.0,
        right_elbow_nm=6.0,
        right_wrist_nm=-2.5,
        left_shoulder_nm=16.0,
        left_elbow_nm=5.0,
        left_wrist_nm=1.5,
    )


def test_forward_rollout_preserves_constraints_and_records_solver_evidence() -> None:
    params = TwoArmParams.publication_default()
    q0, v0 = _initial_state(params)
    config = ForwardTwoArmConfig(duration_s=0.03, step_s=0.00025)

    trace = rollout_forward_two_arm(
        q0,
        v0,
        constant_control(_control()),
        params,
        config,
    )

    assert trace.time.size == 121
    assert trace.q.shape == (121, 7)
    assert trace.contact_force_on_club_n.shape == (121, 2, 2)
    assert np.all(trace.constraint_rank == 4)
    assert np.max(trace.position_constraint_norm_m) <= config.projection_tolerance_m
    assert np.max(trace.velocity_constraint_norm_m_s) <= config.velocity_tolerance_m_s
    assert np.max(trace.kkt_residual_norm) <= params.kkt_tolerance
    assert np.max(trace.acceleration_constraint_residual_norm) <= params.kkt_tolerance
    assert trace.maximum_projection_correction_m > 0.0
    assert trace.projection_energy_change_j.shape == trace.time.shape
    assert abs(np.sum(trace.projection_energy_change_j)) < 1e-3


def test_zero_command_rollout_has_convergent_work_energy_balance() -> None:
    params = TwoArmParams.publication_default()
    q0, v0 = _initial_state(params)
    coarse = rollout_forward_two_arm(
        q0,
        v0,
        constant_control(TwoArmControl.zero()),
        params,
        ForwardTwoArmConfig(duration_s=0.04, step_s=0.0005),
    )
    fine = rollout_forward_two_arm(
        q0,
        v0,
        constant_control(TwoArmControl.zero()),
        params,
        ForwardTwoArmConfig(duration_s=0.04, step_s=0.00025),
    )

    coarse_drift = abs(coarse.mechanical_energy_j[-1] - coarse.mechanical_energy_j[0])
    fine_drift = abs(fine.mechanical_energy_j[-1] - fine.mechanical_energy_j[0])
    assert fine_drift < coarse_drift
    assert fine_drift < 2e-3
    assert abs(np.sum(fine.projection_energy_change_j)) < abs(
        np.sum(coarse.projection_energy_change_j)
    )
    np.testing.assert_allclose(
        fine.mechanical_energy_j,
        [
            mechanical_energy(q, v, params)
            for q, v in zip(fine.q, fine.qdot, strict=True)
        ],
    )


def test_branched_zero_command_shares_cut_state_then_diverges() -> None:
    params = TwoArmParams.publication_default()
    q0, v0 = _initial_state(params)
    config = ForwardTwoArmConfig(duration_s=0.05, step_s=0.00025)
    baseline = rollout_forward_two_arm(
        q0,
        v0,
        constant_control(_control()),
        params,
        config,
    )
    cut_index = 80

    branch = branch_zero_command(
        baseline,
        cut_index=cut_index,
        horizon_s=0.02,
        params=params,
    )

    np.testing.assert_array_equal(branch.q[0], baseline.q[cut_index])
    np.testing.assert_array_equal(branch.qdot[0], baseline.qdot[cut_index])
    assert all(control == TwoArmControl.zero() for control in branch.controls)
    baseline_comparison = baseline.q[cut_index : cut_index + branch.time.size]
    assert np.linalg.norm(branch.q[-1] - baseline_comparison[-1]) > 1e-5
    assert branch.branch_source_index == cut_index
    assert branch.counterfactual_kind == "forward_branched_zero_command"


def test_zero_grip_moment_arm_removes_force_generated_club_couple() -> None:
    params = replace(
        TwoArmParams.publication_default(),
        right_grip_offset_m=0.0,
        left_grip_offset_m=0.0,
    )
    q0, v0 = _initial_state(params)
    trace = rollout_forward_two_arm(
        q0,
        v0,
        constant_control(_control()),
        params,
        ForwardTwoArmConfig(duration_s=0.01, step_s=0.00025),
    )
    for index, configuration in enumerate(trace.q):
        points = kinematics(configuration, params)
        velocity = constraint_jacobian(configuration, params) @ trace.qdot[index]
        assert np.linalg.norm(velocity) < 1e-8
        wrench = contact_wrench(
            trace.contact_force_on_club_n[index, 0],
            trace.contact_force_on_club_n[index, 1],
            points["right_grip"],
            points["left_grip"],
            points["club_center"],
            np.zeros(2),
            np.zeros(2),
        )
        assert wrench.moment_about_center_nm == pytest.approx(0.0, abs=1e-12)
        assert np.linalg.norm(constraint_vector(configuration, params)) < 1e-8


def test_invalid_projection_or_branch_requests_fail_closed() -> None:
    with pytest.raises(ValueError, match="step_s"):
        ForwardTwoArmConfig(duration_s=0.1, step_s=0.0)
    params = TwoArmParams.publication_default()
    q0, v0 = _initial_state(params)
    trace = rollout_forward_two_arm(
        q0,
        v0,
        constant_control(TwoArmControl.zero()),
        params,
        ForwardTwoArmConfig(duration_s=0.005, step_s=0.001),
    )
    with pytest.raises(IndexError, match="cut_index"):
        branch_zero_command(
            trace, cut_index=trace.time.size, horizon_s=0.01, params=params
        )
