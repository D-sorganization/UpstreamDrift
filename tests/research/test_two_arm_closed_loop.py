"""Mechanical contracts for the publication two-arm closed-loop model."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.two_arm_closed_loop import (
    TwoArmControl,
    TwoArmParams,
    contact_wrench,
    constraint_acceleration_bias_audit,
    control_generalized_force,
    coriolis_vector,
    decompose_contact_forces,
    drift_control_attribution,
    mass_matrix,
    solve_constrained_dynamics,
    two_arm_joint_transfer_trajectory,
)
from src.shared.python.biomechanics.drift_control_transfer import (
    compute_power_and_work,
)

pytestmark = pytest.mark.scientific


def _state() -> tuple[TwoArmParams, np.ndarray, np.ndarray]:
    params = TwoArmParams.publication_default()
    q = params.consistent_configuration(
        club_center=np.array([0.0, -0.52]),
        club_angle_rad=0.18,
        right_elbow_branch=1,
        left_elbow_branch=-1,
    )
    qdot = np.zeros(7)
    return params, q, qdot


def test_mass_matrix_is_symmetric_positive_definite() -> None:
    params, q, _ = _state()
    matrix = mass_matrix(q, params)

    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(matrix)) > 1e-8


def test_coriolis_vector_matches_exact_two_link_arm_terms() -> None:
    params, q, _ = _state()
    qdot = np.array([1.2, -0.7, -0.9, 0.4, 0.3, -0.2, 1.1])
    coupling = (
        0.5 * params.forearm_mass_kg * params.upper_length_m * params.forearm_length_m
    )
    expected = np.zeros(7)
    for shoulder_index, elbow_index in ((0, 1), (2, 3)):
        sine = np.sin(q[elbow_index])
        shoulder_speed = qdot[shoulder_index]
        elbow_speed = qdot[elbow_index]
        expected[shoulder_index] = (
            -coupling * sine * (2.0 * shoulder_speed * elbow_speed + elbow_speed**2)
        )
        expected[elbow_index] = coupling * sine * shoulder_speed**2

    np.testing.assert_allclose(coriolis_vector(q, qdot, params), expected, atol=1e-8)


def test_independent_hand_constraints_have_full_row_rank() -> None:
    params, q, qdot = _state()
    result = solve_constrained_dynamics(q, qdot, TwoArmControl.zero(), params)

    assert result.constraint_rank == 4
    assert result.kkt_residual_norm < 1e-9
    assert result.acceleration_constraint_residual_norm < 1e-9


def test_exact_constraint_acceleration_bias_matches_directional_audit() -> None:
    params, q, _ = _state()
    qdot = np.array([1.2, -0.7, -0.9, 0.4, 0.3, -0.2, 1.1])
    assert constraint_acceleration_bias_audit(q, qdot, params) < 1e-9


def test_singular_hand_geometry_fails_closed() -> None:
    params, q, qdot = _state()
    params = replace(
        params,
        right_shoulder_m=(0.0, 0.0),
        left_shoulder_m=(0.0, 0.0),
        right_grip_offset_m=0.0,
        left_grip_offset_m=0.0,
    )
    q = params.consistent_configuration(
        club_center=np.array([0.0, -0.62]),
        club_angle_rad=0.0,
        right_elbow_branch=1,
        left_elbow_branch=-1,
    )
    with pytest.raises(ValueError, match="constraint Jacobian"):
        solve_constrained_dynamics(q, qdot, TwoArmControl.zero(), params)


def test_same_state_total_equals_drift_plus_control() -> None:
    params, q, qdot = _state()
    control = TwoArmControl(
        right_shoulder_nm=38.0,
        right_elbow_nm=12.0,
        right_wrist_nm=-7.0,
        left_shoulder_nm=34.0,
        left_elbow_nm=9.0,
        left_wrist_nm=5.0,
    )

    split = drift_control_attribution(q, qdot, control, params)

    np.testing.assert_allclose(
        split.total.qddot,
        split.drift.qddot + split.control.qddot,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        split.total.contact_force_on_club_n,
        split.drift.contact_force_on_club_n + split.control.contact_force_on_club_n,
        atol=1e-10,
    )


def test_wrist_torques_act_on_forearms_and_club_with_opposite_signs() -> None:
    control = TwoArmControl(right_wrist_nm=8.0, left_wrist_nm=-3.0)
    generalized = control_generalized_force(control)

    np.testing.assert_allclose(generalized, [-8.0, -8.0, 3.0, 3.0, 0, 0, 5.0])
    assert np.sum(generalized[[0, 2, 6]]) == pytest.approx(0.0)


def test_common_and_differential_contact_modes_reconstruct_forces_and_power() -> None:
    right = np.array([18.0, -42.0])
    left = np.array([-10.0, 30.0])
    right_point = np.array([0.08, -0.5])
    left_point = np.array([-0.08, -0.5])
    club_center = np.array([0.0, -0.5])
    right_velocity = np.array([4.0, 1.0])
    left_velocity = np.array([3.5, 0.7])

    modes = decompose_contact_forces(right, left)
    np.testing.assert_allclose(modes.resultant_n, right + left)
    np.testing.assert_allclose(modes.differential_n, 0.5 * (right - left))
    np.testing.assert_allclose(
        modes.resultant_n / 2.0 + modes.differential_n,
        right,
    )
    np.testing.assert_allclose(
        modes.resultant_n / 2.0 - modes.differential_n,
        left,
    )

    wrench = contact_wrench(
        right,
        left,
        right_point,
        left_point,
        club_center,
        right_velocity,
        left_velocity,
    )
    expected_power = float(right @ right_velocity + left @ left_velocity)
    assert wrench.contact_power_w == pytest.approx(expected_power)
    assert wrench.resultant_force_n.shape == (2,)


def test_two_arm_adapter_reports_every_joint_and_contact_power() -> None:
    params, q, _ = _state()
    time = np.array([0.0, 0.05, 0.10])
    configuration = np.tile(q, (time.size, 1))
    velocity = np.zeros_like(configuration)
    controls = (
        TwoArmControl.zero(),
        TwoArmControl(
            right_shoulder_nm=30.0,
            right_elbow_nm=8.0,
            right_wrist_nm=-4.0,
            left_shoulder_nm=28.0,
            left_elbow_nm=7.0,
            left_wrist_nm=3.0,
        ),
        TwoArmControl(
            right_shoulder_nm=34.0,
            right_elbow_nm=6.0,
            right_wrist_nm=5.0,
            left_shoulder_nm=31.0,
            left_elbow_nm=5.0,
            left_wrist_nm=-2.0,
        ),
    )
    trajectory = two_arm_joint_transfer_trajectory(
        time, configuration, velocity, controls, params
    )

    assert trajectory.joint_names == (
        "right_shoulder",
        "right_elbow",
        "right_hand",
        "left_shoulder",
        "left_elbow",
        "left_hand",
    )
    np.testing.assert_allclose(
        trajectory.force_total,
        trajectory.force_drift + trajectory.force_control,
    )
    np.testing.assert_allclose(
        trajectory.couple_total,
        trajectory.couple_drift + trajectory.couple_control,
    )
    power = compute_power_and_work(trajectory)
    np.testing.assert_allclose(
        power.total_power_total,
        power.total_power_drift + power.total_power_control,
    )


def test_two_arm_adapter_fails_on_velocity_constraint_violation() -> None:
    params, q, _ = _state()
    time = np.array([0.0, 0.05])
    configuration = np.tile(q, (time.size, 1))
    velocity = np.zeros_like(configuration)
    velocity[:, 4] = 1.0

    with pytest.raises(ValueError, match="velocity constraints"):
        two_arm_joint_transfer_trajectory(
            time,
            configuration,
            velocity,
            (TwoArmControl.zero(), TwoArmControl.zero()),
            params,
        )
