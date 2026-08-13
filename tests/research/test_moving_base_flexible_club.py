"""Contracts for coupled moving-base, two-hand, flexible-club dynamics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.moving_base_flexible_club import (
    MovingBaseFlexibleConfig,
    MovingBaseFlexibleParams,
    constraint_acceleration_bias_audit,
    control_generalized_force,
    initial_state,
    mass_matrix,
    mechanical_energy,
    rollout,
    solve_constrained_dynamics,
)
from scripts.research.proximal_distal_energy.two_arm_closed_loop import TwoArmControl


def test_parameter_contract_fails_closed() -> None:
    params = MovingBaseFlexibleParams.publication_default()
    with pytest.raises(ValueError, match="base_mass_kg"):
        replace(params, base_mass_kg=0.0)
    with pytest.raises(ValueError, match="shaft_stiffness_nm_rad"):
        replace(params, shaft_stiffness_nm_rad=-1.0)
    with pytest.raises(ValueError, match="grip offsets"):
        replace(params, right_grip_offset_m=np.nan)


def test_initial_state_is_closed_and_mass_is_positive_definite() -> None:
    params = MovingBaseFlexibleParams.publication_default()
    q, qdot = initial_state(params)
    solution = solve_constrained_dynamics(q, qdot, TwoArmControl.zero(), params)

    assert solution.constraint_rank == 4
    assert solution.kkt_residual_norm < params.kkt_tolerance
    assert solution.acceleration_constraint_residual_norm < params.kkt_tolerance
    assert np.min(np.linalg.eigvalsh(mass_matrix(q, params))) > 0.0


def test_exact_constraint_acceleration_bias_matches_directional_audit() -> None:
    params = MovingBaseFlexibleParams.publication_default()
    q, qdot = initial_state(params)
    qdot[[0, 1, 2, 3, 8]] = [1.1, -0.4, 0.8, 0.25, -0.7]
    # Projecting is unnecessary for the local Jdot identity; it holds for every
    # finite configuration and velocity in this autonomous geometry.
    assert constraint_acceleration_bias_audit(q, qdot, params) < 1e-9


def test_control_mapping_preserves_wrist_action_reaction() -> None:
    control = TwoArmControl(right_wrist_nm=3.0, left_wrist_nm=-1.0)
    generalized = control_generalized_force(control)

    assert generalized.shape == (10,)
    np.testing.assert_allclose(generalized[[0, 1]], -3.0)
    np.testing.assert_allclose(generalized[[2, 3]], 1.0)
    assert generalized[8] == pytest.approx(2.0)
    rigid_rotation = np.zeros(10)
    rigid_rotation[[0, 2, 8]] = 1.0
    assert generalized @ rigid_rotation == pytest.approx(0.0)


def test_unforced_lossless_rollout_closes_energy_and_constraints() -> None:
    params = replace(
        MovingBaseFlexibleParams.publication_default(),
        gravity_m_s2=0.0,
        base_damping_ns_m=0.0,
        shaft_damping_nms_rad=0.0,
        joint_damping_nms_rad=0.0,
    )
    q, qdot = initial_state(params)
    trace = rollout(
        q,
        qdot,
        lambda _t, _q, _v: TwoArmControl.zero(),
        params,
        MovingBaseFlexibleConfig(duration_s=0.01, step_s=0.0005),
    )

    assert np.max(trace.position_constraint_norm_m) < 2e-9
    assert np.max(trace.velocity_constraint_norm_m_s) < 2e-9
    assert np.max(trace.kkt_residual_norm) < params.kkt_tolerance
    assert np.max(np.abs(trace.constraint_two_sided_power_residual_w)) < 1e-9
    assert abs(trace.mechanical_energy_j[-1] - trace.mechanical_energy_j[0]) < 1e-7


def test_coupled_rollout_moves_base_and_flexes_club() -> None:
    params = MovingBaseFlexibleParams.publication_default()
    q, qdot = initial_state(params)
    control = TwoArmControl(
        right_shoulder_nm=18.0,
        right_elbow_nm=7.0,
        right_wrist_nm=-3.0,
        left_shoulder_nm=16.0,
        left_elbow_nm=6.0,
        left_wrist_nm=2.0,
    )
    trace = rollout(
        q,
        qdot,
        lambda _t, _q, _v: control,
        params,
        MovingBaseFlexibleConfig(duration_s=0.03, step_s=0.0005),
    )

    assert np.max(np.linalg.norm(trace.q[:, 4:6], axis=1)) > 1e-7
    assert np.max(np.abs(trace.q[:, 9])) > 1e-7
    assert np.all(np.isfinite(trace.force_generated_couple_nm))
    assert np.max(np.abs(trace.contact_power_identity_residual_w)) < 1e-9
    assert np.allclose(
        trace.mechanical_energy_j,
        [
            mechanical_energy(qi, vi, params)
            for qi, vi in zip(trace.q, trace.qdot, strict=True)
        ],
    )


def test_coincident_grip_negative_control_removes_force_couple() -> None:
    params = replace(
        MovingBaseFlexibleParams.publication_default(),
        right_grip_offset_m=0.0,
        left_grip_offset_m=0.0,
    )
    q, qdot = initial_state(params)
    trace = rollout(
        q,
        qdot,
        lambda _t, _q, _v: TwoArmControl.zero(),
        params,
        MovingBaseFlexibleConfig(duration_s=0.005, step_s=0.0005),
    )
    np.testing.assert_allclose(trace.force_generated_couple_nm, 0.0, atol=1e-12)
