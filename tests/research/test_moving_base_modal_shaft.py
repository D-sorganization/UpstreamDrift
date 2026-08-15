"""Contracts for higher-order shaft modes in the forward two-hand solve."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.moving_base_modal_shaft import (
    ModalShaftCouplingConfig,
    ModalShaftCouplingParams,
    initial_state,
    mass_matrix,
    modal_shaft_basis,
    rollout,
    solve_constrained_dynamics,
)
from scripts.research.proximal_distal_energy.two_arm_closed_loop import TwoArmControl

pytestmark = pytest.mark.scientific


def test_modal_contract_fails_closed() -> None:
    params = ModalShaftCouplingParams.publication_default(mode_count=3)
    with pytest.raises(ValueError, match="mode_count"):
        replace(params, mode_count=0)
    with pytest.raises(ValueError, match="quadrature_order"):
        replace(params, quadrature_order=1)


def test_quadrature_modes_reproduce_fe_reference_and_are_mass_normalized() -> None:
    params = ModalShaftCouplingParams.publication_default(mode_count=6)
    basis = modal_shaft_basis(params)

    np.testing.assert_allclose(basis.modal_mass, np.eye(6), atol=2e-10)
    assert basis.maximum_frequency_discrepancy_relative < 2e-3
    assert np.all(np.diff(basis.coupled_frequencies_hz) > 0.0)
    assert basis.calibration_status == "synthetic_reference_not_equipment_calibrated"


def test_initial_state_is_closed_and_kkt_system_is_well_posed() -> None:
    params = ModalShaftCouplingParams.publication_default(mode_count=3)
    q, qdot = initial_state(params)
    solved = solve_constrained_dynamics(q, qdot, TwoArmControl.zero(), params)

    assert solved.constraint_rank == 4
    assert solved.kkt_residual_norm < params.mechanism.kkt_tolerance
    assert solved.acceleration_constraint_residual_norm < params.mechanism.kkt_tolerance
    assert np.min(np.linalg.eigvalsh(mass_matrix(q, params))) > 0.0


def test_driven_rollout_couples_base_contacts_and_multiple_shaft_modes() -> None:
    params = ModalShaftCouplingParams.publication_default(mode_count=3)
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
        ModalShaftCouplingConfig(duration_s=0.02, step_s=0.0001),
    )

    assert np.max(np.linalg.norm(trace.q[:, 4:6], axis=1)) > 1e-8
    assert np.max(np.abs(trace.modal_coordinates[:, 0])) > 1e-9
    assert np.max(np.abs(trace.modal_coordinates[:, 1:])) > 1e-11
    assert np.max(trace.shaft_strain_energy_j) > 0.0
    assert np.max(trace.position_constraint_norm_m) < 2e-9
    assert np.max(trace.velocity_constraint_norm_m_s) < 2e-9
    assert np.max(np.abs(trace.contact_power_identity_residual_w)) < 1e-9


def test_unforced_lossless_rollout_closes_energy() -> None:
    base = ModalShaftCouplingParams.publication_default(mode_count=3)
    params = replace(
        base,
        mechanism=replace(
            base.mechanism,
            gravity_m_s2=0.0,
            base_damping_ns_m=0.0,
            joint_damping_nms_rad=0.0,
        ),
        damping_ratio=0.0,
    )
    q, qdot = initial_state(params)
    trace = rollout(
        q,
        qdot,
        lambda _t, _q, _v: TwoArmControl.zero(),
        params,
        ModalShaftCouplingConfig(duration_s=0.005, step_s=0.0001),
    )

    assert abs(trace.mechanical_energy_j[-1] - trace.mechanical_energy_j[0]) < 1e-7
    assert np.max(trace.kkt_residual_norm) < params.mechanism.kkt_tolerance


def test_coincident_grip_control_removes_force_generated_couple() -> None:
    base = ModalShaftCouplingParams.publication_default(mode_count=3)
    params = replace(
        base,
        mechanism=replace(
            base.mechanism,
            right_grip_offset_m=0.0,
            left_grip_offset_m=0.0,
        ),
    )
    q, qdot = initial_state(params)
    trace = rollout(
        q,
        qdot,
        lambda _t, _q, _v: TwoArmControl.zero(),
        params,
        ModalShaftCouplingConfig(duration_s=0.002, step_s=0.0001),
    )

    np.testing.assert_allclose(trace.force_generated_couple_nm, 0.0, atol=1e-12)
