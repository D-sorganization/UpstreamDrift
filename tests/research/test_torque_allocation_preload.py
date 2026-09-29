"""Contracts for matched club-torque allocation and transmission preload."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.torque_allocation_preload import (
    RoleReversalProgram,
    TransmissionChannel,
    allocate_matched_angular_acceleration,
    evaluate_continuous_role_reversal,
    evaluate_role_reversal,
    matched_allocation_sweep,
)
from scripts.research.proximal_distal_energy.two_arm_closed_loop import TwoArmParams

pytestmark = pytest.mark.scientific


def _state(params: TwoArmParams) -> tuple[np.ndarray, np.ndarray]:
    q = params.consistent_configuration(np.array([0.0, -0.50]), 0.16)
    return q, np.zeros_like(q)


@pytest.mark.parametrize("channel", ["proximal", "wrist"])
def test_channel_allocations_match_the_same_club_angular_task(channel: str) -> None:
    params = TwoArmParams.publication_default()
    q, qdot = _state(params)

    result = allocate_matched_angular_acceleration(
        q,
        qdot,
        target_control_angular_acceleration_rad_s2=120.0,
        channel=channel,
        params=params,
    )

    assert result.control_angular_acceleration_rad_s2 == pytest.approx(120.0)
    assert result.net_control_moment_nm == pytest.approx(
        params.club_inertia_kg_m2 * 120.0,
        rel=1e-10,
        abs=1e-10,
    )
    assert result.direct_wrist_moment_nm + result.grip_force_couple_nm == pytest.approx(
        result.net_control_moment_nm,
        rel=1e-10,
        abs=1e-10,
    )


def test_proximal_and_wrist_extremes_have_distinct_mechanical_signatures() -> None:
    params = TwoArmParams.publication_default()
    q, qdot = _state(params)
    proximal = allocate_matched_angular_acceleration(q, qdot, 120.0, "proximal", params)
    wrist = allocate_matched_angular_acceleration(q, qdot, 120.0, "wrist", params)

    assert proximal.direct_wrist_moment_nm == pytest.approx(0.0)
    assert proximal.grip_force_couple_nm == pytest.approx(
        proximal.net_control_moment_nm
    )
    assert abs(wrist.direct_wrist_moment_nm) > 0.0
    assert not np.isclose(
        proximal.hand_force_rms_n,
        wrist.hand_force_rms_n,
        rtol=1e-3,
        atol=1e-6,
    )


def test_allocation_sweep_is_task_matched_at_every_geometry_and_fraction() -> None:
    study = matched_allocation_sweep(
        club_angles_rad=np.linspace(-0.10, 0.30, 5),
        wrist_fractions=np.linspace(0.0, 1.0, 6),
        target_net_control_moment_nm=8.0,
    )

    assert study.club_angles_rad.shape == (5,)
    assert study.wrist_fractions.shape == (6,)
    assert study.hand_force_rms_n.shape == (5, 6)
    np.testing.assert_allclose(study.net_control_moment_nm, 8.0, atol=1e-9)
    np.testing.assert_allclose(
        study.direct_wrist_moment_nm + study.grip_force_couple_nm,
        study.net_control_moment_nm,
        atol=1e-9,
    )


def test_role_reversal_programs_match_pre_and_post_net_targets() -> None:
    proposed = RoleReversalProgram.persistent_direction()
    opposite = RoleReversalProgram.opposite_role_reversal()

    assert proposed.pre_net_torque_nm == pytest.approx(opposite.pre_net_torque_nm)
    assert proposed.post_net_torque_nm == pytest.approx(opposite.post_net_torque_nm)
    assert proposed.arm_pre_nm * proposed.arm_post_nm > 0.0
    assert proposed.wrist_pre_nm * proposed.wrist_post_nm > 0.0
    assert opposite.arm_pre_nm * opposite.arm_post_nm < 0.0
    assert opposite.wrist_pre_nm * opposite.wrist_post_nm < 0.0


def test_dead_zone_penalizes_sign_reversal_and_preload_reduces_delay() -> None:
    channel = TransmissionChannel(
        stiffness_nm_rad=600.0,
        dead_zone_rad=0.012,
        time_constant_s=0.018,
    )
    proposed = evaluate_role_reversal(
        RoleReversalProgram.persistent_direction(),
        arm_channel=channel,
        wrist_channel=channel,
        duration_s=0.12,
        step_s=0.0001,
        initialize_at_preload=True,
    )
    opposite = evaluate_role_reversal(
        RoleReversalProgram.opposite_role_reversal(),
        arm_channel=channel,
        wrist_channel=channel,
        duration_s=0.12,
        step_s=0.0001,
        initialize_at_preload=True,
    )
    relaxed = evaluate_role_reversal(
        RoleReversalProgram.opposite_role_reversal(),
        arm_channel=channel,
        wrist_channel=channel,
        duration_s=0.12,
        step_s=0.0001,
        initialize_at_preload=False,
    )

    assert proposed.arm_zero_transmission_duration_s == pytest.approx(0.0)
    assert proposed.wrist_zero_transmission_duration_s == pytest.approx(0.0)
    assert opposite.arm_zero_transmission_duration_s > 0.0
    assert opposite.wrist_zero_transmission_duration_s > 0.0
    assert opposite.arm_zero_transmission_duration_bounds_s == pytest.approx(
        (0.0114, 0.0116)
    )
    assert opposite.wrist_zero_transmission_duration_bounds_s == pytest.approx(
        (0.0219, 0.0221)
    )
    assert opposite.temporal_resolution_s == pytest.approx(0.0001)
    assert proposed.net_torque_error_impulse_nms < opposite.net_torque_error_impulse_nms
    assert opposite.net_torque_error_impulse_nms < relaxed.net_torque_error_impulse_nms


def test_continuous_preparation_carries_internal_state_through_transition() -> None:
    channel = TransmissionChannel(
        stiffness_nm_rad=600.0,
        dead_zone_rad=0.012,
        time_constant_s=0.018,
    )
    persistent = evaluate_continuous_role_reversal(
        RoleReversalProgram.persistent_direction(),
        arm_channel=channel,
        wrist_channel=channel,
        preparation_duration_s=0.18,
        post_transition_duration_s=0.12,
        step_s=0.0001,
    )
    reversal = evaluate_continuous_role_reversal(
        RoleReversalProgram.opposite_role_reversal(),
        arm_channel=channel,
        wrist_channel=channel,
        preparation_duration_s=0.18,
        post_transition_duration_s=0.12,
        step_s=0.0001,
    )

    index = persistent.transition_index
    assert persistent.time_s[index] == pytest.approx(0.0)
    assert persistent.preparation_duration_s == pytest.approx(0.18)
    assert persistent.transmitted_arm_torque_nm[index] == pytest.approx(
        persistent.transmitted_arm_torque_nm[index - 1], abs=0.002
    )
    assert persistent.transmitted_wrist_torque_nm[index] == pytest.approx(
        persistent.transmitted_wrist_torque_nm[index - 1], abs=0.002
    )
    assert persistent.transmitted_arm_torque_nm[index] > 0.0
    assert persistent.transmitted_wrist_torque_nm[index] < 0.0
    assert reversal.transmitted_arm_torque_nm[index] < 0.0
    assert reversal.transmitted_wrist_torque_nm[index] > 0.0
    assert persistent.arm_zero_transmission_duration_s == pytest.approx(0.0)
    assert persistent.wrist_zero_transmission_duration_s == pytest.approx(0.0)
    assert reversal.arm_zero_transmission_duration_s > 0.0
    assert reversal.wrist_zero_transmission_duration_s > 0.0
    assert (
        persistent.net_torque_error_impulse_nms < reversal.net_torque_error_impulse_nms
    )


def test_invalid_transmission_and_unreachable_allocation_fail_closed() -> None:
    with pytest.raises(ValueError, match="stiffness_nm_rad"):
        TransmissionChannel(
            stiffness_nm_rad=0.0,
            dead_zone_rad=0.01,
            time_constant_s=0.02,
        )
    params = TwoArmParams.publication_default()
    q, qdot = _state(params)
    with pytest.raises(ValueError, match="channel"):
        allocate_matched_angular_acceleration(q, qdot, 1.0, "unknown", params)
