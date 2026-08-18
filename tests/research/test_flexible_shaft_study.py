"""Scientific contracts for the reduced-order flexible-shaft study."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.flexible_shaft_study import (
    FlexibleShaftParams,
    acceleration_decomposition,
    mechanical_energy,
    rollout_flexible,
    shaft_energy,
    shaft_power_terms,
    trace_kinematics,
    velocity_bias_power_identity_residual,
)

pytestmark = pytest.mark.scientific


def test_acceleration_contributions_reconstruct_total() -> None:
    params = FlexibleShaftParams.reference()
    state = np.array([-1.7, 0.5, 0.08, 4.0, -2.0, 3.0])

    decomposition = acceleration_decomposition(state, 0.17, params)

    reconstructed = sum(
        decomposition[name]
        for name in (
            "control",
            "momentum",
            "gravity",
            "joint_damping",
            "shaft_elastic",
            "shaft_damping",
        )
    )
    np.testing.assert_allclose(reconstructed, decomposition["total"], atol=1e-11)


def test_zero_gravity_and_zero_damping_remove_declared_contributions() -> None:
    params = FlexibleShaftParams.reference().with_updates(
        gravity_enabled=False,
        joint_damping_enabled=False,
        shaft_damping_nms_rad=0.0,
    )
    state = np.array([-1.7, 0.5, 0.08, 4.0, -2.0, 3.0])

    decomposition = acceleration_decomposition(state, 0.17, params)

    np.testing.assert_allclose(decomposition["gravity"], 0.0, atol=1e-12)
    np.testing.assert_allclose(decomposition["joint_damping"], 0.0, atol=1e-12)
    np.testing.assert_allclose(decomposition["shaft_damping"], 0.0, atol=1e-12)


def test_shaft_energy_and_power_have_consistent_signs() -> None:
    stiffness = 80.0
    damping = 0.6
    flex = 0.12
    flex_rate = -3.0

    energy = shaft_energy(flex, stiffness)
    storage_rate, damping_power = shaft_power_terms(flex, flex_rate, stiffness, damping)

    assert energy == pytest.approx(0.5 * stiffness * flex**2)
    assert storage_rate == pytest.approx(stiffness * flex * flex_rate)
    assert damping_power == pytest.approx(-damping * flex_rate**2)
    assert damping_power <= 0.0


def test_unforced_undamped_rollout_closes_mechanical_energy() -> None:
    params = FlexibleShaftParams.reference().with_updates(
        gravity_enabled=True,
        joint_damping_enabled=False,
        shaft_damping_nms_rad=0.0,
        shoulder_torque_nm=0.0,
        wrist_drive_nm=0.0,
        wrist_restrain_nm=0.0,
    )
    initial = np.array([-1.2, 0.35, 0.04, 0.5, -0.2, 0.1])

    trace = rollout_flexible(params, initial_state=initial, horizon_s=0.08, dt_s=0.0002)
    energy = np.array([mechanical_energy(state, params) for state in trace.state])

    assert np.ptp(energy) < 2e-5


def test_reference_rollout_is_deterministic_and_finite() -> None:
    params = FlexibleShaftParams.reference()
    first = rollout_flexible(params, horizon_s=0.12, dt_s=0.0005)
    second = rollout_flexible(params, horizon_s=0.12, dt_s=0.0005)

    np.testing.assert_array_equal(first.state, second.state)
    assert np.all(np.isfinite(first.state))
    assert np.max(np.abs(first.state[:, 2])) < 1.0


def test_tip_velocity_is_exact_state_kinematics() -> None:
    params = FlexibleShaftParams.reference()
    trace = rollout_flexible(params, horizon_s=0.02, dt_s=0.0005)
    kinematics = trace_kinematics(trace, params)
    index = 17
    state = trace.state[index]
    step = 1e-7
    q_minus = state[:3] - step * state[3:]
    q_plus = state[:3] + step * state[3:]
    from src.shared.python.pendulum_simulator.physics_triple import forward_kinematics

    tip_minus = np.asarray(forward_kinematics(*q_minus, params.triple())["tip"])
    tip_plus = np.asarray(forward_kinematics(*q_plus, params.triple())["tip"])

    np.testing.assert_allclose(
        kinematics["tip_velocity"][index],
        (tip_plus - tip_minus) / (2.0 * step),
        atol=1e-8,
    )


def test_velocity_bias_power_identity_closes() -> None:
    params = FlexibleShaftParams.reference()
    state = np.array([-1.7, 0.5, 0.08, 4.0, -2.0, 3.0])

    assert abs(velocity_bias_power_identity_residual(state, params)) < 1e-7
