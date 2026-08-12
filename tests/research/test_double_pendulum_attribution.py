"""Contracts for the exact double-pendulum joint-transfer adapter."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.double_pendulum_attribution import (
    double_pendulum_support_reaction_decomposition,
    double_pendulum_joint_transfer_trajectory,
)
from scripts.research.proximal_distal_energy.interaction_forces import (
    force_power_decomposition,
    reaction_force_decomposition,
)
from scripts.research.proximal_distal_energy.run_experiments import rollout_program
from scripts.research.proximal_distal_energy.swing_model import PlanarInertials
from scripts.research.proximal_distal_energy.torque_programs import (
    restrain_then_drive_program,
)
from src.shared.python.biomechanics.drift_control_transfer import (
    compute_path_frame,
    compute_power_and_work,
    project_forces_onto_path,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend

pytestmark = pytest.mark.scientific


@pytest.fixture(scope="module")
def reference_trace() -> tuple[
    GolfModelParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    params = GolfModelParams.default()
    program = restrain_then_drive_program(60.0, 15.0, 10.0, 0.10)
    t, q, v, u = rollout_program(params, program)
    return params, t[::20], q[::20], v[::20], u[::20]


def test_adapter_closes_force_couple_and_power_splits(
    reference_trace: tuple[
        GolfModelParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    params, t, q, v, u = reference_trace
    trajectory = double_pendulum_joint_transfer_trajectory(t, q, v, u, params)

    assert trajectory.joint_names == ("shoulder", "wrist")
    assert trajectory.model_tier == "exact_planar_double_pendulum"
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


def test_wrist_force_and_power_reproduce_existing_exact_mechanics(
    reference_trace: tuple[
        GolfModelParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    params, t, q, v, u = reference_trace
    inertials = PlanarInertials.from_params(params)
    backend = make_backend("ode", params)
    qdd = np.vstack(
        [
            backend.forward_dynamics(qk, vk, uk)
            for qk, vk, uk in zip(q, v, u, strict=True)
        ]
    )
    existing_force = reaction_force_decomposition(inertials, q, v, qdd)
    existing_power = force_power_decomposition(inertials, q, v, existing_force)

    trajectory = double_pendulum_joint_transfer_trajectory(t, q, v, u, params)
    np.testing.assert_allclose(trajectory.force_total[:, 1], existing_force.total)
    power = compute_power_and_work(trajectory)
    np.testing.assert_allclose(power.force_power_total[:, 1], existing_power.total)


def test_shoulder_force_satisfies_arm_subsystem_balance(
    reference_trace: tuple[
        GolfModelParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    params, t, q, v, u = reference_trace
    inertials = PlanarInertials.from_params(params)
    backend = make_backend("ode", params)
    qdd = np.vstack(
        [
            backend.forward_dynamics(qk, vk, uk)
            for qk, vk, uk in zip(q, v, u, strict=True)
        ]
    )
    theta = q[:, 0]
    omega = v[:, 0]
    alpha = qdd[:, 0]
    direction = np.column_stack((np.sin(theta), -np.cos(theta)))
    tangent = np.column_stack((np.cos(theta), np.sin(theta)))
    arm_com_acceleration = inertials.lc1 * (
        alpha[:, None] * tangent - omega[:, None] ** 2 * direction
    )
    gravity = np.array([0.0, -inertials.g_proj])

    trajectory = double_pendulum_joint_transfer_trajectory(t, q, v, u, params)
    net_external = (
        trajectory.force_total[:, 0]
        - trajectory.force_total[:, 1]
        + inertials.m1 * gravity
    )
    np.testing.assert_allclose(
        net_external,
        inertials.m1 * arm_com_acceleration,
        atol=1e-10,
    )


def test_fixed_shoulder_has_undefined_path_projection_but_wrist_is_valid(
    reference_trace: tuple[
        GolfModelParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    params, t, q, v, u = reference_trace
    trajectory = double_pendulum_joint_transfer_trajectory(t, q, v, u, params)
    frame = compute_path_frame(trajectory.velocity, speed_epsilon=1e-9)
    projection = project_forces_onto_path(trajectory, frame)

    assert not np.any(frame.valid[:, 0])
    assert not frame.valid[0, 1]
    assert np.all(frame.valid[1:, 1])
    assert np.all(np.isnan(projection.total_along[:, 0]))
    assert np.isnan(projection.total_along[0, 1])
    assert np.all(np.isfinite(projection.total_along[1:, 1]))


def test_support_reaction_decomposition_closes_pointwise(
    reference_trace: tuple[
        GolfModelParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    params, t, q, v, u = reference_trace
    result = double_pendulum_support_reaction_decomposition(t, q, v, u, params)

    np.testing.assert_allclose(
        result.total,
        result.configuration + result.velocity + result.control,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.ztcf,
        result.configuration + result.velocity,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.zvcf,
        result.configuration + result.control,
        atol=1e-10,
    )
    assert result.force_direction == "support_on_mechanism"
    assert result.model_scope == "fixed_base_support_reaction_proxy"


def test_support_counterfactuals_are_pointwise_not_forward_rollouts(
    reference_trace: tuple[
        GolfModelParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ],
) -> None:
    params, t, q, v, u = reference_trace
    result = double_pendulum_support_reaction_decomposition(t, q, v, u, params)

    assert result.total.shape == (t.size, 2)
    np.testing.assert_array_equal(result.time, t)
    assert np.linalg.norm(result.velocity) > 0.0
    assert np.linalg.norm(result.control) > 0.0
