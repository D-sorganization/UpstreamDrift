"""One-arm three-link adapter for the joint-transfer model ladder.

This tier uses the repository's existing triple-pendulum equations: upper arm,
forearm, and club are represented by distal point masses.  It adds an elbow
and separates shoulder, elbow, and wrist reactions, but it is not presented as
a distributed-inertia anatomical arm.  Total and drift are evaluated at the
same state; control is their pointwise difference.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.biomechanics.drift_control_transfer import (
    JointTransferTrajectory,
)
from src.shared.python.pendulum_simulator.physics_triple import (
    TriplePendulumParams,
    equations_of_motion,
    forward_kinematics,
    friction_torque_vector,
    net_joint_forces,
)


def _validated_arrays(
    time: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_array = np.asarray(time, dtype=float).reshape(-1)
    expected = (time_array.size, 3)
    arrays = [np.asarray(value, dtype=float) for value in (q, v, controls)]
    for name, value in zip(("q", "v", "controls"), arrays, strict=True):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
    if time_array.size < 2 or np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must contain at least two strictly increasing samples")
    if not all(np.all(np.isfinite(value)) for value in (time_array, *arrays)):
        raise ValueError("trajectory inputs must contain only finite values")
    return time_array, arrays[0], arrays[1], arrays[2]


def _accelerations_and_forces(
    time: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    controls: np.ndarray,
    params: TriplePendulumParams,
) -> tuple[np.ndarray, np.ndarray]:
    accelerations = np.empty_like(q)
    forces = np.empty((time.size, 3, 2))
    source_names = ("shoulder", "wrist1", "wrist2")
    for index, (sample_time, q_sample, v_sample, control_sample) in enumerate(
        zip(time, q, v, controls, strict=True)
    ):
        state = np.concatenate((q_sample, v_sample))
        torque = tuple(float(value) for value in control_sample)
        qddot = equations_of_motion(
            state,
            float(sample_time),
            params,
            lambda _time, torque=torque: torque,
        )[3:]
        accelerations[index] = qddot
        force_map = net_joint_forces(state, qddot, params)
        for joint_index, source_name in enumerate(source_names):
            forces[index, joint_index] = force_map[source_name]
    return accelerations, forces


def _positions_and_velocities(
    q: np.ndarray,
    v: np.ndarray,
    params: TriplePendulumParams,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = q.shape[0]
    positions = np.empty((sample_count, 3, 2))
    velocities = np.zeros((sample_count, 3, 2))
    for index, sample in enumerate(q):
        points = forward_kinematics(sample[0], sample[1], sample[2], params)
        positions[index, 0] = points["shoulder"]
        positions[index, 1] = points["wrist1"]
        positions[index, 2] = points["wrist2"]

    theta1 = q[:, 0]
    theta2 = q[:, 0] + q[:, 1]
    omega1 = v[:, 0]
    omega2 = v[:, 0] + v[:, 1]
    tangent1 = np.column_stack((np.cos(theta1), np.sin(theta1)))
    tangent2 = np.column_stack((np.cos(theta2), np.sin(theta2)))
    velocities[:, 1] = params.L1 * omega1[:, None] * tangent1
    velocities[:, 2] = velocities[:, 1] + (params.L2 * omega2[:, None] * tangent2)
    return positions, velocities


def one_arm_joint_transfer_trajectory(
    time: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    controls: np.ndarray,
    params: TriplePendulumParams,
) -> JointTransferTrajectory:
    """Return shoulder/elbow/wrist transfer quantities at identical states."""
    time_array, q_array, v_array, control_array = _validated_arrays(
        time, q, v, controls
    )
    _, force_total = _accelerations_and_forces(
        time_array, q_array, v_array, control_array, params
    )
    _, force_drift = _accelerations_and_forces(
        time_array, q_array, v_array, np.zeros_like(control_array), params
    )
    force_control = force_total - force_drift
    position, velocity = _positions_and_velocities(q_array, v_array, params)

    couple_drift = np.vstack(
        [
            friction_torque_vector(sample[0], sample[1], sample[2], params)
            for sample in v_array
        ]
    )
    couple_control = control_array.copy()
    couple_total = couple_drift + couple_control
    angular_velocity = np.column_stack(
        (
            v_array[:, 0],
            v_array[:, 0] + v_array[:, 1],
            v_array[:, 0] + v_array[:, 1] + v_array[:, 2],
        )
    )
    return JointTransferTrajectory(
        time=time_array,
        joint_names=("shoulder", "elbow", "wrist"),
        position=position,
        velocity=velocity,
        force_total=force_total,
        force_drift=force_drift,
        force_control=force_control,
        couple_total=couple_total,
        couple_drift=couple_drift,
        couple_control=couple_control,
        angular_velocity=angular_velocity,
        model_tier="one_arm_three_link_point_mass",
        force_direction="proximal_on_distal",
        frame="planar_cartesian_x_target_y_up",
        reference_point="shoulder_elbow_and_wrist_joint_centers",
        units="SI",
    )


__all__ = ["one_arm_joint_transfer_trajectory"]
