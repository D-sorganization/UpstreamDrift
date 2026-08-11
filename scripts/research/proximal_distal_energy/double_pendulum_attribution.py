"""Exact double-pendulum adapter for joint-transfer attribution.

The adapter evaluates total and zero-applied-control dynamics at each achieved
state.  Their difference is the pointwise control contribution; it is not a
subtraction between forward trajectories and is not a zero-velocity
counterfactual.  Joint forces use the proximal-on-distal convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.research.proximal_distal_energy.interaction_forces import (
    reaction_force_decomposition,
)
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    hand_velocity,
)
from src.shared.python.biomechanics.drift_control_transfer import (
    JointTransferTrajectory,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend


def _trace_arrays(
    time: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_array = np.asarray(time, dtype=float).reshape(-1)
    q_array = np.asarray(q, dtype=float)
    v_array = np.asarray(v, dtype=float)
    control_array = np.asarray(controls, dtype=float)
    expected = (time_array.size, 2)
    for name, value in (
        ("q", q_array),
        ("v", v_array),
        ("controls", control_array),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {value.shape}")
    if time_array.size < 2 or np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must contain at least two strictly increasing samples")
    if not all(
        np.all(np.isfinite(value))
        for value in (time_array, q_array, v_array, control_array)
    ):
        raise ValueError("trajectory inputs must contain only finite values")
    return time_array, q_array, v_array, control_array


def _direction(angle: np.ndarray) -> np.ndarray:
    return np.column_stack((np.sin(angle), -np.cos(angle)))


def _tangent(angle: np.ndarray) -> np.ndarray:
    return np.column_stack((np.cos(angle), np.sin(angle)))


def _arm_com_acceleration(
    inertials: PlanarInertials,
    q: np.ndarray,
    v: np.ndarray,
    qdd: np.ndarray,
) -> np.ndarray:
    theta = q[:, 0]
    omega = v[:, 0]
    alpha = qdd[:, 0]
    return inertials.lc1 * (
        alpha[:, None] * _tangent(theta) - omega[:, None] ** 2 * _direction(theta)
    )


def _joint_forces(
    inertials: PlanarInertials,
    q: np.ndarray,
    v: np.ndarray,
    qdd: np.ndarray,
) -> np.ndarray:
    wrist = reaction_force_decomposition(inertials, q, v, qdd).total
    arm_acceleration = _arm_com_acceleration(inertials, q, v, qdd)
    gravity = np.array([0.0, -inertials.g_proj])
    shoulder = inertials.m1 * arm_acceleration - inertials.m1 * gravity[None, :] + wrist
    return np.stack((shoulder, wrist), axis=1)


@dataclass(frozen=True)
class SupportReactionDecomposition:
    """Pointwise support-reaction attribution for the fixed-base mechanism.

    The planar support force is a model demonstration, not a bilateral human
    force-plate prediction.  ZTCF and ZVCF are evaluated at the achieved
    configuration; neither is a forward counterfactual trajectory.
    """

    time: np.ndarray
    total: np.ndarray
    configuration: np.ndarray
    velocity: np.ndarray
    control: np.ndarray
    ztcf: np.ndarray
    zvcf: np.ndarray
    force_direction: str = "support_on_mechanism"
    frame: str = "inclined_swing_plane_cartesian_x_target_y_up"
    units: str = "SI"
    model_scope: str = "fixed_base_support_reaction_proxy"

    def __post_init__(self) -> None:
        time = np.asarray(self.time, dtype=float).reshape(-1)
        arrays = {
            name: np.asarray(getattr(self, name), dtype=float)
            for name in (
                "total",
                "configuration",
                "velocity",
                "control",
                "ztcf",
                "zvcf",
            )
        }
        if time.size < 2 or np.any(np.diff(time) <= 0.0):
            raise ValueError("time must contain at least two increasing samples")
        for name, value in arrays.items():
            if value.shape != (time.size, 2) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {(time.size, 2)}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "time", time)


def double_pendulum_support_reaction_decomposition(
    time: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    controls: np.ndarray,
    params: GolfModelParams,
) -> SupportReactionDecomposition:
    """Decompose the fixed shoulder's planar support reaction pointwise.

    The velocity component is ``ZTCF - configuration`` and the control
    component is ``total - ZTCF``.  The independently evaluated ZVCF provides
    a falsification check: it must equal ``configuration + control`` for this
    control-affine rigid mechanism.
    """
    time_array, q_array, v_array, control_array = _trace_arrays(time, q, v, controls)
    inertials = PlanarInertials.from_params(params)
    backend = make_backend("ode", params)
    zero_velocity = np.zeros_like(v_array)
    zero_control = np.zeros_like(control_array)

    def accelerations(velocity: np.ndarray, control: np.ndarray) -> np.ndarray:
        return np.vstack(
            [
                backend.forward_dynamics(q_sample, v_sample, u_sample)
                for q_sample, v_sample, u_sample in zip(
                    q_array, velocity, control, strict=True
                )
            ]
        )

    total_qdd = accelerations(v_array, control_array)
    ztcf_qdd = accelerations(v_array, zero_control)
    zvcf_qdd = accelerations(zero_velocity, control_array)
    configuration_qdd = accelerations(zero_velocity, zero_control)
    total = _joint_forces(inertials, q_array, v_array, total_qdd)[:, 0]
    ztcf = _joint_forces(inertials, q_array, v_array, ztcf_qdd)[:, 0]
    zvcf = _joint_forces(inertials, q_array, zero_velocity, zvcf_qdd)[:, 0]
    configuration = _joint_forces(inertials, q_array, zero_velocity, configuration_qdd)[
        :, 0
    ]
    velocity = ztcf - configuration
    control = total - ztcf
    if not np.allclose(zvcf, configuration + control, atol=1e-10, rtol=1e-10):
        raise RuntimeError("support-reaction ZVCF closure failed")
    return SupportReactionDecomposition(
        time=time_array,
        total=total,
        configuration=configuration,
        velocity=velocity,
        control=control,
        ztcf=ztcf,
        zvcf=zvcf,
    )


def double_pendulum_joint_transfer_trajectory(
    time: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    controls: np.ndarray,
    params: GolfModelParams,
) -> JointTransferTrajectory:
    """Adapt an achieved double-pendulum trace to the canonical schema.

    The shoulder is fixed, so its linear hand-path projection is intentionally
    undefined.  Its force remains useful for subsystem balance and vector
    impulse.  Couple power uses the angular velocity of the distal body:
    absolute arm speed at the shoulder and absolute club speed at the wrist.
    """
    time_array, q_array, v_array, control_array = _trace_arrays(time, q, v, controls)
    inertials = PlanarInertials.from_params(params)
    backend = make_backend("ode", params)
    total_qdd = np.vstack(
        [
            backend.forward_dynamics(q_sample, v_sample, u_sample)
            for q_sample, v_sample, u_sample in zip(
                q_array, v_array, control_array, strict=True
            )
        ]
    )
    drift_qdd = np.vstack(
        [
            backend.forward_dynamics(q_sample, v_sample, np.zeros(2))
            for q_sample, v_sample in zip(q_array, v_array, strict=True)
        ]
    )
    force_total = _joint_forces(inertials, q_array, v_array, total_qdd)
    force_drift = _joint_forces(inertials, q_array, v_array, drift_qdd)
    force_control = force_total - force_drift

    shoulder_position = np.zeros((time_array.size, 2))
    wrist_position = inertials.l1 * _direction(q_array[:, 0])
    position = np.stack((shoulder_position, wrist_position), axis=1)
    shoulder_velocity = np.zeros((time_array.size, 2))
    wrist_velocity = hand_velocity(inertials, q_array, v_array)
    velocity = np.stack((shoulder_velocity, wrist_velocity), axis=1)

    damping_couple = np.column_stack(
        (
            -inertials.damping_shoulder * v_array[:, 0],
            -inertials.damping_wrist * v_array[:, 1],
        )
    )
    couple_drift = damping_couple
    couple_control = control_array.copy()
    couple_total = couple_drift + couple_control
    angular_velocity = np.column_stack((v_array[:, 0], v_array[:, 0] + v_array[:, 1]))

    return JointTransferTrajectory(
        time=time_array,
        joint_names=("shoulder", "wrist"),
        position=position,
        velocity=velocity,
        force_total=force_total,
        force_drift=force_drift,
        force_control=force_control,
        couple_total=couple_total,
        couple_drift=couple_drift,
        couple_control=couple_control,
        angular_velocity=angular_velocity,
        model_tier="exact_planar_double_pendulum",
        force_direction="proximal_on_distal",
        frame="inclined_swing_plane_cartesian_x_target_y_up",
        reference_point="shoulder_and_wrist_joint_centers",
        units="SI",
    )


__all__ = [
    "SupportReactionDecomposition",
    "double_pendulum_joint_transfer_trajectory",
    "double_pendulum_support_reaction_decomposition",
]
