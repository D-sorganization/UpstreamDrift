"""Coupled planar dynamics with a moving base and compliant club.

The model extends the forward two-hand closed loop without prescribing base
motion or shaft deflection.  Two two-link arms translate with a finite-mass
base, both hands constrain a floating proximal club segment, and a distal club
segment is connected by a linear torsional spring and damper.  Contact forces,
base motion, and flex are therefore simultaneous outputs of one KKT solve.

Coordinates are ``[r_s, r_e, l_s, l_e, x_b, y_b, x_g, y_g, alpha, beta]``.
The arm angles use shoulder plus relative elbow coordinates; ``alpha`` is the
proximal-club angle and ``beta`` is distal angle relative to ``alpha``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from scripts.research.proximal_distal_energy.two_arm_closed_loop import (
    TwoArmControl,
    TwoArmParams,
)

FloatArray = npt.NDArray[np.float64]
ControlLaw = Callable[[float, FloatArray, FloatArray], TwoArmControl]
N_COORDINATES = 10
N_CONSTRAINTS = 4


def _direction(angle_rad: float) -> FloatArray:
    return np.array([np.sin(angle_rad), -np.cos(angle_rad)])


def _derivative(angle_rad: float) -> FloatArray:
    return np.array([np.cos(angle_rad), np.sin(angle_rad)])


def _cross_z(offset: FloatArray, force: FloatArray) -> float:
    return float(offset[0] * force[1] - offset[1] * force[0])


def _state(name: str, value: object) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (N_COORDINATES,) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must have shape ({N_COORDINATES},) with finite values"
        )
    return array.copy()


@dataclass(frozen=True, slots=True)
class MovingBaseFlexibleParams:
    """Declared inertial, geometric, elastic, and numerical parameters."""

    right_shoulder_offset_m: tuple[float, float]
    left_shoulder_offset_m: tuple[float, float]
    upper_length_m: float
    forearm_length_m: float
    upper_mass_kg: float
    forearm_mass_kg: float
    upper_inertia_kg_m2: float
    forearm_inertia_kg_m2: float
    base_mass_kg: float
    base_stiffness_n_m: float
    base_damping_ns_m: float
    proximal_club_length_m: float
    distal_club_length_m: float
    proximal_club_mass_kg: float
    distal_club_mass_kg: float
    proximal_club_inertia_kg_m2: float
    distal_club_inertia_kg_m2: float
    shaft_stiffness_nm_rad: float
    shaft_damping_nms_rad: float
    joint_damping_nms_rad: float
    right_grip_offset_m: float
    left_grip_offset_m: float
    gravity_m_s2: float = 9.80665
    rank_tolerance: float = 1e-10
    constraint_tolerance_m: float = 1e-8
    kkt_tolerance: float = 1e-8

    @classmethod
    def publication_default(cls) -> MovingBaseFlexibleParams:
        """Return the deterministic reference parameter set."""
        upper_length = 0.32
        forearm_length = 0.30
        upper_mass = 1.9
        forearm_mass = 1.35
        proximal_length = 0.46
        distal_length = 0.54
        proximal_mass = 0.18
        distal_mass = 0.27
        return cls(
            right_shoulder_offset_m=(0.19, 0.0),
            left_shoulder_offset_m=(-0.19, 0.0),
            upper_length_m=upper_length,
            forearm_length_m=forearm_length,
            upper_mass_kg=upper_mass,
            forearm_mass_kg=forearm_mass,
            upper_inertia_kg_m2=upper_mass * upper_length**2 / 12.0,
            forearm_inertia_kg_m2=forearm_mass * forearm_length**2 / 12.0,
            base_mass_kg=35.0,
            base_stiffness_n_m=24000.0,
            base_damping_ns_m=500.0,
            proximal_club_length_m=proximal_length,
            distal_club_length_m=distal_length,
            proximal_club_mass_kg=proximal_mass,
            distal_club_mass_kg=distal_mass,
            proximal_club_inertia_kg_m2=proximal_mass * proximal_length**2 / 12.0,
            distal_club_inertia_kg_m2=distal_mass * distal_length**2 / 12.0,
            shaft_stiffness_nm_rad=80.0,
            shaft_damping_nms_rad=0.6,
            joint_damping_nms_rad=0.08,
            right_grip_offset_m=0.065,
            left_grip_offset_m=-0.065,
        )

    def __post_init__(self) -> None:
        positive = (
            "upper_length_m",
            "forearm_length_m",
            "upper_mass_kg",
            "forearm_mass_kg",
            "upper_inertia_kg_m2",
            "forearm_inertia_kg_m2",
            "base_mass_kg",
            "base_stiffness_n_m",
            "proximal_club_length_m",
            "distal_club_length_m",
            "proximal_club_mass_kg",
            "distal_club_mass_kg",
            "proximal_club_inertia_kg_m2",
            "distal_club_inertia_kg_m2",
            "shaft_stiffness_nm_rad",
            "rank_tolerance",
            "constraint_tolerance_m",
            "kkt_tolerance",
        )
        nonnegative = (
            "base_damping_ns_m",
            "shaft_damping_nms_rad",
            "joint_damping_nms_rad",
            "gravity_m_s2",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        offsets = np.asarray(
            [self.right_grip_offset_m, self.left_grip_offset_m], dtype=float
        )
        if not np.all(np.isfinite(offsets)):
            raise ValueError("grip offsets must be finite")
        for name in ("right_shoulder_offset_m", "left_shoulder_offset_m"):
            point = np.asarray(getattr(self, name), dtype=float)
            if point.shape != (2,) or not np.all(np.isfinite(point)):
                raise ValueError(f"{name} must contain two finite values")


@dataclass(frozen=True, slots=True)
class MovingBaseFlexibleConfig:
    """Forward integration and projection tolerances."""

    duration_s: float
    step_s: float
    start_time_s: float = 0.0
    projection_tolerance_m: float = 1e-10
    velocity_tolerance_m_s: float = 1e-9
    maximum_projection_iterations: int = 12

    def __post_init__(self) -> None:
        for name in (
            "duration_s",
            "step_s",
            "projection_tolerance_m",
            "velocity_tolerance_m_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.start_time_s):
            raise ValueError("start_time_s must be finite")
        if self.maximum_projection_iterations < 1:
            raise ValueError("maximum_projection_iterations must be at least one")
        count = self.duration_s / self.step_s
        if not np.isclose(count, round(count), atol=1e-10, rtol=0.0):
            raise ValueError("duration_s must be an integer multiple of step_s")

    @property
    def interval_count(self) -> int:
        return int(round(self.duration_s / self.step_s))


@dataclass(frozen=True, slots=True)
class DynamicsSolution:
    """One full-rank constrained dynamics result."""

    qddot: FloatArray
    multipliers_n: FloatArray
    contact_force_on_club_n: FloatArray
    constraint_rank: int
    kkt_residual_norm: float
    acceleration_constraint_residual_norm: float


@dataclass(frozen=True, slots=True)
class MovingBaseFlexibleTrace:
    """Recorded trajectory, interaction, energy, and closure evidence."""

    time: FloatArray
    q: FloatArray
    qdot: FloatArray
    qddot: FloatArray
    controls: tuple[TwoArmControl, ...]
    multipliers_n: FloatArray
    contact_force_on_club_n: FloatArray
    force_generated_couple_nm: FloatArray
    direct_wrist_torque_nm: FloatArray
    contact_power_w: FloatArray
    contact_wrench_power_w: FloatArray
    contact_power_identity_residual_w: FloatArray
    shaft_elastic_moment_nm: FloatArray
    shaft_damping_moment_nm: FloatArray
    shaft_strain_energy_j: FloatArray
    clubhead_position_m: FloatArray
    clubhead_velocity_m_s: FloatArray
    mechanical_energy_j: FloatArray
    applied_control_power_w: FloatArray
    dissipation_power_w: FloatArray
    position_constraint_norm_m: FloatArray
    velocity_constraint_norm_m_s: FloatArray
    kkt_residual_norm: FloatArray
    acceleration_constraint_residual_norm: FloatArray
    projection_correction_norm_m: FloatArray
    projection_energy_change_j: FloatArray
    model_tier: str = "moving_base_two_hand_flexible_club"


def initial_state(
    params: MovingBaseFlexibleParams,
    *,
    grip_center_m: npt.ArrayLike = (0.0, -0.5),
    club_angle_rad: float = 0.16,
) -> tuple[FloatArray, FloatArray]:
    """Return an exactly closed state at zero base displacement and flex."""
    old = TwoArmParams(
        right_shoulder_m=params.right_shoulder_offset_m,
        left_shoulder_m=params.left_shoulder_offset_m,
        upper_length_m=params.upper_length_m,
        forearm_length_m=params.forearm_length_m,
        upper_mass_kg=params.upper_mass_kg,
        forearm_mass_kg=params.forearm_mass_kg,
        upper_inertia_kg_m2=params.upper_inertia_kg_m2,
        forearm_inertia_kg_m2=params.forearm_inertia_kg_m2,
        club_mass_kg=params.proximal_club_mass_kg + params.distal_club_mass_kg,
        club_inertia_kg_m2=(
            params.proximal_club_inertia_kg_m2 + params.distal_club_inertia_kg_m2
        ),
        right_grip_offset_m=params.right_grip_offset_m,
        left_grip_offset_m=params.left_grip_offset_m,
        gravity_m_s2=params.gravity_m_s2,
    )
    old_q = old.consistent_configuration(
        np.asarray(grip_center_m, dtype=float), club_angle_rad
    )
    q = np.zeros(N_COORDINATES)
    q[:4] = old_q[:4]
    q[6:8] = old_q[4:6]
    q[8] = old_q[6]
    return q, np.zeros(N_COORDINATES)


def _body_jacobians(
    q: FloatArray, params: MovingBaseFlexibleParams
) -> tuple[list[tuple[float, FloatArray, FloatArray]], dict[str, FloatArray]]:
    """Return ``(mass, COM Jacobian, convective acceleration)`` bodies."""
    base_jacobian = np.zeros((2, N_COORDINATES))
    base_jacobian[:, 4:6] = np.eye(2)
    bodies: list[tuple[float, FloatArray, FloatArray]] = [
        (params.base_mass_kg, base_jacobian, np.zeros(2))
    ]
    points: dict[str, FloatArray] = {"base": q[4:6].copy()}
    base = q[4:6]
    for side, shoulder_index, elbow_index, offset in (
        ("right", 0, 1, params.right_shoulder_offset_m),
        ("left", 2, 3, params.left_shoulder_offset_m),
    ):
        shoulder_angle = q[shoulder_index]
        forearm_angle = shoulder_angle + q[elbow_index]
        shoulder = base + np.asarray(offset)
        elbow = shoulder + params.upper_length_m * _direction(shoulder_angle)
        hand = elbow + params.forearm_length_m * _direction(forearm_angle)
        points[f"{side}_shoulder"] = shoulder
        points[f"{side}_elbow"] = elbow
        points[f"{side}_hand"] = hand

        upper = np.zeros((2, N_COORDINATES))
        upper[:, 4:6] = np.eye(2)
        upper[:, shoulder_index] = (
            0.5 * params.upper_length_m * _derivative(shoulder_angle)
        )
        upper_bias = -(0.5 * params.upper_length_m * _direction(shoulder_angle))
        bodies.append((params.upper_mass_kg, upper, upper_bias))

        fore = np.zeros((2, N_COORDINATES))
        fore[:, 4:6] = np.eye(2)
        fore[:, shoulder_index] = params.upper_length_m * _derivative(
            shoulder_angle
        ) + 0.5 * params.forearm_length_m * _derivative(forearm_angle)
        fore[:, elbow_index] = (
            0.5 * params.forearm_length_m * _derivative(forearm_angle)
        )
        # Coefficients are multiplied by squared angular rates in velocity_bias.
        fore_bias = np.column_stack(
            (
                -params.upper_length_m * _direction(shoulder_angle),
                -0.5 * params.forearm_length_m * _direction(forearm_angle),
            )
        )
        bodies.append((params.forearm_mass_kg, fore, fore_bias))

    alpha = q[8]
    beta = q[9]
    distal_angle = alpha + beta
    center = q[6:8]
    flex_joint = center + params.proximal_club_length_m * _direction(alpha)
    clubhead = flex_joint + params.distal_club_length_m * _direction(distal_angle)
    points["grip_center"] = center.copy()
    points["flex_joint"] = flex_joint
    points["clubhead"] = clubhead
    points["right_grip"] = center + params.right_grip_offset_m * _direction(alpha)
    points["left_grip"] = center + params.left_grip_offset_m * _direction(alpha)

    proximal = np.zeros((2, N_COORDINATES))
    proximal[:, 6:8] = np.eye(2)
    proximal[:, 8] = 0.5 * params.proximal_club_length_m * _derivative(alpha)
    bodies.append(
        (
            params.proximal_club_mass_kg,
            proximal,
            -0.5 * params.proximal_club_length_m * _direction(alpha),
        )
    )
    distal = np.zeros((2, N_COORDINATES))
    distal[:, 6:8] = np.eye(2)
    distal[:, 8] = params.proximal_club_length_m * _derivative(
        alpha
    ) + 0.5 * params.distal_club_length_m * _derivative(distal_angle)
    distal[:, 9] = 0.5 * params.distal_club_length_m * _derivative(distal_angle)
    distal_bias = np.column_stack(
        (
            -params.proximal_club_length_m * _direction(alpha),
            -0.5 * params.distal_club_length_m * _direction(distal_angle),
        )
    )
    bodies.append((params.distal_club_mass_kg, distal, distal_bias))
    return bodies, points


def kinematics(q: object, params: MovingBaseFlexibleParams) -> dict[str, FloatArray]:
    """Return declared joint, grip, flex-joint, and clubhead positions."""
    _, points = _body_jacobians(_state("q", q), params)
    return points


def mass_matrix(q: object, params: MovingBaseFlexibleParams) -> FloatArray:
    """Return the symmetric mass matrix assembled from COM Jacobians."""
    state = _state("q", q)
    bodies, _ = _body_jacobians(state, params)
    matrix = np.zeros((N_COORDINATES, N_COORDINATES))
    for mass, jacobian, _bias in bodies:
        matrix += mass * jacobian.T @ jacobian
    for shoulder_index, elbow_index in ((0, 1), (2, 3)):
        upper_angular = np.zeros(N_COORDINATES)
        upper_angular[shoulder_index] = 1.0
        forearm_angular = upper_angular.copy()
        forearm_angular[elbow_index] = 1.0
        matrix += params.upper_inertia_kg_m2 * np.outer(upper_angular, upper_angular)
        matrix += params.forearm_inertia_kg_m2 * np.outer(
            forearm_angular, forearm_angular
        )
    proximal_angular = np.zeros(N_COORDINATES)
    proximal_angular[8] = 1.0
    distal_angular = proximal_angular.copy()
    distal_angular[9] = 1.0
    matrix += params.proximal_club_inertia_kg_m2 * np.outer(
        proximal_angular, proximal_angular
    )
    matrix += params.distal_club_inertia_kg_m2 * np.outer(
        distal_angular, distal_angular
    )
    return matrix


def _velocity_bias(
    q: FloatArray, qdot: FloatArray, params: MovingBaseFlexibleParams
) -> FloatArray:
    bodies, _ = _body_jacobians(q, params)
    result = np.zeros(N_COORDINATES)
    body_index = 1
    for shoulder_index, elbow_index in ((0, 1), (2, 3)):
        mass, jacobian, coefficient = bodies[body_index]
        result += mass * jacobian.T @ (coefficient * qdot[shoulder_index] ** 2)
        body_index += 1
        mass, jacobian, coefficients = bodies[body_index]
        angular_rates = np.array(
            [qdot[shoulder_index], qdot[shoulder_index] + qdot[elbow_index]]
        )
        result += mass * jacobian.T @ (coefficients @ angular_rates**2)
        body_index += 1
    mass, jacobian, coefficient = bodies[body_index]
    result += mass * jacobian.T @ (coefficient * qdot[8] ** 2)
    body_index += 1
    mass, jacobian, coefficients = bodies[body_index]
    club_rates = np.array([qdot[8], qdot[8] + qdot[9]])
    result += mass * jacobian.T @ (coefficients @ club_rates**2)
    return result


def _potential_gradient(q: FloatArray, params: MovingBaseFlexibleParams) -> FloatArray:
    bodies, _ = _body_jacobians(q, params)
    gradient = np.zeros(N_COORDINATES)
    for mass, jacobian, _bias in bodies:
        gradient += mass * params.gravity_m_s2 * jacobian[1]
    gradient[4:6] += params.base_stiffness_n_m * q[4:6]
    gradient[9] += params.shaft_stiffness_nm_rad * q[9]
    return gradient


def _damping_force(qdot: FloatArray, params: MovingBaseFlexibleParams) -> FloatArray:
    force = np.zeros(N_COORDINATES)
    force[:4] = -params.joint_damping_nms_rad * qdot[:4]
    force[4:6] = -params.base_damping_ns_m * qdot[4:6]
    force[9] = -params.shaft_damping_nms_rad * qdot[9]
    return force


def constraint_vector(q: object, params: MovingBaseFlexibleParams) -> FloatArray:
    """Return both hand-minus-grip closure residuals."""
    points = kinematics(q, params)
    return np.concatenate(
        (
            points["right_hand"] - points["right_grip"],
            points["left_hand"] - points["left_grip"],
        )
    )


def constraint_jacobian(q: object, params: MovingBaseFlexibleParams) -> FloatArray:
    """Return the exact four-by-ten hand closure Jacobian."""
    state = _state("q", q)
    jacobian = np.zeros((N_CONSTRAINTS, N_COORDINATES))
    for row, shoulder_index, elbow_index, grip_offset in (
        (0, 0, 1, params.right_grip_offset_m),
        (2, 2, 3, params.left_grip_offset_m),
    ):
        shoulder = state[shoulder_index]
        forearm = shoulder + state[elbow_index]
        jacobian[row : row + 2, shoulder_index] = params.upper_length_m * _derivative(
            shoulder
        ) + params.forearm_length_m * _derivative(forearm)
        jacobian[row : row + 2, elbow_index] = params.forearm_length_m * _derivative(
            forearm
        )
        jacobian[row : row + 2, 4:6] = np.eye(2)
        jacobian[row : row + 2, 6:8] = -np.eye(2)
        jacobian[row : row + 2, 8] = -grip_offset * _derivative(state[8])
    return jacobian


def _constraint_acceleration_bias(
    q: FloatArray, qdot: FloatArray, params: MovingBaseFlexibleParams
) -> FloatArray:
    speed = float(np.linalg.norm(qdot))
    if speed == 0.0:
        return np.zeros(N_CONSTRAINTS)
    step = 1e-6 / max(1.0, speed)
    derivative = (
        constraint_jacobian(q + step * qdot, params)
        - constraint_jacobian(q - step * qdot, params)
    ) / (2.0 * step)
    return derivative @ qdot


def control_generalized_force(control: TwoArmControl) -> FloatArray:
    """Map joint controls with explicit wrist action and reaction."""
    return np.array(
        [
            control.right_shoulder_nm - control.right_wrist_nm,
            control.right_elbow_nm - control.right_wrist_nm,
            control.left_shoulder_nm - control.left_wrist_nm,
            control.left_elbow_nm - control.left_wrist_nm,
            0.0,
            0.0,
            0.0,
            0.0,
            control.right_wrist_nm + control.left_wrist_nm,
            0.0,
        ]
    )


def solve_constrained_dynamics(
    q: object,
    qdot: object,
    control: TwoArmControl,
    params: MovingBaseFlexibleParams,
) -> DynamicsSolution:
    """Solve the coupled KKT system and fail closed on invalid topology."""
    state = _state("q", q)
    velocity = _state("qdot", qdot)
    violation = float(np.linalg.norm(constraint_vector(state, params)))
    if violation > params.constraint_tolerance_m:
        raise ValueError(
            "configuration violates hand constraints: "
            f"{violation:.3e} m > {params.constraint_tolerance_m:.3e} m"
        )
    matrix = mass_matrix(state, params)
    if float(np.min(np.linalg.eigvalsh(matrix))) <= params.rank_tolerance:
        raise ValueError("mass matrix is not positive definite")
    jacobian = constraint_jacobian(state, params)
    rank = int(np.linalg.matrix_rank(jacobian, tol=params.rank_tolerance))
    if rank != N_CONSTRAINTS:
        raise ValueError(
            f"constraint Jacobian rank is {rank}; expected {N_CONSTRAINTS}"
        )
    bias = _velocity_bias(state, velocity, params) + _potential_gradient(state, params)
    generalized = control_generalized_force(control) + _damping_force(velocity, params)
    gamma = _constraint_acceleration_bias(state, velocity, params)
    kkt = np.block(
        [
            [matrix, -jacobian.T],
            [jacobian, np.zeros((N_CONSTRAINTS, N_CONSTRAINTS))],
        ]
    )
    rhs = np.concatenate((generalized - bias, -gamma))
    try:
        result = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("KKT system is singular; no fallback is allowed") from exc
    residual = float(np.linalg.norm(kkt @ result - rhs))
    acceleration_residual = float(
        np.linalg.norm(jacobian @ result[:N_COORDINATES] + gamma)
    )
    if residual > params.kkt_tolerance or acceleration_residual > params.kkt_tolerance:
        raise RuntimeError(
            "coupled constrained solve exceeded tolerance: "
            f"KKT={residual:.3e}, constraint={acceleration_residual:.3e}"
        )
    multipliers = result[N_COORDINATES:]
    return DynamicsSolution(
        qddot=result[:N_COORDINATES],
        multipliers_n=multipliers,
        contact_force_on_club_n=-multipliers.reshape(2, 2),
        constraint_rank=rank,
        kkt_residual_norm=residual,
        acceleration_constraint_residual_norm=acceleration_residual,
    )


def potential_energy(q: object, params: MovingBaseFlexibleParams) -> float:
    """Return gravity plus base and shaft elastic potential energy."""
    state = _state("q", q)
    bodies, _ = _body_jacobians(state, params)
    # COM positions are reconstructed from the same geometry used by Jacobians.
    base = state[4:6]
    values = [params.base_mass_kg * base[1]]
    for shoulder_index, elbow_index in ((0, 1), (2, 3)):
        shoulder_y = base[1]
        upper_y = shoulder_y - 0.5 * params.upper_length_m * np.cos(
            state[shoulder_index]
        )
        forearm_y = (
            shoulder_y
            - params.upper_length_m * np.cos(state[shoulder_index])
            - 0.5
            * params.forearm_length_m
            * np.cos(state[shoulder_index] + state[elbow_index])
        )
        values.extend(
            [params.upper_mass_kg * upper_y, params.forearm_mass_kg * forearm_y]
        )
    proximal_y = state[7] - 0.5 * params.proximal_club_length_m * np.cos(state[8])
    distal_y = (
        state[7]
        - params.proximal_club_length_m * np.cos(state[8])
        - 0.5 * params.distal_club_length_m * np.cos(state[8] + state[9])
    )
    values.extend(
        [
            params.proximal_club_mass_kg * proximal_y,
            params.distal_club_mass_kg * distal_y,
        ]
    )
    del bodies
    return float(
        params.gravity_m_s2 * sum(values)
        + 0.5 * params.base_stiffness_n_m * float(base @ base)
        + 0.5 * params.shaft_stiffness_nm_rad * state[9] ** 2
    )


def mechanical_energy(
    q: object, qdot: object, params: MovingBaseFlexibleParams
) -> float:
    """Return kinetic plus all conservative potential energy."""
    state = _state("q", q)
    velocity = _state("qdot", qdot)
    return 0.5 * float(
        velocity @ mass_matrix(state, params) @ velocity
    ) + potential_energy(state, params)


def _mass_metric_correction(
    q: FloatArray,
    residual: FloatArray,
    params: MovingBaseFlexibleParams,
) -> FloatArray:
    matrix = mass_matrix(q, params)
    jacobian = constraint_jacobian(q, params)
    inverse_jacobian = np.linalg.solve(matrix, jacobian.T)
    schur = jacobian @ inverse_jacobian
    if np.linalg.matrix_rank(schur, tol=params.rank_tolerance) != N_CONSTRAINTS:
        raise ValueError("constraint projection is singular; no fallback is allowed")
    return inverse_jacobian @ np.linalg.solve(schur, residual)


def _project_configuration(
    q: FloatArray,
    params: MovingBaseFlexibleParams,
    config: MovingBaseFlexibleConfig,
) -> tuple[FloatArray, float]:
    result = q.copy()
    total = np.zeros_like(q)
    for _ in range(config.maximum_projection_iterations):
        residual = constraint_vector(result, params)
        if np.linalg.norm(residual) <= config.projection_tolerance_m:
            return result, float(np.linalg.norm(total))
        correction = _mass_metric_correction(result, residual, params)
        result -= correction
        total -= correction
    raise ValueError("position projection failed to converge")


def _project_velocity(
    q: FloatArray,
    qdot: FloatArray,
    params: MovingBaseFlexibleParams,
    config: MovingBaseFlexibleConfig,
) -> FloatArray:
    jacobian = constraint_jacobian(q, params)
    residual = jacobian @ qdot
    if np.linalg.norm(residual) <= config.velocity_tolerance_m_s:
        return qdot.copy()
    result = qdot - _mass_metric_correction(q, residual, params)
    if np.linalg.norm(jacobian @ result) > config.velocity_tolerance_m_s:
        raise ValueError("velocity projection failed to converge")
    return result


def _grip_velocity(
    q: FloatArray,
    qdot: FloatArray,
    offset_m: float,
) -> FloatArray:
    return qdot[6:8] + offset_m * qdot[8] * _derivative(q[8])


def _clubhead_jacobian(q: FloatArray, params: MovingBaseFlexibleParams) -> FloatArray:
    jacobian = np.zeros((2, N_COORDINATES))
    jacobian[:, 6:8] = np.eye(2)
    jacobian[:, 8] = params.proximal_club_length_m * _derivative(
        q[8]
    ) + params.distal_club_length_m * _derivative(q[8] + q[9])
    jacobian[:, 9] = params.distal_club_length_m * _derivative(q[8] + q[9])
    return jacobian


def rollout(
    q0: object,
    qdot0: object,
    control_law: ControlLaw,
    params: MovingBaseFlexibleParams,
    config: MovingBaseFlexibleConfig,
) -> MovingBaseFlexibleTrace:
    """Integrate the coupled constrained system with explicit projections."""
    q_initial, first_correction = _project_configuration(
        _state("q0", q0), params, config
    )
    velocity_initial = _project_velocity(
        q_initial, _state("qdot0", qdot0), params, config
    )
    samples = config.interval_count + 1
    time = np.asarray(
        config.start_time_s + np.arange(samples, dtype=np.float64) * config.step_s,
        dtype=np.float64,
    )
    q = np.empty((samples, N_COORDINATES))
    qdot = np.empty_like(q)
    corrections = np.zeros(samples)
    projection_energy = np.zeros(samples)
    q[0], qdot[0], corrections[0] = q_initial, velocity_initial, first_correction
    for index in range(samples - 1):
        control = control_law(float(time[index]), q[index].copy(), qdot[index].copy())
        solution = solve_constrained_dynamics(q[index], qdot[index], control, params)
        half_velocity = qdot[index] + 0.5 * config.step_s * solution.qddot
        trial_q = q[index] + config.step_s * half_velocity
        energy_before_q = mechanical_energy(trial_q, half_velocity, params)
        q[index + 1], corrections[index + 1] = _project_configuration(
            trial_q, params, config
        )
        half_velocity = _project_velocity(q[index + 1], half_velocity, params, config)
        q_projection_energy = (
            mechanical_energy(q[index + 1], half_velocity, params) - energy_before_q
        )
        next_control = control_law(
            float(time[index + 1]), q[index + 1].copy(), half_velocity.copy()
        )
        next_solution = solve_constrained_dynamics(
            q[index + 1], half_velocity, next_control, params
        )
        trial_velocity = half_velocity + 0.5 * config.step_s * next_solution.qddot
        energy_before_v = mechanical_energy(q[index + 1], trial_velocity, params)
        qdot[index + 1] = _project_velocity(
            q[index + 1], trial_velocity, params, config
        )
        projection_energy[index + 1] = (
            mechanical_energy(q[index + 1], qdot[index + 1], params)
            - energy_before_v
            + q_projection_energy
        )

    controls = tuple(
        control_law(float(t), state.copy(), velocity.copy())
        for t, state, velocity in zip(time, q, qdot, strict=True)
    )
    qddot = np.empty_like(q)
    multipliers = np.empty((samples, N_CONSTRAINTS))
    contacts = np.empty((samples, 2, 2))
    rank = np.empty(samples, dtype=np.int64)
    kkt = np.empty(samples)
    acceleration = np.empty(samples)
    position = np.empty(samples)
    velocity_residual = np.empty(samples)
    energy = np.empty(samples)
    force_couple = np.empty(samples)
    direct_wrist = np.empty(samples)
    contact_power = np.empty(samples)
    wrench_power = np.empty(samples)
    applied_power = np.empty(samples)
    dissipation_power = np.empty(samples)
    clubhead = np.empty((samples, 2))
    clubhead_velocity = np.empty_like(clubhead)
    for index, (state, velocity, control) in enumerate(
        zip(q, qdot, controls, strict=True)
    ):
        solved = solve_constrained_dynamics(state, velocity, control, params)
        qddot[index] = solved.qddot
        multipliers[index] = solved.multipliers_n
        contacts[index] = solved.contact_force_on_club_n
        rank[index] = solved.constraint_rank
        kkt[index] = solved.kkt_residual_norm
        acceleration[index] = solved.acceleration_constraint_residual_norm
        position[index] = np.linalg.norm(constraint_vector(state, params))
        velocity_residual[index] = np.linalg.norm(
            constraint_jacobian(state, params) @ velocity
        )
        energy[index] = mechanical_energy(state, velocity, params)
        points = kinematics(state, params)
        right_offset = points["right_grip"] - points["grip_center"]
        left_offset = points["left_grip"] - points["grip_center"]
        force_couple[index] = _cross_z(right_offset, contacts[index, 0]) + _cross_z(
            left_offset, contacts[index, 1]
        )
        direct_wrist[index] = control.right_wrist_nm + control.left_wrist_nm
        right_velocity = _grip_velocity(state, velocity, params.right_grip_offset_m)
        left_velocity = _grip_velocity(state, velocity, params.left_grip_offset_m)
        contact_power[index] = (
            contacts[index, 0] @ right_velocity + contacts[index, 1] @ left_velocity
        )
        wrench_power[index] = (contacts[index, 0] + contacts[index, 1]) @ velocity[
            6:8
        ] + force_couple[index] * velocity[8]
        generalized_control = control_generalized_force(control)
        applied_power[index] = generalized_control @ velocity
        damping = _damping_force(velocity, params)
        dissipation_power[index] = damping @ velocity
        clubhead[index] = points["clubhead"]
        clubhead_velocity[index] = _clubhead_jacobian(state, params) @ velocity
    flex = q[:, 9]
    flex_rate = qdot[:, 9]
    return MovingBaseFlexibleTrace(
        time=time,
        q=q,
        qdot=qdot,
        qddot=qddot,
        controls=controls,
        multipliers_n=multipliers,
        contact_force_on_club_n=contacts,
        force_generated_couple_nm=force_couple,
        direct_wrist_torque_nm=direct_wrist,
        contact_power_w=contact_power,
        contact_wrench_power_w=wrench_power,
        contact_power_identity_residual_w=contact_power - wrench_power,
        shaft_elastic_moment_nm=-params.shaft_stiffness_nm_rad * flex,
        shaft_damping_moment_nm=-params.shaft_damping_nms_rad * flex_rate,
        shaft_strain_energy_j=0.5 * params.shaft_stiffness_nm_rad * flex**2,
        clubhead_position_m=clubhead,
        clubhead_velocity_m_s=clubhead_velocity,
        mechanical_energy_j=energy,
        applied_control_power_w=applied_power,
        dissipation_power_w=dissipation_power,
        position_constraint_norm_m=position,
        velocity_constraint_norm_m_s=velocity_residual,
        kkt_residual_norm=kkt,
        acceleration_constraint_residual_norm=acceleration,
        projection_correction_norm_m=corrections,
        projection_energy_change_j=projection_energy,
    )


__all__ = [
    "DynamicsSolution",
    "MovingBaseFlexibleConfig",
    "MovingBaseFlexibleParams",
    "MovingBaseFlexibleTrace",
    "constraint_jacobian",
    "constraint_vector",
    "control_generalized_force",
    "initial_state",
    "kinematics",
    "mass_matrix",
    "mechanical_energy",
    "potential_energy",
    "rollout",
    "solve_constrained_dynamics",
]
