"""Forward rotating-base, two-hand, compliant-club mechanism tier.

The seven generalized coordinates are torso rotation, lead/trail arm rotation
relative to the torso, grip-center translation, proximal-club rotation, and
distal-club rotation relative to the proximal segment. Two holonomic point
constraints per hand close the bilateral loop. The tier is intentionally
reduced and planar; its torso coordinate is not a human coaching observable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
N_COORDINATES = 7
N_CONSTRAINTS = 4


def _direction(angle: float) -> FloatArray:
    return np.array([np.sin(angle), -np.cos(angle)], dtype=float)


def _direction_derivative(angle: float) -> FloatArray:
    return np.array([np.cos(angle), np.sin(angle)], dtype=float)


def _rotate(vector: npt.ArrayLike, angle: float) -> FloatArray:
    x, y = np.asarray(vector, dtype=float)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([cosine * x - sine * y, sine * x + cosine * y])


def _rotate_derivative(vector: npt.ArrayLike, angle: float) -> FloatArray:
    x, y = np.asarray(vector, dtype=float)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([-sine * x - cosine * y, cosine * x - sine * y])


def _finite_vector(name: str, value: object, shape: tuple[int, ...]) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape {shape}")
    return array.copy()


@dataclass(frozen=True, slots=True)
class RotatingBaseParams:
    """Physical parameters for the qualified reduced mechanism tier."""

    torso_inertia_kg_m2: float = 4.8
    torso_stiffness_nm_rad: float = 0.0
    torso_damping_nms_rad: float = 0.35
    lead_shoulder_offset_m: tuple[float, float] = (0.19, 0.0)
    trail_shoulder_offset_m: tuple[float, float] = (-0.19, 0.0)
    arm_length_m: float = 0.62
    arm_mass_kg: float = 3.1
    arm_inertia_kg_m2: float = 0.0993
    arm_damping_nms_rad: float = 0.10
    proximal_club_length_m: float = 0.46
    distal_club_length_m: float = 0.54
    proximal_club_mass_kg: float = 0.18
    distal_club_mass_kg: float = 0.27
    proximal_club_inertia_kg_m2: float = 0.003174
    distal_club_inertia_kg_m2: float = 0.006561
    shaft_stiffness_nm_rad: float = 80.0
    shaft_damping_nms_rad: float = 0.6
    lead_grip_offset_m: float = 0.065
    trail_grip_offset_m: float = -0.065
    gravity_m_s2: float = 9.80665
    rank_tolerance: float = 1e-10
    kkt_tolerance: float = 1e-8

    @classmethod
    def publication_default(cls) -> RotatingBaseParams:
        """Return the deterministic publication parameter set."""
        return cls()

    def __post_init__(self) -> None:
        positive = (
            "torso_inertia_kg_m2",
            "arm_length_m",
            "arm_mass_kg",
            "arm_inertia_kg_m2",
            "proximal_club_length_m",
            "distal_club_length_m",
            "proximal_club_mass_kg",
            "distal_club_mass_kg",
            "proximal_club_inertia_kg_m2",
            "distal_club_inertia_kg_m2",
            "shaft_stiffness_nm_rad",
            "rank_tolerance",
            "kkt_tolerance",
        )
        nonnegative = (
            "torso_stiffness_nm_rad",
            "torso_damping_nms_rad",
            "arm_damping_nms_rad",
            "shaft_damping_nms_rad",
            "gravity_m_s2",
        )
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in nonnegative:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("lead_shoulder_offset_m", "trail_shoulder_offset_m"):
            _finite_vector(name, getattr(self, name), (2,))
        grips = np.array([self.lead_grip_offset_m, self.trail_grip_offset_m])
        if not np.all(np.isfinite(grips)):
            raise ValueError("grip offsets must be finite")


@dataclass(frozen=True, slots=True)
class RotatingBaseConfig:
    """Integration and projection contract."""

    duration_s: float
    step_s: float
    projection_tolerance_m: float = 1e-11
    velocity_tolerance_m_s: float = 1e-10
    maximum_projection_iterations: int = 16

    def __post_init__(self) -> None:
        for name in (
            "duration_s",
            "step_s",
            "projection_tolerance_m",
            "velocity_tolerance_m_s",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_projection_iterations < 1:
            raise ValueError("maximum_projection_iterations must be positive")
        intervals = self.duration_s / self.step_s
        if not np.isclose(intervals, round(intervals), atol=1e-10):
            raise ValueError("duration_s must be an integer multiple of step_s")

    @property
    def interval_count(self) -> int:
        return int(round(self.duration_s / self.step_s))


@dataclass(frozen=True, slots=True)
class RotatingBaseState:
    """One finite generalized state."""

    q: FloatArray
    qdot: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "q", _finite_vector("q", self.q, (N_COORDINATES,)))
        object.__setattr__(
            self, "qdot", _finite_vector("qdot", self.qdot, (N_COORDINATES,))
        )


@dataclass(frozen=True, slots=True)
class TorsoTwoHandControl:
    """Torso, bilateral arm, and bilateral wrist generalized commands."""

    torso_nm: float = 0.0
    lead_arm_nm: float = 0.0
    trail_arm_nm: float = 0.0
    lead_wrist_nm: float = 0.0
    trail_wrist_nm: float = 0.0

    def __post_init__(self) -> None:
        if not np.all(np.isfinite(tuple(self.as_array()))):
            raise ValueError("control values must be finite")

    def as_array(self) -> FloatArray:
        return np.array(
            [
                self.torso_nm,
                self.lead_arm_nm,
                self.trail_arm_nm,
                self.lead_wrist_nm,
                self.trail_wrist_nm,
            ],
            dtype=float,
        )


@dataclass(frozen=True, slots=True)
class DynamicsSolution:
    """One full-rank constrained dynamics solution."""

    qddot: FloatArray
    multipliers_n: FloatArray
    force_on_hands_n: FloatArray
    force_on_club_n: FloatArray
    force_generated_couple_nm: float
    constraint_rank: int
    kkt_residual_norm: float
    acceleration_constraint_residual_norm: float


@dataclass(frozen=True, slots=True)
class RotatingBaseTrace:
    """Qualified trajectory and its force, power, and energy ledgers."""

    time: FloatArray
    q: FloatArray
    qdot: FloatArray
    qddot: FloatArray
    controls: tuple[TorsoTwoHandControl, ...]
    force_on_club_n: FloatArray
    force_generated_couple_nm: FloatArray
    contact_power_on_club_w: FloatArray
    contact_power_identity_residual_w: FloatArray
    clubhead_velocity_m_s: FloatArray
    clubhead_speed_m_s: FloatArray
    distal_segment_kinetic_energy_j: FloatArray
    mechanical_energy_j: FloatArray
    control_power_w: FloatArray
    dissipation_power_w: FloatArray
    position_constraint_norm_m: FloatArray
    velocity_constraint_norm_m_s: FloatArray
    projection_energy_change_j: FloatArray
    work_energy_closure_j: float
    model_tier: str = "planar_rotating_base_two_hand_compliant_club"


ControlLaw = Callable[[float, RotatingBaseState], TorsoTwoHandControl]


def _points(q: FloatArray, params: RotatingBaseParams) -> dict[str, FloatArray]:
    phi, lead_relative, trail_relative, x_grip, y_grip, alpha, beta = q
    center = np.array([x_grip, y_grip])
    lead_shoulder = _rotate(params.lead_shoulder_offset_m, phi)
    trail_shoulder = _rotate(params.trail_shoulder_offset_m, phi)
    lead_hand = lead_shoulder + params.arm_length_m * _direction(phi + lead_relative)
    trail_hand = trail_shoulder + params.arm_length_m * _direction(phi + trail_relative)
    lead_grip = center + params.lead_grip_offset_m * _direction(alpha)
    trail_grip = center + params.trail_grip_offset_m * _direction(alpha)
    flex_joint = center + params.proximal_club_length_m * _direction(alpha)
    clubhead = flex_joint + params.distal_club_length_m * _direction(alpha + beta)
    return {
        "lead_shoulder": lead_shoulder,
        "trail_shoulder": trail_shoulder,
        "lead_hand": lead_hand,
        "trail_hand": trail_hand,
        "grip_center": center,
        "lead_grip": lead_grip,
        "trail_grip": trail_grip,
        "flex_joint": flex_joint,
        "clubhead": clubhead,
    }


def kinematics(q: object, params: RotatingBaseParams) -> dict[str, FloatArray]:
    """Return all declared points in the planar world frame."""
    return _points(_finite_vector("q", q, (N_COORDINATES,)), params)


def constraint_vector(q: object, params: RotatingBaseParams) -> FloatArray:
    """Return lead and trail hand-minus-grip position residuals."""
    points = kinematics(q, params)
    return np.concatenate(
        (
            points["lead_hand"] - points["lead_grip"],
            points["trail_hand"] - points["trail_grip"],
        )
    )


def constraint_jacobian(q: object, params: RotatingBaseParams) -> FloatArray:
    """Return the analytic four-by-seven loop-closure Jacobian."""
    state = _finite_vector("q", q, (N_COORDINATES,))
    phi, lead_relative, trail_relative, _x, _y, alpha, _beta = state
    matrix = np.zeros((N_CONSTRAINTS, N_COORDINATES))
    for row, relative_index, shoulder_offset, grip_offset in (
        (0, 1, params.lead_shoulder_offset_m, params.lead_grip_offset_m),
        (2, 2, params.trail_shoulder_offset_m, params.trail_grip_offset_m),
    ):
        absolute = phi + state[relative_index]
        arm_derivative = params.arm_length_m * _direction_derivative(absolute)
        matrix[row : row + 2, 0] = (
            _rotate_derivative(shoulder_offset, phi) + arm_derivative
        )
        matrix[row : row + 2, relative_index] = arm_derivative
        matrix[row : row + 2, 3:5] = -np.eye(2)
        matrix[row : row + 2, 5] = -grip_offset * _direction_derivative(alpha)
    return matrix


def _body_jacobians(
    q: FloatArray, params: RotatingBaseParams
) -> tuple[tuple[float, FloatArray, FloatArray], ...]:
    phi, lead_relative, trail_relative, _x, _y, alpha, beta = q
    bodies: list[tuple[float, FloatArray, FloatArray]] = []
    for relative_index, shoulder_offset in (
        (1, params.lead_shoulder_offset_m),
        (2, params.trail_shoulder_offset_m),
    ):
        absolute = phi + q[relative_index]
        arm_term = 0.5 * params.arm_length_m * _direction_derivative(absolute)
        jacobian = np.zeros((2, N_COORDINATES))
        jacobian[:, 0] = _rotate_derivative(shoulder_offset, phi) + arm_term
        jacobian[:, relative_index] = arm_term
        angular = np.zeros(N_COORDINATES)
        angular[[0, relative_index]] = 1.0
        bodies.append((params.arm_mass_kg, jacobian, angular))
    proximal = np.zeros((2, N_COORDINATES))
    proximal[:, 3:5] = np.eye(2)
    proximal[:, 5] = 0.5 * params.proximal_club_length_m * _direction_derivative(alpha)
    proximal_angular = np.zeros(N_COORDINATES)
    proximal_angular[5] = 1.0
    bodies.append((params.proximal_club_mass_kg, proximal, proximal_angular))
    distal = np.zeros((2, N_COORDINATES))
    distal[:, 3:5] = np.eye(2)
    distal[:, 5] = params.proximal_club_length_m * _direction_derivative(
        alpha
    ) + 0.5 * params.distal_club_length_m * _direction_derivative(alpha + beta)
    distal[:, 6] = (
        0.5 * params.distal_club_length_m * _direction_derivative(alpha + beta)
    )
    distal_angular = np.zeros(N_COORDINATES)
    distal_angular[[5, 6]] = 1.0
    bodies.append((params.distal_club_mass_kg, distal, distal_angular))
    return tuple(bodies)


def mass_matrix(q: object, params: RotatingBaseParams) -> FloatArray:
    """Return the symmetric positive-definite generalized mass matrix."""
    state = _finite_vector("q", q, (N_COORDINATES,))
    matrix = np.zeros((N_COORDINATES, N_COORDINATES))
    matrix[0, 0] += params.torso_inertia_kg_m2
    inertias = (
        params.arm_inertia_kg_m2,
        params.arm_inertia_kg_m2,
        params.proximal_club_inertia_kg_m2,
        params.distal_club_inertia_kg_m2,
    )
    for (mass, linear, angular), inertia in zip(
        _body_jacobians(state, params), inertias, strict=True
    ):
        matrix += mass * linear.T @ linear + inertia * np.outer(angular, angular)
    return 0.5 * (matrix + matrix.T)


def potential_energy(q: object, params: RotatingBaseParams) -> float:
    """Return gravitational, torso-spring, and shaft-spring energy."""
    state = _finite_vector("q", q, (N_COORDINATES,))
    points = _points(state, params)
    phi, lead_relative, trail_relative, _x, _y, alpha, beta = state
    lead_com = points["lead_shoulder"] + 0.5 * params.arm_length_m * _direction(
        phi + lead_relative
    )
    trail_com = points["trail_shoulder"] + 0.5 * params.arm_length_m * _direction(
        phi + trail_relative
    )
    center = points["grip_center"]
    proximal_com = center + 0.5 * params.proximal_club_length_m * _direction(alpha)
    distal_com = points["flex_joint"] + 0.5 * params.distal_club_length_m * _direction(
        alpha + beta
    )
    gravity = params.gravity_m_s2 * (
        params.arm_mass_kg * (lead_com[1] + trail_com[1])
        + params.proximal_club_mass_kg * proximal_com[1]
        + params.distal_club_mass_kg * distal_com[1]
    )
    elastic = 0.5 * params.torso_stiffness_nm_rad * phi**2
    elastic += 0.5 * params.shaft_stiffness_nm_rad * beta**2
    return float(gravity + elastic)


def mechanical_energy(state: RotatingBaseState, params: RotatingBaseParams) -> float:
    """Return kinetic plus conservative potential energy."""
    return float(
        0.5 * state.qdot @ mass_matrix(state.q, params) @ state.qdot
        + potential_energy(state.q, params)
    )


def distal_segment_kinetic_energy(
    state: RotatingBaseState, params: RotatingBaseParams
) -> float:
    """Return distal-club translational plus rotational kinetic energy."""
    distal = _body_jacobians(state.q, params)[3]
    _mass, linear_jacobian, angular_jacobian = distal
    linear_velocity = linear_jacobian @ state.qdot
    angular_velocity = float(angular_jacobian @ state.qdot)
    return float(
        0.5 * params.distal_club_mass_kg * linear_velocity @ linear_velocity
        + 0.5 * params.distal_club_inertia_kg_m2 * angular_velocity**2
    )


def _gradient(function: Callable[[FloatArray], float], q: FloatArray) -> FloatArray:
    step = 2e-6
    result = np.empty(N_COORDINATES)
    for index in range(N_COORDINATES):
        offset = np.zeros(N_COORDINATES)
        offset[index] = step
        result[index] = (function(q + offset) - function(q - offset)) / (2.0 * step)
    return result


def _bias_force(
    q: FloatArray, qdot: FloatArray, params: RotatingBaseParams
) -> FloatArray:
    step = 2e-6
    derivatives = np.empty((N_COORDINATES, N_COORDINATES, N_COORDINATES))
    for index in range(N_COORDINATES):
        offset = np.zeros(N_COORDINATES)
        offset[index] = step
        derivatives[index] = (
            mass_matrix(q + offset, params) - mass_matrix(q - offset, params)
        ) / (2.0 * step)
    coriolis = np.zeros(N_COORDINATES)
    for i in range(N_COORDINATES):
        for j in range(N_COORDINATES):
            for k in range(N_COORDINATES):
                christoffel = 0.5 * (
                    derivatives[k, i, j] + derivatives[j, i, k] - derivatives[i, j, k]
                )
                coriolis[i] += christoffel * qdot[j] * qdot[k]
    return coriolis + _gradient(lambda value: potential_energy(value, params), q)


def _damping_force(qdot: FloatArray, params: RotatingBaseParams) -> FloatArray:
    force = np.zeros(N_COORDINATES)
    force[0] = -params.torso_damping_nms_rad * qdot[0]
    force[1:3] = -params.arm_damping_nms_rad * qdot[1:3]
    force[6] = -params.shaft_damping_nms_rad * qdot[6]
    return force


def control_generalized_force(control: TorsoTwoHandControl) -> FloatArray:
    """Map actuator moments to generalized forces by virtual work."""
    force = np.zeros(N_COORDINATES)
    force[0] += control.torso_nm
    force[1] += control.lead_arm_nm
    force[2] += control.trail_arm_nm
    for relative_index, wrist in (
        (1, control.lead_wrist_nm),
        (2, control.trail_wrist_nm),
    ):
        force[0] -= wrist
        force[relative_index] -= wrist
        force[5] += wrist
    return force


def _constraint_acceleration_bias(
    q: FloatArray, qdot: FloatArray, params: RotatingBaseParams
) -> FloatArray:
    phi, _lead, _trail, _x, _y, alpha, _beta = q
    result = np.empty(N_CONSTRAINTS)
    for row, relative_index, shoulder_offset, grip_offset in (
        (0, 1, params.lead_shoulder_offset_m, params.lead_grip_offset_m),
        (2, 2, params.trail_shoulder_offset_m, params.trail_grip_offset_m),
    ):
        absolute = phi + q[relative_index]
        absolute_rate = qdot[0] + qdot[relative_index]
        hand_bias = -_rotate(shoulder_offset, phi) * qdot[0] ** 2
        hand_bias -= params.arm_length_m * _direction(absolute) * absolute_rate**2
        grip_bias = -grip_offset * _direction(alpha) * qdot[5] ** 2
        result[row : row + 2] = hand_bias - grip_bias
    return result


def solve_constrained_dynamics(
    state: RotatingBaseState,
    control: TorsoTwoHandControl,
    params: RotatingBaseParams,
) -> DynamicsSolution:
    """Solve the full-rank KKT system and return bilateral reactions."""
    position_residual = constraint_vector(state.q, params)
    if np.linalg.norm(position_residual) > 1e-7:
        raise ValueError("state violates bilateral position constraints")
    jacobian = constraint_jacobian(state.q, params)
    rank = int(np.linalg.matrix_rank(jacobian, tol=params.rank_tolerance))
    if rank != N_CONSTRAINTS:
        raise ValueError("bilateral constraint Jacobian is rank deficient")
    matrix = mass_matrix(state.q, params)
    bias = _bias_force(state.q, state.qdot, params)
    applied = control_generalized_force(control) + _damping_force(state.qdot, params)
    gamma = _constraint_acceleration_bias(state.q, state.qdot, params)
    kkt = np.block([[matrix, -jacobian.T], [jacobian, np.zeros((N_CONSTRAINTS,) * 2)]])
    rhs = np.concatenate((applied - bias, -gamma))
    solved = np.linalg.solve(kkt, rhs)
    qddot, multipliers = solved[:N_COORDINATES], solved[N_COORDINATES:]
    force_hands = multipliers.reshape(2, 2)
    force_club = -force_hands
    offsets = np.array([params.lead_grip_offset_m, params.trail_grip_offset_m])[
        :, None
    ] * _direction(state.q[5])
    couple = float(
        np.sum(offsets[:, 0] * force_club[:, 1] - offsets[:, 1] * force_club[:, 0])
    )
    residual = kkt @ solved - rhs
    acceleration_residual = jacobian @ qddot + gamma
    return DynamicsSolution(
        qddot=qddot,
        multipliers_n=multipliers,
        force_on_hands_n=force_hands,
        force_on_club_n=force_club,
        force_generated_couple_nm=couple,
        constraint_rank=rank,
        kkt_residual_norm=float(np.linalg.norm(residual)),
        acceleration_constraint_residual_norm=float(
            np.linalg.norm(acceleration_residual)
        ),
    )


def initial_state(
    params: RotatingBaseParams,
    *,
    torso_angle_rad: float = 0.0,
    torso_rate_rad_s: float = 0.0,
    club_rate_rad_s: float = 0.0,
) -> RotatingBaseState:
    """Return an exactly closed, velocity-consistent reference state."""
    alpha = 0.5 * np.pi
    shoulder_x = abs(params.lead_shoulder_offset_m[0])
    grip_x = abs(params.lead_grip_offset_m)
    horizontal = shoulder_x - grip_x
    if horizontal >= params.arm_length_m:
        raise ValueError("arm length must reach the separated grips")
    center_y = -float(np.sqrt(params.arm_length_m**2 - horizontal**2))
    center = np.array([0.0, center_y])
    q = np.array([torso_angle_rad, 0.0, 0.0, *center, alpha, 0.0])
    points = _points(q, params)
    for index, side in ((1, "lead"), (2, "trail")):
        vector = points[f"{side}_grip"] - points[f"{side}_shoulder"]
        absolute_angle = float(np.arctan2(vector[0], -vector[1]))
        q[index] = absolute_angle - torso_angle_rad
    fixed = np.array([0, 5, 6])
    unknown = np.array([1, 2, 3, 4])
    qdot = np.zeros(N_COORDINATES)
    qdot[fixed] = [torso_rate_rad_s, club_rate_rad_s, 0.0]
    jacobian = constraint_jacobian(q, params)
    qdot[unknown] = np.linalg.solve(
        jacobian[:, unknown], -jacobian[:, fixed] @ qdot[fixed]
    )
    state = RotatingBaseState(q, qdot)
    if np.linalg.norm(constraint_vector(q, params)) > 1e-10:
        raise ValueError("reference configuration did not close")
    return state


def _project_configuration(
    q: FloatArray, params: RotatingBaseParams, config: RotatingBaseConfig
) -> tuple[FloatArray, float]:
    projected = q.copy()
    correction_norm = 0.0
    for _ in range(config.maximum_projection_iterations):
        residual = constraint_vector(projected, params)
        if np.linalg.norm(residual) <= config.projection_tolerance_m:
            return projected, correction_norm
        jacobian = constraint_jacobian(projected, params)
        inverse_mass = np.linalg.inv(mass_matrix(projected, params))
        correction = (
            -inverse_mass
            @ jacobian.T
            @ np.linalg.solve(jacobian @ inverse_mass @ jacobian.T, residual)
        )
        projected += correction
        correction_norm += float(np.linalg.norm(correction))
    raise ValueError("configuration projection did not converge")


def _project_velocity(
    q: FloatArray, qdot: FloatArray, params: RotatingBaseParams
) -> FloatArray:
    jacobian = constraint_jacobian(q, params)
    inverse_mass = np.linalg.inv(mass_matrix(q, params))
    return qdot - inverse_mass @ jacobian.T @ np.linalg.solve(
        jacobian @ inverse_mass @ jacobian.T, jacobian @ qdot
    )


def _point_velocity_jacobians(
    q: FloatArray, params: RotatingBaseParams
) -> tuple[FloatArray, FloatArray, FloatArray]:
    alpha = q[5]
    lead_grip = np.zeros((2, N_COORDINATES))
    trail_grip = np.zeros((2, N_COORDINATES))
    lead_grip[:, 3:5] = np.eye(2)
    trail_grip[:, 3:5] = np.eye(2)
    lead_grip[:, 5] = params.lead_grip_offset_m * _direction_derivative(alpha)
    trail_grip[:, 5] = params.trail_grip_offset_m * _direction_derivative(alpha)
    center = np.zeros((2, N_COORDINATES))
    center[:, 3:5] = np.eye(2)
    beta = q[6]
    clubhead = center.copy()
    clubhead[:, 5] = params.proximal_club_length_m * _direction_derivative(
        alpha
    ) + params.distal_club_length_m * _direction_derivative(alpha + beta)
    clubhead[:, 6] = params.distal_club_length_m * _direction_derivative(alpha + beta)
    return lead_grip, trail_grip, clubhead


def rollout(
    initial: RotatingBaseState,
    control_law: ControlLaw,
    params: RotatingBaseParams,
    config: RotatingBaseConfig,
) -> RotatingBaseTrace:
    """Integrate and return a constraint- and energy-audited trajectory."""
    samples = config.interval_count + 1
    time = np.linspace(0.0, config.duration_s, samples)
    q = np.empty((samples, N_COORDINATES))
    qdot = np.empty_like(q)
    qddot = np.empty_like(q)
    q[0], qdot[0] = initial.q, initial.qdot
    projection_energy = np.zeros(samples)
    controls: list[TorsoTwoHandControl] = []
    for index in range(samples - 1):
        state = RotatingBaseState(q[index], qdot[index])
        control = control_law(float(time[index]), state)
        solution = solve_constrained_dynamics(state, control, params)
        controls.append(control)
        qddot[index] = solution.qddot
        trial_velocity = qdot[index] + config.step_s * solution.qddot
        trial_q = q[index] + config.step_s * trial_velocity
        before = mechanical_energy(RotatingBaseState(trial_q, trial_velocity), params)
        q[index + 1], _ = _project_configuration(trial_q, params, config)
        projected_velocity = _project_velocity(q[index + 1], trial_velocity, params)
        after = mechanical_energy(
            RotatingBaseState(q[index + 1], projected_velocity), params
        )
        qdot[index + 1] = projected_velocity
        projection_energy[index + 1] = after - before
    final_state = RotatingBaseState(q[-1], qdot[-1])
    controls.append(control_law(float(time[-1]), final_state))
    qddot[-1] = solve_constrained_dynamics(final_state, controls[-1], params).qddot

    forces = np.empty((samples, 2, 2))
    couples = np.empty(samples)
    contact_power = np.empty(samples)
    identity_residual = np.empty(samples)
    clubhead_velocity = np.empty((samples, 2))
    energy = np.empty(samples)
    distal_energy = np.empty(samples)
    control_power = np.empty(samples)
    dissipation_power = np.empty(samples)
    position_residual = np.empty(samples)
    velocity_residual = np.empty(samples)
    for index, control in enumerate(controls):
        state = RotatingBaseState(q[index], qdot[index])
        solution = solve_constrained_dynamics(state, control, params)
        forces[index] = solution.force_on_club_n
        couples[index] = solution.force_generated_couple_nm
        lead_jacobian, trail_jacobian, clubhead_jacobian = _point_velocity_jacobians(
            state.q, params
        )
        grip_velocities = np.stack(
            (lead_jacobian @ state.qdot, trail_jacobian @ state.qdot)
        )
        contact_power[index] = float(np.sum(forces[index] * grip_velocities))
        resultant = np.sum(forces[index], axis=0)
        center_velocity = state.qdot[3:5]
        wrench_power = float(
            resultant @ center_velocity + couples[index] * state.qdot[5]
        )
        identity_residual[index] = contact_power[index] - wrench_power
        clubhead_velocity[index] = clubhead_jacobian @ state.qdot
        energy[index] = mechanical_energy(state, params)
        distal_energy[index] = distal_segment_kinetic_energy(state, params)
        control_power[index] = control_generalized_force(control) @ state.qdot
        dissipation_power[index] = -(
            params.torso_damping_nms_rad * state.qdot[0] ** 2
            + params.arm_damping_nms_rad * np.sum(state.qdot[1:3] ** 2)
            + params.shaft_damping_nms_rad * state.qdot[6] ** 2
        )
        position_residual[index] = np.linalg.norm(constraint_vector(state.q, params))
        velocity_residual[index] = np.linalg.norm(
            constraint_jacobian(state.q, params) @ state.qdot
        )
    expected_change = float(
        np.trapezoid(control_power + dissipation_power, x=time)
        + np.sum(projection_energy)
    )
    closure = float(energy[-1] - energy[0] - expected_change)
    return RotatingBaseTrace(
        time=time,
        q=q,
        qdot=qdot,
        qddot=qddot,
        controls=tuple(controls),
        force_on_club_n=forces,
        force_generated_couple_nm=couples,
        contact_power_on_club_w=contact_power,
        contact_power_identity_residual_w=identity_residual,
        clubhead_velocity_m_s=clubhead_velocity,
        clubhead_speed_m_s=np.linalg.norm(clubhead_velocity, axis=1),
        distal_segment_kinetic_energy_j=distal_energy,
        mechanical_energy_j=energy,
        control_power_w=control_power,
        dissipation_power_w=dissipation_power,
        position_constraint_norm_m=position_residual,
        velocity_constraint_norm_m_s=velocity_residual,
        projection_energy_change_j=projection_energy,
        work_energy_closure_j=closure,
    )


__all__ = [
    "RotatingBaseConfig",
    "RotatingBaseParams",
    "RotatingBaseState",
    "RotatingBaseTrace",
    "TorsoTwoHandControl",
    "constraint_jacobian",
    "constraint_vector",
    "control_generalized_force",
    "distal_segment_kinetic_energy",
    "initial_state",
    "kinematics",
    "mass_matrix",
    "mechanical_energy",
    "potential_energy",
    "rollout",
    "solve_constrained_dynamics",
]
