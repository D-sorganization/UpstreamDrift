"""Mechanically explicit planar two-arm closed-loop research model.

The model exists to support the model-ladder evidence in the proximal-distal
study.  It is deliberately separate from the legacy eight-coordinate golfer
model, whose club translation is implicitly anchored at one hand.  Here the
club has three independent planar coordinates and each hand contributes two
independent position constraints.

The contact multipliers are model outputs, not measurements of muscle effort.
Likewise, a net club wrench does not identify the individual hand forces unless
the independent contact constraints (or an explicit allocation rule) are part
of the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.biomechanics.drift_control_transfer import (
    JointTransferTrajectory,
)

N_COORDINATES = 7
N_CONSTRAINTS = 4


@dataclass(frozen=True)
class TwoArmParams:
    """Physical parameters for two two-link arms and a floating planar club."""

    right_shoulder_m: tuple[float, float]
    left_shoulder_m: tuple[float, float]
    upper_length_m: float
    forearm_length_m: float
    upper_mass_kg: float
    forearm_mass_kg: float
    upper_inertia_kg_m2: float
    forearm_inertia_kg_m2: float
    club_mass_kg: float
    club_inertia_kg_m2: float
    right_grip_offset_m: float
    left_grip_offset_m: float
    gravity_m_s2: float = 9.80665
    rank_tolerance: float = 1e-10
    constraint_tolerance_m: float = 1e-8
    kkt_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        positive = {
            "upper_length_m": self.upper_length_m,
            "forearm_length_m": self.forearm_length_m,
            "upper_mass_kg": self.upper_mass_kg,
            "forearm_mass_kg": self.forearm_mass_kg,
            "upper_inertia_kg_m2": self.upper_inertia_kg_m2,
            "forearm_inertia_kg_m2": self.forearm_inertia_kg_m2,
            "club_mass_kg": self.club_mass_kg,
            "club_inertia_kg_m2": self.club_inertia_kg_m2,
            "rank_tolerance": self.rank_tolerance,
            "constraint_tolerance_m": self.constraint_tolerance_m,
            "kkt_tolerance": self.kkt_tolerance,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.gravity_m_s2) or self.gravity_m_s2 < 0.0:
            raise ValueError("gravity_m_s2 must be finite and non-negative")
        for name, point in (
            ("right_shoulder_m", self.right_shoulder_m),
            ("left_shoulder_m", self.left_shoulder_m),
        ):
            array = np.asarray(point, dtype=float)
            if array.shape != (2,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain two finite coordinates")
        offsets = np.asarray(
            [self.right_grip_offset_m, self.left_grip_offset_m], dtype=float
        )
        if not np.all(np.isfinite(offsets)):
            raise ValueError("grip offsets must be finite")

    @classmethod
    def publication_default(cls) -> TwoArmParams:
        """Return a deterministic, human-scale parameter set for experiments."""
        upper_length = 0.32
        forearm_length = 0.30
        upper_mass = 1.9
        forearm_mass = 1.35
        return cls(
            right_shoulder_m=(0.19, 0.0),
            left_shoulder_m=(-0.19, 0.0),
            upper_length_m=upper_length,
            forearm_length_m=forearm_length,
            upper_mass_kg=upper_mass,
            forearm_mass_kg=forearm_mass,
            upper_inertia_kg_m2=upper_mass * upper_length**2 / 12.0,
            forearm_inertia_kg_m2=forearm_mass * forearm_length**2 / 12.0,
            club_mass_kg=0.45,
            club_inertia_kg_m2=0.055,
            right_grip_offset_m=0.065,
            left_grip_offset_m=-0.065,
        )

    def consistent_configuration(
        self,
        club_center: np.ndarray,
        club_angle_rad: float,
        right_elbow_branch: int = 1,
        left_elbow_branch: int = -1,
    ) -> np.ndarray:
        """Construct a constraint-consistent configuration using planar IK."""
        center = _vector2("club_center", club_center)
        if not np.isfinite(club_angle_rad):
            raise ValueError("club_angle_rad must be finite")
        direction = _segment_direction(club_angle_rad)
        right_target = center + self.right_grip_offset_m * direction
        left_target = center + self.left_grip_offset_m * direction
        right_angles = _inverse_arm(
            np.asarray(self.right_shoulder_m),
            right_target,
            self.upper_length_m,
            self.forearm_length_m,
            right_elbow_branch,
        )
        left_angles = _inverse_arm(
            np.asarray(self.left_shoulder_m),
            left_target,
            self.upper_length_m,
            self.forearm_length_m,
            left_elbow_branch,
        )
        return np.array(
            [
                right_angles[0],
                right_angles[1],
                left_angles[0],
                left_angles[1],
                center[0],
                center[1],
                club_angle_rad,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class TwoArmControl:
    """Applied joint torques; wrist torques act equally and oppositely."""

    right_shoulder_nm: float = 0.0
    right_elbow_nm: float = 0.0
    right_wrist_nm: float = 0.0
    left_shoulder_nm: float = 0.0
    left_elbow_nm: float = 0.0
    left_wrist_nm: float = 0.0

    def __post_init__(self) -> None:
        values = np.asarray(list(self.__dict__.values()), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("all control torques must be finite")

    @classmethod
    def zero(cls) -> TwoArmControl:
        """Return the pointwise zero-applied-control condition."""
        return cls()


@dataclass(frozen=True)
class DynamicsSolution:
    """One same-state constrained dynamics solution."""

    qddot: np.ndarray
    multipliers_n: np.ndarray
    contact_force_on_club_n: np.ndarray
    constraint_rank: int
    kkt_residual_norm: float
    acceleration_constraint_residual_norm: float


@dataclass(frozen=True)
class DriftControlSplit:
    """Pointwise total, drift, and linear control-response solutions."""

    total: DynamicsSolution
    drift: DynamicsSolution
    control: DynamicsSolution


@dataclass(frozen=True)
class ContactModes:
    """Resultant/common and differential/internal hand-force coordinates."""

    resultant_n: np.ndarray
    differential_n: np.ndarray


@dataclass(frozen=True)
class ContactWrench:
    """Net club wrench and contact power about a declared club center."""

    resultant_force_n: np.ndarray
    moment_about_center_nm: float
    contact_power_w: float


def _vector2(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape (2,) with finite values")
    return array


def _state_vector(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (N_COORDINATES,) or not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} must have shape ({N_COORDINATES},) with finite values"
        )
    return array


def _segment_direction(angle_rad: float) -> np.ndarray:
    return np.array([np.sin(angle_rad), -np.cos(angle_rad)])


def _segment_derivative(angle_rad: float) -> np.ndarray:
    return np.array([np.cos(angle_rad), np.sin(angle_rad)])


def _inverse_arm(
    shoulder: np.ndarray,
    target: np.ndarray,
    upper_length: float,
    forearm_length: float,
    branch: int,
) -> tuple[float, float]:
    if branch not in (-1, 1):
        raise ValueError("elbow branch must be -1 or 1")
    delta = target - shoulder
    radius_squared = float(delta @ delta)
    cosine = (radius_squared - upper_length**2 - forearm_length**2) / (
        2.0 * upper_length * forearm_length
    )
    if cosine < -1.0 - 1e-12 or cosine > 1.0 + 1e-12:
        raise ValueError("target is outside the two-link arm workspace")
    elbow = branch * float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    target_angle = float(np.arctan2(delta[0], -delta[1]))
    offset = float(
        np.arctan2(
            forearm_length * np.sin(elbow),
            upper_length + forearm_length * np.cos(elbow),
        )
    )
    return target_angle - offset, elbow


def kinematics(q: np.ndarray, params: TwoArmParams) -> dict[str, np.ndarray]:
    """Return shoulders, elbows, hands, club center, and grip points."""
    state = _state_vector("q", q)
    right_shoulder = np.asarray(params.right_shoulder_m, dtype=float)
    left_shoulder = np.asarray(params.left_shoulder_m, dtype=float)
    right_elbow = right_shoulder + params.upper_length_m * _segment_direction(state[0])
    right_hand = right_elbow + params.forearm_length_m * _segment_direction(
        state[0] + state[1]
    )
    left_elbow = left_shoulder + params.upper_length_m * _segment_direction(state[2])
    left_hand = left_elbow + params.forearm_length_m * _segment_direction(
        state[2] + state[3]
    )
    center = state[4:6]
    club_direction = _segment_direction(state[6])
    return {
        "right_shoulder": right_shoulder,
        "right_elbow": right_elbow,
        "right_hand": right_hand,
        "left_shoulder": left_shoulder,
        "left_elbow": left_elbow,
        "left_hand": left_hand,
        "club_center": center.copy(),
        "right_grip": center + params.right_grip_offset_m * club_direction,
        "left_grip": center + params.left_grip_offset_m * club_direction,
    }


def constraint_vector(q: np.ndarray, params: TwoArmParams) -> np.ndarray:
    """Return the four independent hand-to-club position constraints."""
    points = kinematics(q, params)
    return np.concatenate(
        (
            points["right_hand"] - points["right_grip"],
            points["left_hand"] - points["left_grip"],
        )
    )


def constraint_jacobian(q: np.ndarray, params: TwoArmParams) -> np.ndarray:
    """Return the analytical 4-by-7 closure Jacobian."""
    state = _state_vector("q", q)
    jacobian = np.zeros((N_CONSTRAINTS, N_COORDINATES))
    right_fore = _segment_derivative(state[0] + state[1])
    jacobian[0:2, 0] = (
        params.upper_length_m * _segment_derivative(state[0])
        + params.forearm_length_m * right_fore
    )
    jacobian[0:2, 1] = params.forearm_length_m * right_fore
    left_fore = _segment_derivative(state[2] + state[3])
    jacobian[2:4, 2] = (
        params.upper_length_m * _segment_derivative(state[2])
        + params.forearm_length_m * left_fore
    )
    jacobian[2:4, 3] = params.forearm_length_m * left_fore
    jacobian[0:2, 4:6] = -np.eye(2)
    jacobian[2:4, 4:6] = -np.eye(2)
    club_derivative = _segment_derivative(state[6])
    jacobian[0:2, 6] = -params.right_grip_offset_m * club_derivative
    jacobian[2:4, 6] = -params.left_grip_offset_m * club_derivative
    return jacobian


def mass_matrix(q: np.ndarray, params: TwoArmParams) -> np.ndarray:
    """Return the symmetric generalized mass matrix from COM Jacobians."""
    state = _state_vector("q", q)
    matrix = np.zeros((N_COORDINATES, N_COORDINATES))
    _add_arm_mass(matrix, state[0], state[1], 0, 1, params)
    _add_arm_mass(matrix, state[2], state[3], 2, 3, params)
    matrix[4, 4] += params.club_mass_kg
    matrix[5, 5] += params.club_mass_kg
    matrix[6, 6] += params.club_inertia_kg_m2
    return matrix


def _add_arm_mass(
    matrix: np.ndarray,
    shoulder_angle: float,
    elbow_angle: float,
    shoulder_index: int,
    elbow_index: int,
    params: TwoArmParams,
) -> None:
    upper_jacobian = np.zeros((2, N_COORDINATES))
    upper_jacobian[:, shoulder_index] = (
        0.5 * params.upper_length_m * _segment_derivative(shoulder_angle)
    )
    forearm_jacobian = np.zeros((2, N_COORDINATES))
    forearm_jacobian[:, shoulder_index] = params.upper_length_m * _segment_derivative(
        shoulder_angle
    ) + 0.5 * params.forearm_length_m * _segment_derivative(
        shoulder_angle + elbow_angle
    )
    forearm_jacobian[:, elbow_index] = (
        0.5
        * params.forearm_length_m
        * _segment_derivative(shoulder_angle + elbow_angle)
    )
    upper_angular = np.zeros(N_COORDINATES)
    upper_angular[shoulder_index] = 1.0
    forearm_angular = np.zeros(N_COORDINATES)
    forearm_angular[[shoulder_index, elbow_index]] = 1.0
    matrix += params.upper_mass_kg * upper_jacobian.T @ upper_jacobian
    matrix += params.upper_inertia_kg_m2 * np.outer(upper_angular, upper_angular)
    matrix += params.forearm_mass_kg * forearm_jacobian.T @ forearm_jacobian
    matrix += params.forearm_inertia_kg_m2 * np.outer(forearm_angular, forearm_angular)


def gravity_vector(q: np.ndarray, params: TwoArmParams) -> np.ndarray:
    """Return the generalized gradient of gravitational potential energy."""
    state = _state_vector("q", q)
    result = np.zeros(N_COORDINATES)
    for shoulder_index, elbow_index in ((0, 1), (2, 3)):
        shoulder = state[shoulder_index]
        forearm = shoulder + state[elbow_index]
        result[shoulder_index] = params.gravity_m_s2 * (
            0.5 * params.upper_mass_kg * params.upper_length_m * np.sin(shoulder)
            + params.forearm_mass_kg
            * (
                params.upper_length_m * np.sin(shoulder)
                + 0.5 * params.forearm_length_m * np.sin(forearm)
            )
        )
        result[elbow_index] = (
            0.5
            * params.forearm_mass_kg
            * params.gravity_m_s2
            * params.forearm_length_m
            * np.sin(forearm)
        )
    result[5] = params.club_mass_kg * params.gravity_m_s2
    return result


def coriolis_vector(
    q: np.ndarray, qdot: np.ndarray, params: TwoArmParams
) -> np.ndarray:
    """Return the exact velocity-bias vector for the two independent arms."""
    state = _state_vector("q", q)
    velocity = _state_vector("qdot", qdot)
    result = np.zeros(N_COORDINATES)
    coupling = (
        0.5 * params.forearm_mass_kg * params.upper_length_m * params.forearm_length_m
    )
    for shoulder_index, elbow_index in ((0, 1), (2, 3)):
        sine = np.sin(state[elbow_index])
        shoulder_speed = velocity[shoulder_index]
        elbow_speed = velocity[elbow_index]
        result[shoulder_index] = (
            -coupling * sine * (2.0 * shoulder_speed * elbow_speed + elbow_speed**2)
        )
        result[elbow_index] = coupling * sine * shoulder_speed**2
    return result


def control_generalized_force(control: TwoArmControl) -> np.ndarray:
    """Map joint torques to generalized force by virtual work.

    A wrist torque is positive on the club and therefore negative on both
    generalized coordinates contributing to the corresponding forearm angle.
    This makes the action/reaction pair explicit.
    """
    return np.array(
        [
            control.right_shoulder_nm - control.right_wrist_nm,
            control.right_elbow_nm - control.right_wrist_nm,
            control.left_shoulder_nm - control.left_wrist_nm,
            control.left_elbow_nm - control.left_wrist_nm,
            0.0,
            0.0,
            control.right_wrist_nm + control.left_wrist_nm,
        ]
    )


def _constraint_acceleration_bias(
    q: np.ndarray, qdot: np.ndarray, params: TwoArmParams
) -> np.ndarray:
    """Return the exact autonomous-constraint bias ``Jdot(q, qdot) qdot``."""
    state = _state_vector("q", q)
    velocity = _state_vector("qdot", qdot)
    result = np.zeros(N_CONSTRAINTS)
    club_rate = velocity[6]
    for row, shoulder_index, elbow_index, grip_offset in (
        (0, 0, 1, params.right_grip_offset_m),
        (2, 2, 3, params.left_grip_offset_m),
    ):
        shoulder = state[shoulder_index]
        forearm = shoulder + state[elbow_index]
        forearm_rate = velocity[shoulder_index] + velocity[elbow_index]
        result[row : row + 2] = (
            -params.upper_length_m
            * velocity[shoulder_index] ** 2
            * _segment_direction(shoulder)
            - params.forearm_length_m * forearm_rate**2 * _segment_direction(forearm)
            + grip_offset * club_rate**2 * _segment_direction(state[6])
        )
    return result


def constraint_acceleration_bias_audit(
    q: np.ndarray,
    qdot: np.ndarray,
    params: TwoArmParams,
) -> float:
    """Compare the exact bias with a centered directional derivative of J."""
    state = _state_vector("q", q)
    velocity = _state_vector("qdot", qdot)
    speed = float(np.linalg.norm(velocity))
    if speed == 0.0:
        return 0.0
    # A five-point stencil permits a larger state perturbation, suppressing
    # cancellation at the reference trajectory's highest rates while retaining
    # fourth-order truncation accuracy. This remains independent of the exact
    # centripetal implementation used by the solver.
    step = 1e-3 / max(1.0, speed)
    directional = (
        -constraint_jacobian(state + 2.0 * step * velocity, params)
        + 8.0 * constraint_jacobian(state + step * velocity, params)
        - 8.0 * constraint_jacobian(state - step * velocity, params)
        + constraint_jacobian(state - 2.0 * step * velocity, params)
    ) / (12.0 * step)
    return float(
        np.linalg.norm(
            _constraint_acceleration_bias(state, velocity, params)
            - directional @ velocity
        )
    )


def solve_constrained_dynamics(
    q: np.ndarray,
    qdot: np.ndarray,
    control: TwoArmControl,
    params: TwoArmParams,
) -> DynamicsSolution:
    """Solve the full-rank KKT system and fail closed on invalid topology."""
    state = _state_vector("q", q)
    velocity = _state_vector("qdot", qdot)
    violation = float(np.linalg.norm(constraint_vector(state, params)))
    if violation > params.constraint_tolerance_m:
        raise ValueError(
            "configuration violates hand constraints: "
            f"{violation:.3e} m > {params.constraint_tolerance_m:.3e} m"
        )
    matrix = mass_matrix(state, params)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
    if minimum_eigenvalue <= params.rank_tolerance:
        raise ValueError("mass matrix is not positive definite")
    jacobian = constraint_jacobian(state, params)
    rank = int(np.linalg.matrix_rank(jacobian, tol=params.rank_tolerance))
    if rank != N_CONSTRAINTS:
        raise ValueError(
            "constraint Jacobian must have full row rank; "
            f"got {rank}, expected {N_CONSTRAINTS}"
        )
    bias = coriolis_vector(state, velocity, params) + gravity_vector(state, params)
    gamma = _constraint_acceleration_bias(state, velocity, params)
    generalized = control_generalized_force(control)
    kkt = np.block(
        [
            [matrix, -jacobian.T],
            [jacobian, np.zeros((N_CONSTRAINTS, N_CONSTRAINTS))],
        ]
    )
    rhs = np.concatenate((generalized - bias, -gamma))
    try:
        solution = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("KKT system is singular; no least-squares fallback") from exc
    residual = float(np.linalg.norm(kkt @ solution - rhs))
    acceleration_residual = float(
        np.linalg.norm(jacobian @ solution[:N_COORDINATES] + gamma)
    )
    if residual > params.kkt_tolerance or acceleration_residual > params.kkt_tolerance:
        raise RuntimeError(
            "constrained solution did not satisfy declared tolerances: "
            f"KKT={residual:.3e}, constraint={acceleration_residual:.3e}"
        )
    multipliers = solution[N_COORDINATES:]
    return DynamicsSolution(
        qddot=solution[:N_COORDINATES],
        multipliers_n=multipliers,
        contact_force_on_club_n=-multipliers.reshape(2, 2),
        constraint_rank=rank,
        kkt_residual_norm=residual,
        acceleration_constraint_residual_norm=acceleration_residual,
    )


def drift_control_attribution(
    q: np.ndarray,
    qdot: np.ndarray,
    control: TwoArmControl,
    params: TwoArmParams,
) -> DriftControlSplit:
    """Return a same-state pointwise drift/control split.

    This is not a subtraction between forward trajectories.  ``drift`` is the
    zero-applied-control KKT solution at the supplied state.  ``control`` is
    the exact linear response obtained by subtracting the two same-state
    solutions, including the change in constraint multipliers.
    """
    total = solve_constrained_dynamics(q, qdot, control, params)
    drift = solve_constrained_dynamics(q, qdot, TwoArmControl.zero(), params)
    qddot = total.qddot - drift.qddot
    multipliers = total.multipliers_n - drift.multipliers_n
    jacobian = constraint_jacobian(q, params)
    control_solution = DynamicsSolution(
        qddot=qddot,
        multipliers_n=multipliers,
        contact_force_on_club_n=-multipliers.reshape(2, 2),
        constraint_rank=total.constraint_rank,
        kkt_residual_norm=float(
            np.linalg.norm(
                mass_matrix(q, params) @ qddot
                - jacobian.T @ multipliers
                - control_generalized_force(control)
            )
        ),
        acceleration_constraint_residual_norm=float(np.linalg.norm(jacobian @ qddot)),
    )
    return DriftControlSplit(total=total, drift=drift, control=control_solution)


def _arm_joint_forces(
    q: np.ndarray,
    qdot: np.ndarray,
    solution: DynamicsSolution,
    params: TwoArmParams,
    *,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if side == "right":
        shoulder_index, elbow_index, contact_index = 0, 1, 0
    elif side == "left":
        shoulder_index, elbow_index, contact_index = 2, 3, 1
    else:
        raise ValueError("side must be 'right' or 'left'")
    shoulder_angle = q[shoulder_index]
    forearm_angle = shoulder_angle + q[elbow_index]
    shoulder_speed = qdot[shoulder_index]
    forearm_speed = shoulder_speed + qdot[elbow_index]
    shoulder_acceleration = solution.qddot[shoulder_index]
    forearm_acceleration = shoulder_acceleration + solution.qddot[elbow_index]

    upper_com_acceleration = (
        0.5
        * params.upper_length_m
        * (
            shoulder_acceleration * _segment_derivative(shoulder_angle)
            - shoulder_speed**2 * _segment_direction(shoulder_angle)
        )
    )
    elbow_acceleration = params.upper_length_m * (
        shoulder_acceleration * _segment_derivative(shoulder_angle)
        - shoulder_speed**2 * _segment_direction(shoulder_angle)
    )
    forearm_com_acceleration = elbow_acceleration + (
        0.5
        * params.forearm_length_m
        * (
            forearm_acceleration * _segment_derivative(forearm_angle)
            - forearm_speed**2 * _segment_direction(forearm_angle)
        )
    )
    gravity = np.array([0.0, -params.gravity_m_s2])
    hand_on_club = solution.contact_force_on_club_n[contact_index]
    elbow_on_forearm = (
        params.forearm_mass_kg * forearm_com_acceleration
        - params.forearm_mass_kg * gravity
        + hand_on_club
    )
    shoulder_on_upper = (
        params.upper_mass_kg * upper_com_acceleration
        - params.upper_mass_kg * gravity
        + elbow_on_forearm
    )
    return shoulder_on_upper, elbow_on_forearm, hand_on_club


def _trajectory_kinematics(
    q: np.ndarray, qdot: np.ndarray, params: TwoArmParams
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = q.shape[0]
    positions = np.empty((sample_count, 6, 2))
    velocities = np.zeros((sample_count, 6, 2))
    for sample_index, (configuration, velocity) in enumerate(zip(q, qdot, strict=True)):
        points = kinematics(configuration, params)
        positions[sample_index] = np.stack(
            (
                points["right_shoulder"],
                points["right_elbow"],
                points["right_hand"],
                points["left_shoulder"],
                points["left_elbow"],
                points["left_hand"],
            )
        )
        for output_offset, shoulder_index, elbow_index in (
            (0, 0, 1),
            (3, 2, 3),
        ):
            shoulder_angle = configuration[shoulder_index]
            forearm_angle = shoulder_angle + configuration[elbow_index]
            shoulder_speed = velocity[shoulder_index]
            forearm_speed = shoulder_speed + velocity[elbow_index]
            elbow_velocity = (
                params.upper_length_m
                * shoulder_speed
                * _segment_derivative(shoulder_angle)
            )
            hand_velocity = elbow_velocity + (
                params.forearm_length_m
                * forearm_speed
                * _segment_derivative(forearm_angle)
            )
            velocities[sample_index, output_offset + 1] = elbow_velocity
            velocities[sample_index, output_offset + 2] = hand_velocity
    return positions, velocities


def two_arm_joint_transfer_trajectory(
    time: np.ndarray,
    q: np.ndarray,
    qdot: np.ndarray,
    controls: tuple[TwoArmControl, ...],
    params: TwoArmParams,
    *,
    velocity_constraint_tolerance_m_s: float = 1e-7,
) -> JointTransferTrajectory:
    """Return all shoulder, elbow, and hand transfer quantities.

    Input states must satisfy both position and velocity closure.  This
    fail-closed boundary prevents a prescribed but mechanically inconsistent
    hand path from entering the publication evidence.
    """
    time_array = np.asarray(time, dtype=float).reshape(-1)
    q_array = np.asarray(q, dtype=float)
    velocity_array = np.asarray(qdot, dtype=float)
    expected = (time_array.size, N_COORDINATES)
    if q_array.shape != expected or velocity_array.shape != expected:
        raise ValueError(f"q and qdot must both have shape {expected}")
    if len(controls) != time_array.size:
        raise ValueError("controls must contain one TwoArmControl per sample")
    if time_array.size < 2 or np.any(np.diff(time_array) <= 0.0):
        raise ValueError("time must contain at least two strictly increasing samples")
    if not all(
        np.all(np.isfinite(value)) for value in (time_array, q_array, velocity_array)
    ):
        raise ValueError("trajectory arrays must contain only finite values")
    if (
        not np.isfinite(velocity_constraint_tolerance_m_s)
        or velocity_constraint_tolerance_m_s <= 0.0
    ):
        raise ValueError("velocity constraint tolerance must be finite and positive")

    total_force = np.empty((time_array.size, 6, 2))
    drift_force = np.empty_like(total_force)
    control_force = np.empty_like(total_force)
    for sample_index, (configuration, velocity, control) in enumerate(
        zip(q_array, velocity_array, controls, strict=True)
    ):
        velocity_residual = float(
            np.linalg.norm(constraint_jacobian(configuration, params) @ velocity)
        )
        if velocity_residual > velocity_constraint_tolerance_m_s:
            raise ValueError(
                f"state violates hand velocity constraints: {velocity_residual:.3e} m/s"
            )
        split = drift_control_attribution(configuration, velocity, control, params)
        for target, solution in (
            (total_force, split.total),
            (drift_force, split.drift),
        ):
            right = _arm_joint_forces(
                configuration, velocity, solution, params, side="right"
            )
            left = _arm_joint_forces(
                configuration, velocity, solution, params, side="left"
            )
            target[sample_index] = np.stack((*right, *left))
        control_force[sample_index] = (
            total_force[sample_index] - drift_force[sample_index]
        )

    positions, joint_velocities = _trajectory_kinematics(
        q_array, velocity_array, params
    )
    couple_control = np.array(
        [
            [
                control.right_shoulder_nm,
                control.right_elbow_nm,
                control.right_wrist_nm,
                control.left_shoulder_nm,
                control.left_elbow_nm,
                control.left_wrist_nm,
            ]
            for control in controls
        ]
    )
    couple_drift = np.zeros_like(couple_control)
    angular_velocity = np.column_stack(
        (
            velocity_array[:, 0],
            velocity_array[:, 0] + velocity_array[:, 1],
            velocity_array[:, 6],
            velocity_array[:, 2],
            velocity_array[:, 2] + velocity_array[:, 3],
            velocity_array[:, 6],
        )
    )
    return JointTransferTrajectory(
        time=time_array,
        joint_names=(
            "right_shoulder",
            "right_elbow",
            "right_hand",
            "left_shoulder",
            "left_elbow",
            "left_hand",
        ),
        position=positions,
        velocity=joint_velocities,
        force_total=total_force,
        force_drift=drift_force,
        force_control=control_force,
        couple_total=couple_control,
        couple_drift=couple_drift,
        couple_control=couple_control,
        angular_velocity=angular_velocity,
        model_tier="two_arm_floating_club_closed_loop",
        force_direction="proximal_on_distal",
        frame="planar_cartesian_x_target_y_up",
        reference_point="shoulder_elbow_and_hand_joint_centers",
        units="SI",
    )


def decompose_contact_forces(
    right_force_n: np.ndarray, left_force_n: np.ndarray
) -> ContactModes:
    """Return resultant/common and differential/internal hand-force modes."""
    right = _vector2("right_force_n", right_force_n)
    left = _vector2("left_force_n", left_force_n)
    return ContactModes(
        resultant_n=right + left,
        differential_n=0.5 * (right - left),
    )


def contact_wrench(
    right_force_n: np.ndarray,
    left_force_n: np.ndarray,
    right_point_m: np.ndarray,
    left_point_m: np.ndarray,
    club_center_m: np.ndarray,
    right_velocity_m_s: np.ndarray,
    left_velocity_m_s: np.ndarray,
) -> ContactWrench:
    """Return net contact wrench and two-point power about ``club_center_m``."""
    right_force = _vector2("right_force_n", right_force_n)
    left_force = _vector2("left_force_n", left_force_n)
    right_point = _vector2("right_point_m", right_point_m)
    left_point = _vector2("left_point_m", left_point_m)
    center = _vector2("club_center_m", club_center_m)
    right_velocity = _vector2("right_velocity_m_s", right_velocity_m_s)
    left_velocity = _vector2("left_velocity_m_s", left_velocity_m_s)
    right_arm = right_point - center
    left_arm = left_point - center
    moment = float(
        right_arm[0] * right_force[1]
        - right_arm[1] * right_force[0]
        + left_arm[0] * left_force[1]
        - left_arm[1] * left_force[0]
    )
    return ContactWrench(
        resultant_force_n=right_force + left_force,
        moment_about_center_nm=moment,
        contact_power_w=float(
            right_force @ right_velocity + left_force @ left_velocity
        ),
    )


__all__ = [
    "ContactModes",
    "ContactWrench",
    "DriftControlSplit",
    "DynamicsSolution",
    "TwoArmControl",
    "TwoArmParams",
    "constraint_jacobian",
    "constraint_vector",
    "contact_wrench",
    "control_generalized_force",
    "coriolis_vector",
    "decompose_contact_forces",
    "drift_control_attribution",
    "gravity_vector",
    "kinematics",
    "mass_matrix",
    "solve_constrained_dynamics",
    "two_arm_joint_transfer_trajectory",
]
