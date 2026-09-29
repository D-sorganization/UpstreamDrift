"""Forward two-hand dynamics with a moving base and distributed shaft modes.

The hand and base topology matches :mod:`moving_base_flexible_club`, while the
single torsional flex coordinate is replaced by mass-normalized bending modes
sampled from the canonical Euler--Bernoulli finite-element shaft.  Gaussian
quadrature reconstructs the shaft's distributed mass and its rigid/modal
coupling.  The declared modal properties are synthetic reference values, not
equipment calibration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache

import numpy as np
import numpy.typing as npt
from scipy.linalg import eigh

from scripts.research.proximal_distal_energy.moving_base_flexible_club import (
    MovingBaseFlexibleParams,
)
from scripts.research.proximal_distal_energy.shaft_beam_reference import (
    BeamReferenceConfig,
    modal_basis,
    model_matrices,
)
from scripts.research.proximal_distal_energy.two_arm_closed_loop import (
    TwoArmControl,
    TwoArmParams,
)

FloatArray = npt.NDArray[np.float64]
ControlLaw = Callable[[float, FloatArray, FloatArray], TwoArmControl]
N_RIGID_COORDINATES = 9
N_CONSTRAINTS = 4


def _direction(angle_rad: float) -> FloatArray:
    return np.array([np.sin(angle_rad), -np.cos(angle_rad)])


def _normal(angle_rad: float) -> FloatArray:
    return np.array([np.cos(angle_rad), np.sin(angle_rad)])


def _cross_z(offset: FloatArray, force: FloatArray) -> float:
    return float(offset[0] * force[1] - offset[1] * force[0])


@dataclass(frozen=True, slots=True)
class ModalShaftCouplingParams:
    """Mechanical, beam, quadrature, damping, and model-boundary contract."""

    mechanism: MovingBaseFlexibleParams
    beam: BeamReferenceConfig
    mode_count: int
    quadrature_order: int = 4
    damping_ratio: float = 0.018

    @classmethod
    def publication_default(cls, *, mode_count: int = 6) -> ModalShaftCouplingParams:
        """Return the declared synthetic modal coupling case."""
        mechanism = replace(
            MovingBaseFlexibleParams.publication_default(),
            proximal_club_length_m=0.46,
            distal_club_length_m=0.54,
        )
        return cls(
            mechanism=mechanism,
            beam=BeamReferenceConfig.publication_default(),
            mode_count=mode_count,
        )

    def __post_init__(self) -> None:
        if self.mode_count < 1 or self.mode_count > self.beam.mode_count:
            raise ValueError("mode_count must be within the declared beam basis")
        if self.quadrature_order < 2 or self.quadrature_order > 12:
            raise ValueError("quadrature_order must be between 2 and 12")
        if not np.isfinite(self.damping_ratio) or not 0.0 <= self.damping_ratio < 1:
            raise ValueError("damping_ratio must be finite and in [0, 1)")
        if not np.isclose(
            self.beam.length_m,
            self.mechanism.proximal_club_length_m + self.mechanism.distal_club_length_m,
        ):
            raise ValueError("beam length must equal the declared club length")

    @property
    def coordinate_count(self) -> int:
        return N_RIGID_COORDINATES + self.mode_count


@dataclass(frozen=True, slots=True)
class ModalShaftCouplingConfig:
    """Forward integration and projection contract."""

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
class ModalShaftBasis:
    """Quadrature realization of the finite-element bending basis."""

    locations_m: FloatArray
    masses_kg: FloatArray
    mode_shapes: FloatArray
    mode_slopes: FloatArray
    fe_frequencies_hz: FloatArray
    coupled_frequencies_hz: FloatArray
    modal_mass: FloatArray
    modal_stiffness: FloatArray
    maximum_frequency_discrepancy_relative: float
    calibration_status: str = "synthetic_reference_not_equipment_calibrated"


@dataclass(frozen=True, slots=True)
class DynamicsSolution:
    """One full-rank constrained acceleration solve."""

    qddot: FloatArray
    multipliers_n: FloatArray
    contact_force_on_club_n: FloatArray
    constraint_rank: int
    kkt_residual_norm: float
    acceleration_constraint_residual_norm: float


@dataclass(frozen=True, slots=True)
class ModalShaftTrace:
    """Trajectory, modal response, interactions, and closure evidence."""

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
    modal_coordinates: FloatArray
    modal_velocities: FloatArray
    modal_tip_deflection_m: FloatArray
    shaft_strain_energy_j: FloatArray
    shaft_damping_power_w: FloatArray
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
    model_tier: str = "moving_base_two_hand_distributed_modal_shaft"


def _state(name: str, value: object, params: ModalShaftCouplingParams) -> FloatArray:
    array = np.asarray(value, dtype=float)
    expected = params.coordinate_count
    if array.shape != (expected,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape ({expected},) with finite values")
    return array.copy()


def _node_dofs(vector: FloatArray, node: int) -> tuple[FloatArray, FloatArray]:
    if node == 0:
        zeros = np.zeros(vector.shape[1])
        return zeros, zeros
    start = 2 * (node - 1)
    return vector[start], vector[start + 1]


def _hermite_shapes(xi: float, length_m: float) -> tuple[FloatArray, FloatArray]:
    values = np.array(
        [
            1.0 - 3.0 * xi**2 + 2.0 * xi**3,
            length_m * (xi - 2.0 * xi**2 + xi**3),
            3.0 * xi**2 - 2.0 * xi**3,
            length_m * (-(xi**2) + xi**3),
        ]
    )
    slopes = np.array(
        [
            (-6.0 * xi + 6.0 * xi**2) / length_m,
            1.0 - 4.0 * xi + 3.0 * xi**2,
            (6.0 * xi - 6.0 * xi**2) / length_m,
            -2.0 * xi + 3.0 * xi**2,
        ]
    )
    return values, slopes


def _area_at(location_m: float, config: BeamReferenceConfig) -> float:
    fraction = location_m / config.length_m
    outer = config.butt_diameter_m + fraction * (
        config.tip_diameter_m - config.butt_diameter_m
    )
    inner = outer - 2.0 * config.wall_thickness_m
    return float(0.25 * np.pi * (outer**2 - inner**2))


@cache
def modal_shaft_basis(params: ModalShaftCouplingParams) -> ModalShaftBasis:
    """Build a mass-normalized quadrature basis from the canonical FE model."""
    mass, stiffness = model_matrices(params.beam)
    fe_frequencies, raw_vectors = modal_basis(mass, stiffness, params.mode_count)
    nodes, weights = np.polynomial.legendre.leggauss(params.quadrature_order)
    element_length = params.beam.length_m / params.beam.element_count
    locations: list[float] = []
    masses: list[float] = []
    raw_shapes: list[FloatArray] = []
    raw_slopes: list[FloatArray] = []
    for element in range(params.beam.element_count):
        left_w, left_theta = _node_dofs(raw_vectors, element)
        right_w, right_theta = _node_dofs(raw_vectors, element + 1)
        dofs = np.vstack((left_w, left_theta, right_w, right_theta))
        for node, weight in zip(nodes, weights, strict=True):
            xi = 0.5 * (float(node) + 1.0)
            location = (element + xi) * element_length
            shape, slope = _hermite_shapes(xi, element_length)
            dm = (
                params.beam.density_kg_m3
                * _area_at(location, params.beam)
                * 0.5
                * element_length
                * float(weight)
            )
            locations.append(location)
            masses.append(dm)
            raw_shapes.append(shape @ dofs)
            raw_slopes.append(slope @ dofs)
    tip_w, tip_theta = _node_dofs(raw_vectors, params.beam.element_count)
    locations.append(params.beam.length_m)
    masses.append(params.beam.head_mass_kg)
    raw_shapes.append(tip_w)
    raw_slopes.append(tip_theta)
    location_array = np.asarray(locations)
    mass_array = np.asarray(masses)
    shape_array = np.asarray(raw_shapes)
    slope_array = np.asarray(raw_slopes)
    gram = shape_array.T @ (mass_array[:, None] * shape_array)
    gram += params.beam.head_rotary_inertia_kg_m2 * np.outer(tip_theta, tip_theta)
    raw_stiffness = raw_vectors.T @ stiffness @ raw_vectors
    eigenvalues, transform = eigh(raw_stiffness, gram)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("quadrature modal eigenproblem returned a nonpositive mode")
    shapes = shape_array @ transform
    slopes = slope_array @ transform
    modal_mass = shapes.T @ (mass_array[:, None] * shapes)
    modal_mass += params.beam.head_rotary_inertia_kg_m2 * np.outer(
        slopes[-1], slopes[-1]
    )
    modal_stiffness = transform.T @ raw_stiffness @ transform
    coupled = np.sqrt(eigenvalues) / (2.0 * np.pi)
    relative = np.abs(coupled - fe_frequencies) / fe_frequencies
    return ModalShaftBasis(
        locations_m=location_array,
        masses_kg=mass_array,
        mode_shapes=shapes,
        mode_slopes=slopes,
        fe_frequencies_hz=fe_frequencies,
        coupled_frequencies_hz=coupled,
        modal_mass=modal_mass,
        modal_stiffness=modal_stiffness,
        maximum_frequency_discrepancy_relative=float(np.max(relative)),
    )


def initial_state(
    params: ModalShaftCouplingParams,
    *,
    grip_center_m: npt.ArrayLike = (0.0, -0.5),
    club_angle_rad: float = 0.16,
) -> tuple[FloatArray, FloatArray]:
    """Return an exactly closed, undeformed modal-shaft state."""
    mechanism = params.mechanism
    old = TwoArmParams(
        right_shoulder_m=mechanism.right_shoulder_offset_m,
        left_shoulder_m=mechanism.left_shoulder_offset_m,
        upper_length_m=mechanism.upper_length_m,
        forearm_length_m=mechanism.forearm_length_m,
        upper_mass_kg=mechanism.upper_mass_kg,
        forearm_mass_kg=mechanism.forearm_mass_kg,
        upper_inertia_kg_m2=mechanism.upper_inertia_kg_m2,
        forearm_inertia_kg_m2=mechanism.forearm_inertia_kg_m2,
        club_mass_kg=params.beam.head_mass_kg,
        club_inertia_kg_m2=params.beam.head_rotary_inertia_kg_m2,
        right_grip_offset_m=mechanism.right_grip_offset_m,
        left_grip_offset_m=mechanism.left_grip_offset_m,
        gravity_m_s2=mechanism.gravity_m_s2,
    )
    old_q = old.consistent_configuration(
        np.asarray(grip_center_m, dtype=float), club_angle_rad
    )
    q = np.zeros(params.coordinate_count)
    q[:4] = old_q[:4]
    q[6:8] = old_q[4:6]
    q[8] = old_q[6]
    return q, np.zeros_like(q)


def _arm_bodies(
    q: FloatArray, params: ModalShaftCouplingParams
) -> list[tuple[float, FloatArray, FloatArray]]:
    mechanism = params.mechanism
    count = params.coordinate_count
    base_jacobian = np.zeros((2, count))
    base_jacobian[:, 4:6] = np.eye(2)
    bodies: list[tuple[float, FloatArray, FloatArray]] = [
        (mechanism.base_mass_kg, base_jacobian, np.zeros(2))
    ]
    for shoulder_index, elbow_index in ((0, 1), (2, 3)):
        shoulder = q[shoulder_index]
        forearm = shoulder + q[elbow_index]
        upper = np.zeros((2, count))
        upper[:, 4:6] = np.eye(2)
        upper[:, shoulder_index] = 0.5 * mechanism.upper_length_m * _normal(shoulder)
        bodies.append(
            (
                mechanism.upper_mass_kg,
                upper,
                -0.5 * mechanism.upper_length_m * _direction(shoulder),
            )
        )
        fore = np.zeros((2, count))
        fore[:, 4:6] = np.eye(2)
        fore[:, shoulder_index] = mechanism.upper_length_m * _normal(
            shoulder
        ) + 0.5 * mechanism.forearm_length_m * _normal(forearm)
        fore[:, elbow_index] = 0.5 * mechanism.forearm_length_m * _normal(forearm)
        coefficients = np.column_stack(
            (
                -mechanism.upper_length_m * _direction(shoulder),
                -0.5 * mechanism.forearm_length_m * _direction(forearm),
            )
        )
        bodies.append((mechanism.forearm_mass_kg, fore, coefficients))
    return bodies


def _shaft_point_jacobians(
    q: FloatArray, params: ModalShaftCouplingParams
) -> list[tuple[float, FloatArray, FloatArray]]:
    basis = modal_shaft_basis(params)
    count = params.coordinate_count
    alpha = q[8]
    eta = q[N_RIGID_COORDINATES:]
    direction = _direction(alpha)
    normal = _normal(alpha)
    bodies: list[tuple[float, FloatArray, FloatArray]] = []
    for location, mass, shape in zip(
        basis.locations_m, basis.masses_kg, basis.mode_shapes, strict=True
    ):
        deflection = float(shape @ eta)
        jacobian = np.zeros((2, count))
        jacobian[:, 6:8] = np.eye(2)
        jacobian[:, 8] = location * normal - deflection * direction
        jacobian[:, N_RIGID_COORDINATES:] = np.outer(normal, shape)
        bodies.append((float(mass), jacobian, shape))
    return bodies


def mass_matrix(q: object, params: ModalShaftCouplingParams) -> FloatArray:
    """Return the distributed shaft and body mass matrix."""
    state = _state("q", q, params)
    matrix = np.zeros((params.coordinate_count, params.coordinate_count))
    for mass, jacobian, _ in _arm_bodies(state, params):
        matrix += mass * jacobian.T @ jacobian
    for mass, jacobian, _ in _shaft_point_jacobians(state, params):
        matrix += mass * jacobian.T @ jacobian
    mechanism = params.mechanism
    for shoulder_index, elbow_index in ((0, 1), (2, 3)):
        upper = np.zeros(params.coordinate_count)
        upper[shoulder_index] = 1.0
        forearm = upper.copy()
        forearm[elbow_index] = 1.0
        matrix += mechanism.upper_inertia_kg_m2 * np.outer(upper, upper)
        matrix += mechanism.forearm_inertia_kg_m2 * np.outer(forearm, forearm)
    head_angular = np.zeros(params.coordinate_count)
    head_angular[8] = 1.0
    head_angular[N_RIGID_COORDINATES:] = modal_shaft_basis(params).mode_slopes[-1]
    matrix += params.beam.head_rotary_inertia_kg_m2 * np.outer(
        head_angular, head_angular
    )
    return matrix


def _velocity_bias(
    q: FloatArray, qdot: FloatArray, params: ModalShaftCouplingParams
) -> FloatArray:
    result = np.zeros(params.coordinate_count)
    bodies = _arm_bodies(q, params)
    body_index = 1
    for shoulder_index, elbow_index in ((0, 1), (2, 3)):
        mass, jacobian, coefficient = bodies[body_index]
        result += mass * jacobian.T @ (coefficient * qdot[shoulder_index] ** 2)
        body_index += 1
        mass, jacobian, coefficients = bodies[body_index]
        rates = np.array(
            [qdot[shoulder_index], qdot[shoulder_index] + qdot[elbow_index]]
        )
        result += mass * jacobian.T @ (coefficients @ rates**2)
        body_index += 1
    alpha = q[8]
    alpha_rate = qdot[8]
    eta = q[N_RIGID_COORDINATES:]
    eta_rate = qdot[N_RIGID_COORDINATES:]
    direction = _direction(alpha)
    normal = _normal(alpha)
    basis = modal_shaft_basis(params)
    for (mass, jacobian, _), location, shape in zip(
        _shaft_point_jacobians(q, params),
        basis.locations_m,
        basis.mode_shapes,
        strict=True,
    ):
        deflection = float(shape @ eta)
        deflection_rate = float(shape @ eta_rate)
        acceleration = (
            -(location * direction + deflection * normal) * alpha_rate**2
            - 2.0 * deflection_rate * direction * alpha_rate
        )
        result += mass * jacobian.T @ acceleration
    return result


def _potential_gradient(q: FloatArray, params: ModalShaftCouplingParams) -> FloatArray:
    mechanism = params.mechanism
    gradient = np.zeros(params.coordinate_count)
    for mass, jacobian, _ in _arm_bodies(q, params):
        gradient += mass * mechanism.gravity_m_s2 * jacobian[1]
    for mass, jacobian, _ in _shaft_point_jacobians(q, params):
        gradient += mass * mechanism.gravity_m_s2 * jacobian[1]
    gradient[4:6] += mechanism.base_stiffness_n_m * q[4:6]
    eta = q[N_RIGID_COORDINATES:]
    gradient[N_RIGID_COORDINATES:] += modal_shaft_basis(params).modal_stiffness @ eta
    return gradient


def _damping_force(qdot: FloatArray, params: ModalShaftCouplingParams) -> FloatArray:
    mechanism = params.mechanism
    force = np.zeros(params.coordinate_count)
    force[:4] = -mechanism.joint_damping_nms_rad * qdot[:4]
    force[4:6] = -mechanism.base_damping_ns_m * qdot[4:6]
    omega = 2.0 * np.pi * modal_shaft_basis(params).coupled_frequencies_hz
    force[N_RIGID_COORDINATES:] = (
        -2.0 * params.damping_ratio * omega * qdot[N_RIGID_COORDINATES:]
    )
    return force


def kinematics(q: object, params: ModalShaftCouplingParams) -> dict[str, FloatArray]:
    """Return arm, grip, shaft-root, and deformed clubhead positions."""
    state = _state("q", q, params)
    mechanism = params.mechanism
    points: dict[str, FloatArray] = {"base": state[4:6].copy()}
    base = state[4:6]
    for side, shoulder_index, elbow_index, offset in (
        ("right", 0, 1, mechanism.right_shoulder_offset_m),
        ("left", 2, 3, mechanism.left_shoulder_offset_m),
    ):
        shoulder = base + np.asarray(offset)
        elbow = shoulder + mechanism.upper_length_m * _direction(state[shoulder_index])
        hand = elbow + mechanism.forearm_length_m * _direction(
            state[shoulder_index] + state[elbow_index]
        )
        points[f"{side}_shoulder"] = shoulder
        points[f"{side}_elbow"] = elbow
        points[f"{side}_hand"] = hand
    center = state[6:8]
    alpha = state[8]
    basis = modal_shaft_basis(params)
    tip_deflection = float(basis.mode_shapes[-1] @ state[N_RIGID_COORDINATES:])
    points["grip_center"] = center.copy()
    points["right_grip"] = center + mechanism.right_grip_offset_m * _direction(alpha)
    points["left_grip"] = center + mechanism.left_grip_offset_m * _direction(alpha)
    points["clubhead"] = (
        center
        + params.beam.length_m * _direction(alpha)
        + tip_deflection * _normal(alpha)
    )
    return points


def constraint_vector(q: object, params: ModalShaftCouplingParams) -> FloatArray:
    """Return both hand-minus-grip closure residuals."""
    points = kinematics(q, params)
    return np.concatenate(
        (
            points["right_hand"] - points["right_grip"],
            points["left_hand"] - points["left_grip"],
        )
    )


def constraint_jacobian(q: object, params: ModalShaftCouplingParams) -> FloatArray:
    """Return the exact hand-closure Jacobian."""
    state = _state("q", q, params)
    mechanism = params.mechanism
    jacobian = np.zeros((N_CONSTRAINTS, params.coordinate_count))
    for row, shoulder_index, elbow_index, grip_offset in (
        (0, 0, 1, mechanism.right_grip_offset_m),
        (2, 2, 3, mechanism.left_grip_offset_m),
    ):
        shoulder = state[shoulder_index]
        forearm = shoulder + state[elbow_index]
        jacobian[row : row + 2, shoulder_index] = mechanism.upper_length_m * _normal(
            shoulder
        ) + mechanism.forearm_length_m * _normal(forearm)
        jacobian[row : row + 2, elbow_index] = mechanism.forearm_length_m * _normal(
            forearm
        )
        jacobian[row : row + 2, 4:6] = np.eye(2)
        jacobian[row : row + 2, 6:8] = -np.eye(2)
        jacobian[row : row + 2, 8] = -grip_offset * _normal(state[8])
    return jacobian


def _constraint_acceleration_bias(
    q: FloatArray, qdot: FloatArray, params: ModalShaftCouplingParams
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


def control_generalized_force(
    control: TwoArmControl, params: ModalShaftCouplingParams
) -> FloatArray:
    """Map joint controls with explicit wrist action and reaction."""
    force = np.zeros(params.coordinate_count)
    force[:4] = (
        control.right_shoulder_nm - control.right_wrist_nm,
        control.right_elbow_nm - control.right_wrist_nm,
        control.left_shoulder_nm - control.left_wrist_nm,
        control.left_elbow_nm - control.left_wrist_nm,
    )
    force[8] = control.right_wrist_nm + control.left_wrist_nm
    return force


def solve_constrained_dynamics(
    q: object,
    qdot: object,
    control: TwoArmControl,
    params: ModalShaftCouplingParams,
) -> DynamicsSolution:
    """Solve the coupled KKT system and fail closed on invalid topology."""
    state = _state("q", q, params)
    velocity = _state("qdot", qdot, params)
    mechanism = params.mechanism
    violation = float(np.linalg.norm(constraint_vector(state, params)))
    if violation > mechanism.constraint_tolerance_m:
        raise ValueError(
            "configuration violates hand constraints: "
            f"{violation:.3e} m > {mechanism.constraint_tolerance_m:.3e} m"
        )
    matrix = mass_matrix(state, params)
    if float(np.min(np.linalg.eigvalsh(matrix))) <= mechanism.rank_tolerance:
        raise ValueError("mass matrix is not positive definite")
    jacobian = constraint_jacobian(state, params)
    rank = int(np.linalg.matrix_rank(jacobian, tol=mechanism.rank_tolerance))
    if rank != N_CONSTRAINTS:
        raise ValueError(f"constraint Jacobian rank is {rank}; expected 4")
    bias = _velocity_bias(state, velocity, params) + _potential_gradient(state, params)
    generalized = control_generalized_force(control, params) + _damping_force(
        velocity, params
    )
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
    except np.linalg.LinAlgError as error:
        raise ValueError("KKT system is singular; no fallback is allowed") from error
    residual = float(np.linalg.norm(kkt @ result - rhs))
    acceleration_residual = float(
        np.linalg.norm(jacobian @ result[: params.coordinate_count] + gamma)
    )
    if (
        residual > mechanism.kkt_tolerance
        or acceleration_residual > mechanism.kkt_tolerance
    ):
        raise RuntimeError(
            "coupled constrained solve exceeded tolerance: "
            f"KKT={residual:.3e}, constraint={acceleration_residual:.3e}"
        )
    multipliers = result[params.coordinate_count :]
    return DynamicsSolution(
        qddot=result[: params.coordinate_count],
        multipliers_n=multipliers,
        contact_force_on_club_n=-multipliers.reshape(2, 2),
        constraint_rank=rank,
        kkt_residual_norm=residual,
        acceleration_constraint_residual_norm=acceleration_residual,
    )


def potential_energy(q: object, params: ModalShaftCouplingParams) -> float:
    """Return gravity, base, and modal strain potential energy."""
    state = _state("q", q, params)
    mechanism = params.mechanism
    gravity = mechanism.base_mass_kg * state[5]
    base = state[4:6]
    for shoulder_index, elbow_index in ((0, 1), (2, 3)):
        shoulder_y = state[5]
        upper_y = shoulder_y - 0.5 * mechanism.upper_length_m * np.cos(
            state[shoulder_index]
        )
        forearm_y = (
            shoulder_y
            - mechanism.upper_length_m * np.cos(state[shoulder_index])
            - 0.5
            * mechanism.forearm_length_m
            * np.cos(state[shoulder_index] + state[elbow_index])
        )
        gravity += mechanism.upper_mass_kg * upper_y
        gravity += mechanism.forearm_mass_kg * forearm_y
    center = state[6:8]
    alpha = state[8]
    eta = state[N_RIGID_COORDINATES:]
    direction = _direction(alpha)
    normal = _normal(alpha)
    basis = modal_shaft_basis(params)
    for location, mass, shape in zip(
        basis.locations_m, basis.masses_kg, basis.mode_shapes, strict=True
    ):
        point = center + location * direction + float(shape @ eta) * normal
        gravity += mass * point[1]
    strain = 0.5 * float(eta @ basis.modal_stiffness @ eta)
    return float(
        mechanism.gravity_m_s2 * gravity
        + 0.5 * mechanism.base_stiffness_n_m * float(base @ base)
        + strain
    )


def mechanical_energy(
    q: object, qdot: object, params: ModalShaftCouplingParams
) -> float:
    """Return kinetic plus conservative potential energy."""
    state = _state("q", q, params)
    velocity = _state("qdot", qdot, params)
    return 0.5 * float(
        velocity @ mass_matrix(state, params) @ velocity
    ) + potential_energy(state, params)


def _mass_metric_correction(
    q: FloatArray, residual: FloatArray, params: ModalShaftCouplingParams
) -> FloatArray:
    matrix = mass_matrix(q, params)
    jacobian = constraint_jacobian(q, params)
    inverse_jacobian = np.linalg.solve(matrix, jacobian.T)
    schur = jacobian @ inverse_jacobian
    if np.linalg.matrix_rank(schur, tol=params.mechanism.rank_tolerance) != 4:
        raise ValueError("constraint projection is singular; no fallback is allowed")
    return inverse_jacobian @ np.linalg.solve(schur, residual)


def _project_configuration(
    q: FloatArray,
    params: ModalShaftCouplingParams,
    config: ModalShaftCouplingConfig,
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
    params: ModalShaftCouplingParams,
    config: ModalShaftCouplingConfig,
) -> FloatArray:
    jacobian = constraint_jacobian(q, params)
    residual = jacobian @ qdot
    if np.linalg.norm(residual) <= config.velocity_tolerance_m_s:
        return qdot.copy()
    result = qdot - _mass_metric_correction(q, residual, params)
    if np.linalg.norm(jacobian @ result) > config.velocity_tolerance_m_s:
        raise ValueError("velocity projection failed to converge")
    return result


def _grip_velocity(q: FloatArray, qdot: FloatArray, offset_m: float) -> FloatArray:
    return qdot[6:8] + offset_m * qdot[8] * _normal(q[8])


def _clubhead_jacobian(q: FloatArray, params: ModalShaftCouplingParams) -> FloatArray:
    basis = modal_shaft_basis(params)
    eta = q[N_RIGID_COORDINATES:]
    tip_shape = basis.mode_shapes[-1]
    deflection = float(tip_shape @ eta)
    jacobian = np.zeros((2, params.coordinate_count))
    jacobian[:, 6:8] = np.eye(2)
    jacobian[:, 8] = params.beam.length_m * _normal(q[8]) - deflection * _direction(
        q[8]
    )
    jacobian[:, N_RIGID_COORDINATES:] = np.outer(_normal(q[8]), tip_shape)
    return jacobian


def rollout(
    q0: object,
    qdot0: object,
    control_law: ControlLaw,
    params: ModalShaftCouplingParams,
    config: ModalShaftCouplingConfig,
) -> ModalShaftTrace:
    """Integrate the coupled constrained modal-shaft system."""
    max_omega = 2.0 * np.pi * modal_shaft_basis(params).coupled_frequencies_hz[-1]
    if config.step_s * max_omega >= 0.5:
        raise ValueError("step_s does not resolve the highest retained shaft mode")
    q_initial, first_correction = _project_configuration(
        _state("q0", q0, params), params, config
    )
    velocity_initial = _project_velocity(
        q_initial, _state("qdot0", qdot0, params), params, config
    )
    samples = config.interval_count + 1
    time = config.start_time_s + np.arange(samples, dtype=float) * config.step_s
    q = np.empty((samples, params.coordinate_count))
    qdot = np.empty_like(q)
    corrections = np.zeros(samples)
    projection_energy = np.zeros(samples)
    q[0], qdot[0], corrections[0] = q_initial, velocity_initial, first_correction
    for index in range(samples - 1):
        control = control_law(float(time[index]), q[index].copy(), qdot[index].copy())
        solved = solve_constrained_dynamics(q[index], qdot[index], control, params)
        half_velocity = qdot[index] + 0.5 * config.step_s * solved.qddot
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
        next_solved = solve_constrained_dynamics(
            q[index + 1], half_velocity, next_control, params
        )
        trial_velocity = half_velocity + 0.5 * config.step_s * next_solved.qddot
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
    count = samples
    qddot = np.empty_like(q)
    multipliers = np.empty((count, N_CONSTRAINTS))
    contacts = np.empty((count, 2, 2))
    force_couple = np.empty(count)
    direct_wrist = np.empty(count)
    contact_power = np.empty(count)
    wrench_power = np.empty(count)
    energy = np.empty(count)
    applied_power = np.empty(count)
    dissipation_power = np.empty(count)
    position = np.empty(count)
    velocity_residual = np.empty(count)
    kkt = np.empty(count)
    acceleration = np.empty(count)
    clubhead = np.empty((count, 2))
    clubhead_velocity = np.empty_like(clubhead)
    for index, (state, velocity, control) in enumerate(
        zip(q, qdot, controls, strict=True)
    ):
        solved = solve_constrained_dynamics(state, velocity, control, params)
        qddot[index] = solved.qddot
        multipliers[index] = solved.multipliers_n
        contacts[index] = solved.contact_force_on_club_n
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
        right_velocity = _grip_velocity(
            state, velocity, params.mechanism.right_grip_offset_m
        )
        left_velocity = _grip_velocity(
            state, velocity, params.mechanism.left_grip_offset_m
        )
        contact_power[index] = (
            contacts[index, 0] @ right_velocity + contacts[index, 1] @ left_velocity
        )
        wrench_power[index] = (contacts[index, 0] + contacts[index, 1]) @ velocity[
            6:8
        ] + force_couple[index] * velocity[8]
        applied_power[index] = control_generalized_force(control, params) @ velocity
        damping = _damping_force(velocity, params)
        dissipation_power[index] = damping @ velocity
        clubhead[index] = points["clubhead"]
        clubhead_velocity[index] = _clubhead_jacobian(state, params) @ velocity
    basis = modal_shaft_basis(params)
    eta = q[:, N_RIGID_COORDINATES:]
    eta_rate = qdot[:, N_RIGID_COORDINATES:]
    strain = 0.5 * np.einsum(
        "ti,ij,tj->t", eta, basis.modal_stiffness, eta, optimize=True
    )
    omega = 2.0 * np.pi * basis.coupled_frequencies_hz
    modal_damping = 2.0 * params.damping_ratio * omega
    damping_power = -np.sum(modal_damping * eta_rate**2, axis=1)
    return ModalShaftTrace(
        time=np.asarray(time),
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
        modal_coordinates=eta,
        modal_velocities=eta_rate,
        modal_tip_deflection_m=eta @ basis.mode_shapes[-1],
        shaft_strain_energy_j=strain,
        shaft_damping_power_w=damping_power,
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
    "ModalShaftBasis",
    "ModalShaftCouplingConfig",
    "ModalShaftCouplingParams",
    "ModalShaftTrace",
    "constraint_jacobian",
    "constraint_vector",
    "control_generalized_force",
    "initial_state",
    "kinematics",
    "mass_matrix",
    "mechanical_energy",
    "modal_shaft_basis",
    "potential_energy",
    "rollout",
    "solve_constrained_dynamics",
]
