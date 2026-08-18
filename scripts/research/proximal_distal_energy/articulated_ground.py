"""Finite planar base and ground wrench for the articulated shaft model.

The three reduced coordinates translate the human tree in world ``x`` and
``z`` and rotate it about world ``y``.  The independently rooted club remains
connected only through the qualified distributed-grip law.  Parameters are
synthetic mechanism references, not force-plate or human calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_distributed_grip import (
    DistributedGripConfig,
    DistributedGripSnapshot,
    distributed_contact_kinematics,
    evaluate_distributed_grip_kinematics,
)
from scripts.research.proximal_distal_energy.articulated_shaft import (
    ArticulatedShaftProperties,
    ShaftEnergy,
    shaft_state_energy,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
    forward_kinematics,
)

FloatArray = NDArray[np.float64]
GroundActivation = Literal["fixed", "translation", "free_moment", "coupled"]
FULL_LABELS = ("base_x", "base_z", "base_pitch_y")


@dataclass(frozen=True, slots=True)
class ArticulatedGroundConfig:
    """Passive ground law, reference transport, and domain contract."""

    activation: GroundActivation = "coupled"
    translation_stiffness_n_m: tuple[float, float] = (15_000.0, 30_000.0)
    translation_damping_n_s_m: tuple[float, float] = (400.0, 800.0)
    free_moment_stiffness_nm_rad: float = 900.0
    free_moment_damping_nm_s_rad: float = 45.0
    center_of_pressure_xz_m: tuple[float, float] = (0.08, -0.95)
    translation_limit_m: float = 0.05
    rotation_limit_rad: float = np.deg2rad(10.0)
    derivative_step: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.activation not in {"fixed", "translation", "free_moment", "coupled"}:
            raise ValueError(
                "activation must be fixed, translation, free_moment, or coupled"
            )
        nonnegative_groups = (
            self.translation_stiffness_n_m,
            self.translation_damping_n_s_m,
        )
        if any(
            len(values) != 2
            or any(not np.isfinite(value) or value < 0.0 for value in values)
            or values[1] <= 0.0
            for values in nonnegative_groups
        ):
            raise ValueError(
                "translation stiffness and damping must be nonnegative pairs "
                "with positive vertical terms"
            )
        for name in (
            "free_moment_stiffness_nm_rad",
            "free_moment_damping_nm_s_rad",
            "translation_limit_m",
            "rotation_limit_rad",
            "derivative_step",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        cop = np.asarray(self.center_of_pressure_xz_m, dtype=float)
        if cop.shape != (2,) or np.any(~np.isfinite(cop)):
            raise ValueError("center_of_pressure_xz_m must be a finite pair")


@dataclass(frozen=True, slots=True)
class ArticulatedGroundProperties:
    """Active base coordinates and their passive constitutive matrices."""

    config: ArticulatedGroundConfig
    active_labels: tuple[str, ...]
    active_full_indices: tuple[int, ...]
    stiffness: FloatArray
    damping: FloatArray
    calibration_status: str = "synthetic_reference_not_force_plate_calibrated"

    @property
    def coordinate_count(self) -> int:
        return len(self.active_labels)


@dataclass(frozen=True, slots=True)
class GroundWrench:
    """Ground-on-body wrench and conjugate passive-power ledger."""

    force_n: FloatArray
    intrinsic_free_moment_nm: float
    transported_moment_nm: float
    generalized_force: FloatArray
    applied_power_w: float
    strain_energy_j: float
    storage_power_w: float
    damping_power_w: float
    power_residual_w: float


@dataclass(frozen=True, slots=True)
class GroundEnergy:
    """Named energy components added to the qualified shaft state."""

    shaft_energy: ShaftEnergy
    base_kinetic_j: float
    ground_strain_j: float
    extra_gravitational_j: float
    total_mechanical_j: float


def _active_indices(activation: GroundActivation) -> tuple[int, ...]:
    return {
        "fixed": (),
        "translation": (0, 1),
        "free_moment": (2,),
        "coupled": (0, 1, 2),
    }[activation]


def build_articulated_ground(
    config: ArticulatedGroundConfig = ArticulatedGroundConfig(),
) -> ArticulatedGroundProperties:
    """Build the active passive-law matrices without fitting an outcome."""

    indices = _active_indices(config.activation)
    full_stiffness = np.diag(
        (*config.translation_stiffness_n_m, config.free_moment_stiffness_nm_rad)
    )
    full_damping = np.diag(
        (*config.translation_damping_n_s_m, config.free_moment_damping_nm_s_rad)
    )
    selector = np.asarray(indices, dtype=int)
    return ArticulatedGroundProperties(
        config=config,
        active_labels=tuple(FULL_LABELS[index] for index in indices),
        active_full_indices=indices,
        stiffness=full_stiffness[np.ix_(selector, selector)],
        damping=full_damping[np.ix_(selector, selector)],
    )


def _full_base_state(
    value: FloatArray, properties: ArticulatedGroundProperties
) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (properties.coordinate_count,) or np.any(~np.isfinite(array)):
        raise ValueError("base state must match the active ground coordinates")
    result = np.zeros(3)
    result[np.asarray(properties.active_full_indices, dtype=int)] = array
    return result


def _base_jacobians(position_m: FloatArray) -> tuple[FloatArray, FloatArray]:
    linear = np.column_stack(
        (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.cross(np.array([0.0, 1.0, 0.0]), position_m),
        )
    )
    angular = np.zeros((3, 3))
    angular[:, 2] = np.array([0.0, 1.0, 0.0])
    return linear, angular


def ground_mass_increment(
    model: SpatialModel,
    q: FloatArray,
    shaft: ArticulatedShaftProperties,
    ground: ArticulatedGroundProperties,
) -> FloatArray:
    """Return rigid/base and base/base blocks for the non-club body tree."""

    shaft_count = shaft.coordinate_count
    ground_count = ground.coordinate_count
    existing = model.nq + shaft_count
    increment = np.zeros((existing + ground_count, existing + ground_count))
    if ground_count == 0:
        return increment
    kin = forward_kinematics(model, np.asarray(q, dtype=float))
    selector = np.asarray(ground.active_full_indices, dtype=int)
    cross = np.zeros((model.nq, ground_count))
    base = np.zeros((ground_count, ground_count))
    for body_index, body in enumerate(model.bodies):
        if body.region == "club":
            continue
        linear_full, angular_full = _base_jacobians(kin.body_position_m[body_index])
        linear = linear_full[:, selector]
        angular = angular_full[:, selector]
        inertia = 0.4 * body.mass_kg * body.radius_m**2
        cross += (
            body.mass_kg * kin.body_linear_jacobian[body_index].T @ linear
            + inertia * kin.body_angular_jacobian[body_index].T @ angular
        )
        base += body.mass_kg * linear.T @ linear + inertia * angular.T @ angular
    increment[: model.nq, existing:] = cross
    increment[existing:, : model.nq] = cross.T
    increment[existing:, existing:] = base
    return increment


def augmented_ground_mass_matrix(
    model: SpatialModel,
    q: FloatArray,
    shaft_mass: FloatArray,
    shaft: ArticulatedShaftProperties,
    ground: ArticulatedGroundProperties,
) -> FloatArray:
    """Append finite-base inertia while preserving exact fixed-base reduction."""

    shaft_mass = np.asarray(shaft_mass, dtype=float)
    existing = model.nq + shaft.coordinate_count
    if shaft_mass.shape != (existing, existing):
        raise ValueError("shaft_mass shape does not match the rigid/shaft coordinates")
    if ground.coordinate_count == 0:
        return shaft_mass.copy()
    matrix = np.zeros(
        (existing + ground.coordinate_count, existing + ground.coordinate_count)
    )
    matrix[:existing, :existing] = shaft_mass
    matrix += ground_mass_increment(model, q, shaft, ground)
    matrix = 0.5 * (matrix + matrix.T)
    minimum = float(np.min(np.linalg.eigvalsh(matrix)))
    if minimum <= 1.0e-10:
        raise RuntimeError(
            "augmented ground mass matrix is not positive definite: "
            f"minimum_eigenvalue={minimum:.17g}"
        )
    return matrix


def ground_mass_increment_coriolis(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    eta_dot: FloatArray,
    base_velocity: FloatArray,
    shaft: ArticulatedShaftProperties,
    ground: ArticulatedGroundProperties,
) -> FloatArray:
    """Return the Christoffel bias of the posture-varying base mass blocks."""

    total = model.nq + shaft.coordinate_count + ground.coordinate_count
    if ground.coordinate_count == 0:
        return np.zeros(total)
    derivatives = np.zeros((total, total, total))
    step = ground.config.derivative_step
    for index, joint in enumerate(model.joints):
        if joint.region == "club":
            continue
        delta = np.zeros(model.nq)
        delta[index] = step
        derivatives[index] = (
            ground_mass_increment(model, q + delta, shaft, ground)
            - ground_mass_increment(model, q - delta, shaft, ground)
        ) / (2.0 * step)
    velocity = np.concatenate((qd, eta_dot, base_velocity))
    first = np.einsum("kij,j,k->i", derivatives, velocity, velocity)
    third = np.einsum("ijk,j,k->i", derivatives, velocity, velocity)
    return first - 0.5 * third


def ground_extra_gravitational_energy(
    model: SpatialModel,
    q: FloatArray,
    base: FloatArray,
    ground: ArticulatedGroundProperties,
) -> float:
    """Return gravity change caused only by finite human-base motion."""

    full = _full_base_state(base, ground)
    if ground.coordinate_count == 0:
        return 0.0
    kin = forward_kinematics(model, np.asarray(q, dtype=float))
    c, s = np.cos(full[2]), np.sin(full[2])
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    translation = np.array([full[0], 0.0, full[1]])
    value = 0.0
    for body_index, body in enumerate(model.bodies):
        if body.region == "club":
            continue
        original = kin.body_position_m[body_index]
        transformed = rotation @ original + translation
        value += body.mass_kg * 9.80665 * (transformed[2] - original[2])
    return float(value)


def ground_extra_potential_gradient(
    model: SpatialModel,
    q: FloatArray,
    base: FloatArray,
    shaft: ArticulatedShaftProperties,
    ground: ArticulatedGroundProperties,
) -> FloatArray:
    """Return the augmented gradient of finite-base gravitational energy."""

    total = model.nq + shaft.coordinate_count + ground.coordinate_count
    gradient = np.zeros(total)
    if ground.coordinate_count == 0:
        return gradient
    step = ground.config.derivative_step
    for index, joint in enumerate(model.joints):
        if joint.region == "club":
            continue
        delta = np.zeros(model.nq)
        delta[index] = step
        gradient[index] = (
            ground_extra_gravitational_energy(model, q + delta, base, ground)
            - ground_extra_gravitational_energy(model, q - delta, base, ground)
        ) / (2.0 * step)
    offset = model.nq + shaft.coordinate_count
    for index in range(ground.coordinate_count):
        delta = np.zeros(ground.coordinate_count)
        delta[index] = step
        gradient[offset + index] = (
            ground_extra_gravitational_energy(model, q, base + delta, ground)
            - ground_extra_gravitational_energy(model, q, base - delta, ground)
        ) / (2.0 * step)
    return gradient


def evaluate_ground_wrench(
    base: FloatArray,
    base_velocity: FloatArray,
    properties: ArticulatedGroundProperties,
) -> GroundWrench:
    """Evaluate ground-on-body force, free moment, and conjugate power."""

    _full_base_state(base, properties)
    _full_base_state(base_velocity, properties)
    generalized = -properties.stiffness @ base - properties.damping @ base_velocity
    full_generalized = np.zeros(3)
    full_generalized[np.asarray(properties.active_full_indices, dtype=int)] = (
        generalized
    )
    force = np.array([full_generalized[0], 0.0, full_generalized[1]])
    intrinsic = float(full_generalized[2])
    cop_x, cop_z = properties.config.center_of_pressure_xz_m
    transported = intrinsic + cop_z * force[0] - cop_x * force[2]
    applied_power = float(generalized @ base_velocity)
    strain = 0.5 * float(base @ properties.stiffness @ base)
    storage_power = float(properties.stiffness @ base @ base_velocity)
    damping_power = -float(base_velocity @ properties.damping @ base_velocity)
    residual = applied_power + storage_power - damping_power
    return GroundWrench(
        force_n=force,
        intrinsic_free_moment_nm=intrinsic,
        transported_moment_nm=float(transported),
        generalized_force=generalized,
        applied_power_w=applied_power,
        strain_energy_j=strain,
        storage_power_w=storage_power,
        damping_power_w=damping_power,
        power_residual_w=float(residual),
    )


def evaluate_ground_coupled_grip(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    eta_dot: FloatArray,
    base: FloatArray,
    base_velocity: FloatArray,
    shaft: ArticulatedShaftProperties,
    ground: ArticulatedGroundProperties,
    *,
    grip_span_m: float,
    hand_contact_local_x_m: float,
    reference_lengths_m: FloatArray,
    config: DistributedGripConfig,
) -> DistributedGripSnapshot:
    """Evaluate grip contact after moving only the articulated human tree."""

    full = _full_base_state(base, ground)
    _full_base_state(base_velocity, ground)
    hand, hand_q_jac, grip, grip_q_jac = distributed_contact_kinematics(
        model,
        q,
        grip_span_m=grip_span_m,
        hand_contact_local_x_m=hand_contact_local_x_m,
        config=config,
    )
    c, s = np.cos(full[2]), np.sin(full[2])
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    translation = np.array([full[0], 0.0, full[1]])
    moved_hand = np.einsum("ij,abj->abi", rotation, hand) + translation
    total = model.nq + shaft.coordinate_count + ground.coordinate_count
    hand_jac = np.zeros((*hand.shape, total))
    grip_jac = np.zeros((*grip.shape, total))
    hand_jac[..., : model.nq] = np.einsum("ij,abjk->abik", rotation, hand_q_jac)
    grip_jac[..., : model.nq] = grip_q_jac
    selector = np.asarray(ground.active_full_indices, dtype=int)
    base_offset = model.nq + shaft.coordinate_count
    for hand_index in range(2):
        for station_index in range(config.station_count_per_hand):
            linear, _ = _base_jacobians(moved_hand[hand_index, station_index])
            hand_jac[hand_index, station_index, :, base_offset:] = linear[:, selector]
    velocity = np.concatenate((qd, eta_dot, base_velocity))
    return evaluate_distributed_grip_kinematics(
        moved_hand,
        hand_jac,
        grip,
        grip_jac,
        velocity,
        reference_lengths_m=reference_lengths_m,
        config=config,
    )


def ground_state_energy(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    eta: FloatArray,
    eta_dot: FloatArray,
    base: FloatArray,
    base_velocity: FloatArray,
    shaft: ArticulatedShaftProperties,
    ground: ArticulatedGroundProperties,
) -> GroundEnergy:
    """Return shaft, base-kinetic, ground-strain, and gravity components."""

    shaft_energy = shaft_state_energy(model, q, qd, eta, eta_dot, shaft)
    _full_base_state(base, ground)
    _full_base_state(base_velocity, ground)
    velocity = np.concatenate((qd, eta_dot, base_velocity))
    kinetic = 0.5 * float(
        velocity @ ground_mass_increment(model, q, shaft, ground) @ velocity
    )
    strain = 0.5 * float(base @ ground.stiffness @ base)
    gravity = ground_extra_gravitational_energy(model, q, base, ground)
    total = shaft_energy.total_mechanical_j + kinetic + strain + gravity
    return GroundEnergy(
        shaft_energy=shaft_energy,
        base_kinetic_j=float(kinetic),
        ground_strain_j=float(strain),
        extra_gravitational_j=float(gravity),
        total_mechanical_j=float(total),
    )


__all__ = [
    "ArticulatedGroundConfig",
    "ArticulatedGroundProperties",
    "GroundEnergy",
    "GroundWrench",
    "augmented_ground_mass_matrix",
    "build_articulated_ground",
    "evaluate_ground_wrench",
    "evaluate_ground_coupled_grip",
    "ground_extra_gravitational_energy",
    "ground_extra_potential_gradient",
    "ground_mass_increment",
    "ground_mass_increment_coriolis",
    "ground_state_energy",
]
