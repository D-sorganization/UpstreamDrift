"""Linearized bending and torsion states for the articulated club.

The elastic coordinates are synthetic equipment-mechanism comparators.  Their
bending frequency is inherited from the repository's finite-element authority;
torsion uses a declared tapered hollow-section calculation.  Neither is a
calibrated production shaft.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
import hashlib
import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_forward_contract import (
    mechanical_energy,
)
from scripts.research.proximal_distal_energy.spatial_full_body import (
    SpatialModel,
    forward_kinematics,
)

FloatArray = NDArray[np.float64]
ShaftActivation = Literal["rigid", "bending", "torsion", "coupled"]
FULL_LABELS = ("bend_x", "bend_y", "torsion")
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs/research/proximal_distal_energy_transfer/data"


@dataclass(frozen=True, slots=True)
class ArticulatedShaftConfig:
    """Constitutive, geometry, and promotion-boundary contract."""

    activation: ShaftActivation = "coupled"
    shaft_length_m: float = 1.08
    poisson_ratio: float = 0.30
    damping_ratio: float = 0.018
    small_deflection_limit: float = 0.05
    twist_limit_rad: float = np.deg2rad(10.0)
    derivative_step_rad: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.activation not in {"rigid", "bending", "torsion", "coupled"}:
            raise ValueError("activation must be rigid, bending, torsion, or coupled")
        for name in (
            "shaft_length_m",
            "small_deflection_limit",
            "twist_limit_rad",
            "derivative_step_rad",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.poisson_ratio) or not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie in (-1, 0.5)")
        if not np.isfinite(self.damping_ratio) or not 0.0 <= self.damping_ratio < 1.0:
            raise ValueError("damping_ratio must be finite and in [0, 1)")


@dataclass(frozen=True, slots=True)
class _BodyMode:
    body_index: int
    mass_kg: float
    inertia_kg_m2: float
    location_m: float
    bend_shape: float
    bend_slope: float
    torsion_shape: float


@dataclass(frozen=True, slots=True)
class ArticulatedShaftProperties:
    """Active elastic matrices and their structural provenance."""

    config: ArticulatedShaftConfig
    active_labels: tuple[str, ...]
    active_full_indices: tuple[int, ...]
    body_modes: tuple[_BodyMode, ...]
    elastic_mass: FloatArray
    elastic_stiffness: FloatArray
    elastic_damping: FloatArray
    bending_frequency_hz: float
    torsion_frequency_hz: float
    fe_bending_frequency_relative_error: float
    calibration_status: str = "synthetic_reference_not_equipment_calibrated"

    @property
    def coordinate_count(self) -> int:
        return len(self.active_labels)


@dataclass(frozen=True, slots=True)
class ShaftEnergy:
    """Named components of the augmented passive mechanical energy."""

    rigid_mechanical_j: float
    elastic_kinetic_j: float
    elastic_strain_j: float
    extra_gravitational_j: float
    total_mechanical_j: float


def _active_indices(activation: ShaftActivation) -> tuple[int, ...]:
    return {
        "rigid": (),
        "bending": (0, 1),
        "torsion": (2,),
        "coupled": (0, 1, 2),
    }[activation]


def _beam_number(beam: Mapping[str, object], key: str) -> float:
    value = beam[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"articulated shaft beam field {key} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise RuntimeError(f"articulated shaft beam field {key} must be finite")
    return result


@cache
def _structural_basis() -> tuple[dict[str, object], dict[str, FloatArray]]:
    json_path = DATA_DIR / "articulated_shaft_structural_basis.json"
    npz_path = DATA_DIR / "articulated_shaft_structural_basis.npz"
    record = json.loads(json_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "articulated-shaft-structural-basis/v1":
        raise RuntimeError("articulated shaft structural basis schema is unsupported")
    for relative, expected in record["source_sha256"].items():
        observed = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"articulated shaft basis source is stale: {relative}")
    with np.load(npz_path) as source:
        arrays = {key: np.asarray(source[key], dtype=float) for key in source.files}
    count = int(record["beam_configuration"]["element_count"]) + 1
    if (
        arrays["locations_m"].shape != (count,)
        or arrays["tip_normalized_shape"].shape != (count,)
        or arrays["tip_normalized_slope_per_m"].shape != (count,)
        or arrays["frequency_hz"].shape != (1,)
        or np.any(~np.isfinite(np.concatenate(tuple(arrays.values()))))
    ):
        raise RuntimeError("articulated shaft structural basis arrays are invalid")
    return record, arrays


def _torsion_stiffness(config: ArticulatedShaftConfig) -> float:
    record, _ = _structural_basis()
    beam = cast(Mapping[str, object], record["beam_configuration"])
    nodes, weights = np.polynomial.legendre.leggauss(64)
    locations = 0.5 * config.shaft_length_m * (nodes + 1.0)
    fraction = locations / config.shaft_length_m
    outer = _beam_number(beam, "butt_diameter_m") + fraction * (
        _beam_number(beam, "tip_diameter_m") - _beam_number(beam, "butt_diameter_m")
    )
    inner = outer - 2.0 * _beam_number(beam, "wall_thickness_m")
    polar = np.pi * (outer**4 - inner**4) / 32.0
    shear = _beam_number(beam, "youngs_modulus_pa") / (
        2.0 * (1.0 + config.poisson_ratio)
    )
    compliance = 0.5 * config.shaft_length_m * float(np.sum(weights / (shear * polar)))
    return 1.0 / compliance


def _body_modes(
    model: SpatialModel, config: ArticulatedShaftConfig
) -> tuple[tuple[_BodyMode, ...], float]:
    record, basis = _structural_basis()
    beam = cast(Mapping[str, object], record["beam_configuration"])
    beam_length = _beam_number(beam, "length_m")
    locations = basis["locations_m"]
    shapes = basis["tip_normalized_shape"]
    slopes = basis["tip_normalized_slope_per_m"]
    result = []
    for index, body in enumerate(model.bodies):
        if body.name not in {"club_shaft_mass", "clubhead_mass"}:
            continue
        location = abs(float(body.com_offset_m[2]))
        beam_location = min(
            beam_length,
            location * beam_length / config.shaft_length_m,
        )
        shape = float(np.interp(beam_location, locations, shapes))
        slope = float(np.interp(beam_location, locations, slopes))
        slope *= beam_length / config.shaft_length_m
        result.append(
            _BodyMode(
                body_index=index,
                mass_kg=body.mass_kg,
                inertia_kg_m2=0.4 * body.mass_kg * body.radius_m**2,
                location_m=location,
                bend_shape=shape,
                bend_slope=slope,
                torsion_shape=location / config.shaft_length_m,
            )
        )
    if len(result) != 2:
        raise RuntimeError("articulated model must contain shaft and clubhead masses")
    return tuple(result), float(basis["frequency_hz"][0])


def build_articulated_shaft(
    model: SpatialModel,
    config: ArticulatedShaftConfig = ArticulatedShaftConfig(),
) -> ArticulatedShaftProperties:
    """Build the active bending/torsion reduction from shared authorities."""

    if not isinstance(config, ArticulatedShaftConfig):
        raise TypeError("config must be an ArticulatedShaftConfig")
    body_modes, bending_frequency = _body_modes(model, config)
    full_mass = np.zeros((3, 3))
    for body in body_modes:
        full_mass[0, 0] += (
            body.mass_kg * body.bend_shape**2 + body.inertia_kg_m2 * body.bend_slope**2
        )
        full_mass[1, 1] += (
            body.mass_kg * body.bend_shape**2 + body.inertia_kg_m2 * body.bend_slope**2
        )
        full_mass[2, 2] += body.inertia_kg_m2 * body.torsion_shape**2
    bending_stiffness = full_mass[0, 0] * (2.0 * np.pi * bending_frequency) ** 2
    torsion_stiffness = _torsion_stiffness(config)
    full_stiffness = np.diag([bending_stiffness, bending_stiffness, torsion_stiffness])
    full_damping = np.diag(
        2.0
        * config.damping_ratio
        * np.sqrt(np.diag(full_mass) * np.diag(full_stiffness))
    )
    indices = _active_indices(config.activation)
    selector = np.asarray(indices, dtype=int)
    mass = full_mass[np.ix_(selector, selector)]
    stiffness = full_stiffness[np.ix_(selector, selector)]
    damping = full_damping[np.ix_(selector, selector)]
    reconstructed = np.sqrt(bending_stiffness / full_mass[0, 0]) / (2.0 * np.pi)
    torsion_frequency = np.sqrt(torsion_stiffness / full_mass[2, 2]) / (2.0 * np.pi)
    return ArticulatedShaftProperties(
        config=config,
        active_labels=tuple(FULL_LABELS[index] for index in indices),
        active_full_indices=indices,
        body_modes=body_modes,
        elastic_mass=mass,
        elastic_stiffness=stiffness,
        elastic_damping=damping,
        bending_frequency_hz=bending_frequency,
        torsion_frequency_hz=float(torsion_frequency),
        fe_bending_frequency_relative_error=float(
            abs(reconstructed - bending_frequency) / bending_frequency
        ),
    )


def _modal_jacobians(
    rotation: FloatArray, body: _BodyMode, indices: tuple[int, ...]
) -> tuple[FloatArray, FloatArray]:
    linear = np.column_stack(
        (
            rotation[:, 0] * body.bend_shape,
            rotation[:, 1] * body.bend_shape,
            np.zeros(3),
        )
    )
    angular = np.column_stack(
        (
            rotation[:, 1] * body.bend_slope,
            -rotation[:, 0] * body.bend_slope,
            rotation[:, 2] * body.torsion_shape,
        )
    )
    selector = np.asarray(indices, dtype=int)
    return linear[:, selector], angular[:, selector]


def mass_increment(
    model: SpatialModel,
    q: FloatArray,
    properties: ArticulatedShaftProperties,
) -> FloatArray:
    """Return only the rigid-elastic and elastic-elastic kinetic blocks."""

    count = properties.coordinate_count
    increment = np.zeros((model.nq + count, model.nq + count))
    if count == 0:
        return increment
    kin = forward_kinematics(model, np.asarray(q, dtype=float))
    cross = np.zeros((model.nq, count))
    elastic = np.zeros((count, count))
    for body in properties.body_modes:
        linear, angular = _modal_jacobians(
            kin.body_rotation[body.body_index],
            body,
            properties.active_full_indices,
        )
        rigid_linear = kin.body_linear_jacobian[body.body_index]
        rigid_angular = kin.body_angular_jacobian[body.body_index]
        cross += body.mass_kg * rigid_linear.T @ linear
        cross += body.inertia_kg_m2 * rigid_angular.T @ angular
        elastic += body.mass_kg * linear.T @ linear
        elastic += body.inertia_kg_m2 * angular.T @ angular
    increment[: model.nq, model.nq :] = cross
    increment[model.nq :, : model.nq] = cross.T
    increment[model.nq :, model.nq :] = elastic
    return increment


def augmented_mass_matrix(
    model: SpatialModel,
    q: FloatArray,
    rigid_mass: FloatArray,
    properties: ArticulatedShaftProperties,
) -> FloatArray:
    """Return the symmetric native-rigid plus linearized elastic mass matrix."""

    count = properties.coordinate_count
    if count == 0:
        return np.asarray(rigid_mass, dtype=float).copy()
    matrix = np.zeros((model.nq + count, model.nq + count))
    matrix[: model.nq, : model.nq] = rigid_mass
    matrix += mass_increment(model, q, properties)
    matrix = 0.5 * (matrix + matrix.T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(matrix)))
    if minimum_eigenvalue <= 1.0e-10:
        raise RuntimeError(
            "augmented shaft mass matrix is not positive definite: "
            f"minimum_eigenvalue={minimum_eigenvalue:.17g}"
        )
    return matrix


def _full_elastic_state(
    value: FloatArray, properties: ArticulatedShaftProperties
) -> FloatArray:
    array = np.asarray(value, dtype=float)
    if array.shape != (properties.coordinate_count,) or np.any(~np.isfinite(array)):
        raise ValueError("elastic state must match the active shaft coordinates")
    result = np.zeros(3)
    result[np.asarray(properties.active_full_indices, dtype=int)] = array
    return result


def extra_gravitational_energy(
    model: SpatialModel,
    q: FloatArray,
    eta: FloatArray,
    properties: ArticulatedShaftProperties,
) -> float:
    """Return first-order gravitational energy of transverse deflection."""

    full = _full_elastic_state(eta, properties)
    if properties.coordinate_count == 0:
        return 0.0
    kin = forward_kinematics(model, np.asarray(q, dtype=float))
    value = 0.0
    gravity = 9.80665
    for body in properties.body_modes:
        local = np.array([body.bend_shape * full[0], body.bend_shape * full[1], 0.0])
        world = kin.body_rotation[body.body_index] @ local
        value += body.mass_kg * gravity * world[2]
    return float(value)


def mass_increment_coriolis(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    eta_dot: FloatArray,
    properties: ArticulatedShaftProperties,
) -> FloatArray:
    """Return Christoffel bias generated by the configuration-varying cross block."""

    count = properties.coordinate_count
    if count == 0:
        return np.zeros(model.nq)
    total = model.nq + count
    derivatives = np.zeros((total, total, total))
    step = properties.config.derivative_step_rad
    for index in model.club_dof_indices[3:]:
        delta = np.zeros(model.nq)
        delta[index] = step
        derivatives[index] = (
            mass_increment(model, q + delta, properties)
            - mass_increment(model, q - delta, properties)
        ) / (2.0 * step)
    velocity = np.concatenate((qd, eta_dot))
    first = np.einsum("kij,j,k->i", derivatives, velocity, velocity)
    third = np.einsum("ijk,j,k->i", derivatives, velocity, velocity)
    return first - 0.5 * third


def extra_potential_gradient(
    model: SpatialModel,
    q: FloatArray,
    eta: FloatArray,
    properties: ArticulatedShaftProperties,
) -> FloatArray:
    """Return the augmented gradient of first-order deflection gravity."""

    count = properties.coordinate_count
    if count == 0:
        return np.zeros(model.nq)
    gradient = np.zeros(model.nq + count)
    step = properties.config.derivative_step_rad
    for index in model.club_dof_indices[3:]:
        delta = np.zeros(model.nq)
        delta[index] = step
        gradient[index] = (
            extra_gravitational_energy(model, q + delta, eta, properties)
            - extra_gravitational_energy(model, q - delta, eta, properties)
        ) / (2.0 * step)
    elastic_step = 1.0e-7
    for index in range(count):
        delta = np.zeros(count)
        delta[index] = elastic_step
        gradient[model.nq + index] = (
            extra_gravitational_energy(model, q, eta + delta, properties)
            - extra_gravitational_energy(model, q, eta - delta, properties)
        ) / (2.0 * elastic_step)
    return gradient


def shaft_state_energy(
    model: SpatialModel,
    q: FloatArray,
    qd: FloatArray,
    eta: FloatArray,
    eta_dot: FloatArray,
    properties: ArticulatedShaftProperties,
) -> ShaftEnergy:
    """Return the named mechanical energy ledger for one augmented state."""

    _full_elastic_state(eta, properties)
    _full_elastic_state(eta_dot, properties)
    rigid = mechanical_energy(model, q, qd)
    velocity = np.concatenate((qd, eta_dot))
    kinetic = 0.5 * float(velocity @ mass_increment(model, q, properties) @ velocity)
    strain = 0.5 * float(eta @ properties.elastic_stiffness @ eta)
    gravity = extra_gravitational_energy(model, q, eta, properties)
    return ShaftEnergy(
        rigid_mechanical_j=rigid,
        elastic_kinetic_j=kinetic,
        elastic_strain_j=strain,
        extra_gravitational_j=gravity,
        total_mechanical_j=rigid + kinetic + strain + gravity,
    )


__all__ = [
    "ArticulatedShaftConfig",
    "ArticulatedShaftProperties",
    "ShaftEnergy",
    "augmented_mass_matrix",
    "build_articulated_shaft",
    "extra_potential_gradient",
    "mass_increment_coriolis",
    "shaft_state_energy",
]
