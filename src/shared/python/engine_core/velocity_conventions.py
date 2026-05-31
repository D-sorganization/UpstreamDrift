"""Floating-base velocity representation conventions.

The suite stores six-dimensional floating-base vectors in the canonical
``SPATIAL_JACOBIAN_ORDER`` order: angular rows first, then linear rows. This
module adds the missing frame/velocity-representation contract around that
row order so engine adapters can normalize native backend conventions before
their dynamics terms reach ``PhysicsEngine`` callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.engine_core.capabilities import SPATIAL_JACOBIAN_ORDER

SPATIAL_VECTOR_SIZE = 6
LINEAR_GRAVITY_UNIT = "m/s^2"
ANGULAR_VELOCITY_UNIT = "rad/s"
LINEAR_VELOCITY_UNIT = "m/s"
CANONICAL_GRAVITY_INERTIAL: tuple[float, float, float] = (0.0, 0.0, -9.80665)


class VelocityRepresentation(str, Enum):
    """Supported floating-base velocity representations.

    Values:
        BODY_FIXED: Angular and linear components are expressed in the base
            body frame.
        INERTIAL: Angular and linear components are expressed in the inertial
            world frame.
        MIXED: Angular components are expressed in the base body frame while
            linear components are expressed in the inertial world frame. This
            matches JaxSim's default floating-base convention.
    """

    BODY_FIXED = "body_fixed"
    INERTIAL = "inertial"
    MIXED = "mixed"


CANONICAL_VELOCITY_REPRESENTATION = VelocityRepresentation.INERTIAL
"""Suite-normalized floating-base velocity representation."""


@dataclass(frozen=True)
class FloatingBaseConvention:
    """Complete floating-base convention contract for engine adapters.

    Attributes:
        velocity_representation: Frame convention for the six-dimensional base
            velocity ordered as ``SPATIAL_JACOBIAN_ORDER``.
        gravity_inertial_mps2: Gravity vector expressed in inertial/world axes,
            in metres per second squared.
        base_frame: Name of the root body frame whose orientation maps body
            vectors into inertial vectors.
        spatial_jacobian_order: Row order for 6D spatial vectors. This must
            stay aligned with ``SPATIAL_JACOBIAN_ORDER``.
    """

    velocity_representation: VelocityRepresentation
    gravity_inertial_mps2: tuple[float, float, float] = CANONICAL_GRAVITY_INERTIAL
    base_frame: str = "base"
    spatial_jacobian_order: tuple[str, str] = SPATIAL_JACOBIAN_ORDER

    def __post_init__(self) -> None:
        if self.spatial_jacobian_order != SPATIAL_JACOBIAN_ORDER:
            raise ValueError("floating-base convention must use SPATIAL_JACOBIAN_ORDER")
        if len(self.gravity_inertial_mps2) != 3:
            raise ValueError("gravity_inertial_mps2 must contain three values")
        if not self.base_frame.strip():
            raise ValueError("base_frame must be non-empty")


CANONICAL_FLOATING_BASE_CONVENTION = FloatingBaseConvention(
    velocity_representation=CANONICAL_VELOCITY_REPRESENTATION,
)


@dataclass(frozen=True)
class SingleFloatingBodyDynamics:
    """Analytic ``h`` and ``g`` terms for a free rigid body at its COM.

    Attributes:
        h: Bias wrench ordered ``[angular; linear]``. Angular entries are
            torques in N*m and linear entries are forces in N, expressed in
            the requested velocity representation.
        g: Gravity wrench ordered ``[angular; linear]``. Angular entries are
            torques in N*m and linear entries are forces in N, expressed in
            the requested velocity representation.
    """

    h: NDArray[np.float64]
    g: NDArray[np.float64]


def convert_floating_base_velocity(
    velocity: ArrayLike,
    *,
    source: VelocityRepresentation,
    target: VelocityRepresentation,
    rotation_inertial_from_body: ArrayLike,
) -> NDArray[np.float64]:
    """Convert a 6D floating-base velocity between supported representations.

    Args:
        velocity: Six values ordered as ``[angular; linear]`` according to
            ``SPATIAL_JACOBIAN_ORDER``. Angular velocity is in rad/s and linear
            velocity is in m/s.
        source: Representation used by ``velocity``.
        target: Desired output representation.
        rotation_inertial_from_body: 3x3 rotation matrix mapping vectors from
            the base body frame into the inertial/world frame.

    Returns:
        A float64 vector ordered ``[angular; linear]`` in the target
        representation.
    """

    velocity_array = _as_spatial_vector(velocity, "velocity")
    rotation = _as_rotation(rotation_inertial_from_body)
    if source == target:
        return velocity_array.copy()
    inertial = _to_inertial(velocity_array, source, rotation)
    return _from_inertial(inertial, target, rotation)


def normalize_floating_base_velocity(
    velocity: ArrayLike,
    *,
    source: VelocityRepresentation,
    rotation_inertial_from_body: ArrayLike,
) -> NDArray[np.float64]:
    """Normalize a backend velocity to the suite's canonical convention.

    Args:
        velocity: Six values ordered as ``[angular; linear]``. Angular velocity
            is in rad/s and linear velocity is in m/s.
        source: Native backend velocity representation.
        rotation_inertial_from_body: 3x3 base orientation mapping body-frame
            vectors into inertial/world-frame vectors.

    Returns:
        The velocity in ``CANONICAL_VELOCITY_REPRESENTATION``.
    """

    return convert_floating_base_velocity(
        velocity,
        source=source,
        target=CANONICAL_VELOCITY_REPRESENTATION,
        rotation_inertial_from_body=rotation_inertial_from_body,
    )


def convert_gravity_vector(
    gravity_inertial_mps2: ArrayLike,
    *,
    target: VelocityRepresentation,
    rotation_inertial_from_body: ArrayLike,
) -> NDArray[np.float64]:
    """Express gravity in the linear frame implied by ``target``.

    Args:
        gravity_inertial_mps2: Gravity vector in inertial/world axes, in m/s^2.
        target: Velocity representation whose linear component frame is needed.
        rotation_inertial_from_body: 3x3 base orientation mapping body-frame
            vectors into inertial/world-frame vectors.

    Returns:
        Three gravity components in m/s^2. ``BODY_FIXED`` returns body-frame
        components; ``INERTIAL`` and ``MIXED`` return inertial-frame components.
    """

    gravity = _as_vector3(gravity_inertial_mps2, "gravity_inertial_mps2")
    rotation = _as_rotation(rotation_inertial_from_body)
    if target == VelocityRepresentation.BODY_FIXED:
        return rotation.T @ gravity
    return gravity.copy()


def single_floating_body_h_g(
    *,
    mass_kg: float,
    inertia_body_kg_m2: ArrayLike,
    angular_velocity: ArrayLike,
    representation: VelocityRepresentation,
    rotation_inertial_from_body: ArrayLike,
    gravity_inertial_mps2: ArrayLike = CANONICAL_GRAVITY_INERTIAL,
) -> SingleFloatingBodyDynamics:
    """Compute documented analytic ``h`` and ``g`` for one free rigid body.

    The base origin is the center of mass, so gravity produces no torque. The
    input angular velocity must be expressed in the angular frame implied by
    ``representation`` and is measured in rad/s. The returned vectors are
    ordered as ``[angular; linear]``. Angular components are torques in N*m;
    linear components are forces in N.

    Args:
        mass_kg: Body mass in kilograms.
        inertia_body_kg_m2: 3x3 body-frame inertia matrix in kg*m^2.
        angular_velocity: Three angular velocity components in rad/s, using
            the angular frame of ``representation``.
        representation: Desired velocity representation for ``h`` and ``g``.
        rotation_inertial_from_body: 3x3 base orientation mapping body-frame
            vectors into inertial/world-frame vectors.
        gravity_inertial_mps2: Gravity vector in inertial/world axes, in m/s^2.

    Returns:
        Analytic bias and gravity terms for the free body under the requested
        representation.
    """

    if mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive")
    inertia_body = _as_matrix3(inertia_body_kg_m2, "inertia_body_kg_m2")
    omega = _as_vector3(angular_velocity, "angular_velocity")
    rotation = _as_rotation(rotation_inertial_from_body)
    gravity = _as_vector3(gravity_inertial_mps2, "gravity_inertial_mps2")

    omega_body = (
        omega
        if representation
        in {
            VelocityRepresentation.BODY_FIXED,
            VelocityRepresentation.MIXED,
        }
        else rotation.T @ omega
    )
    h_torque_body = np.cross(omega_body, inertia_body @ omega_body)
    gravity_force_inertial = mass_kg * gravity

    h_body = _stack_spatial(h_torque_body, np.zeros(3, dtype=np.float64))
    g_inertial = _stack_spatial(np.zeros(3, dtype=np.float64), gravity_force_inertial)
    h = convert_floating_base_velocity(
        h_body,
        source=VelocityRepresentation.BODY_FIXED,
        target=representation,
        rotation_inertial_from_body=rotation,
    )
    g = _from_inertial(g_inertial, representation, rotation)
    return SingleFloatingBodyDynamics(h=h, g=g)


def _to_inertial(
    velocity: NDArray[np.float64],
    source: VelocityRepresentation,
    rotation: NDArray[np.float64],
) -> NDArray[np.float64]:
    angular, linear = _split_spatial(velocity)
    if source == VelocityRepresentation.INERTIAL:
        return velocity.copy()
    if source == VelocityRepresentation.BODY_FIXED:
        return _stack_spatial(rotation @ angular, rotation @ linear)
    if source == VelocityRepresentation.MIXED:
        return _stack_spatial(rotation @ angular, linear)
    raise ValueError(f"unsupported velocity representation: {source}")


def _from_inertial(
    velocity: NDArray[np.float64],
    target: VelocityRepresentation,
    rotation: NDArray[np.float64],
) -> NDArray[np.float64]:
    angular, linear = _split_spatial(velocity)
    if target == VelocityRepresentation.INERTIAL:
        return velocity.copy()
    if target == VelocityRepresentation.BODY_FIXED:
        return _stack_spatial(rotation.T @ angular, rotation.T @ linear)
    if target == VelocityRepresentation.MIXED:
        return _stack_spatial(rotation.T @ angular, linear)
    raise ValueError(f"unsupported velocity representation: {target}")


def _split_spatial(
    vector: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    return vector[:3], vector[3:]


def _stack_spatial(angular: ArrayLike, linear: ArrayLike) -> NDArray[np.float64]:
    return np.concatenate(
        [_as_vector3(angular, "angular"), _as_vector3(linear, "linear")]
    )


def _as_spatial_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (SPATIAL_VECTOR_SIZE,):
        raise ValueError(f"{name} must have shape ({SPATIAL_VECTOR_SIZE},)")
    return array


def _as_vector3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return array


def _as_matrix3(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3)")
    return array


def _as_rotation(value: ArrayLike) -> NDArray[np.float64]:
    rotation = _as_matrix3(value, "rotation_inertial_from_body")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12):
        raise ValueError("rotation_inertial_from_body must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12):
        raise ValueError("rotation_inertial_from_body must have determinant +1")
    return rotation
