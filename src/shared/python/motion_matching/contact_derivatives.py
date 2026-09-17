"""Analytic state derivatives of the shared sphere-ground contact law."""

from __future__ import annotations

import math
from typing import NamedTuple, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    ContactSample,
    GroundPlane,
    sphere_ground_contact,
)

Array: TypeAlias = NDArray[np.float64]


class ContactDerivatives(NamedTuple):
    """Contact sample and total-force Jacobians in world coordinates."""

    sample: ContactSample
    dforce_dcenter: Array
    dforce_dvelocity: Array
    differentiable: bool


def _readonly_matrix(value: Array) -> Array:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _at_roundoff_zero(value: float, scale: float = 1.0) -> bool:
    return bool(abs(value) <= 8.0 * np.finfo(float).eps * max(1.0, scale))


def _friction_scale_derivative(
    speed: float, parameters: ContactParameters
) -> tuple[float, float]:
    """Return ``a`` and ``u * da/du`` for friction ``-N * a(u) * t``."""
    transition = parameters.transition_velocity_m_s
    static = parameters.static_friction
    delta = static - parameters.dynamic_friction
    viscous = parameters.viscous_friction
    if speed == 0.0:
        return static / transition, 0.0

    ratio = speed / transition
    if ratio < 1e-4:
        ratio_squared = ratio * ratio
        quadratic = (delta + static / 3.0) / transition
        quartic = (5.0 * delta / 6.0 + 2.0 * static / 15.0) / transition
        scale = (
            static / transition
            + viscous * ratio
            - quadratic * ratio_squared
            - viscous * ratio * ratio_squared / 3.0
            + quartic * ratio_squared * ratio_squared
        )
        radial = (
            viscous * ratio
            - 2.0 * quadratic * ratio_squared
            - viscous * ratio * ratio_squared
            + 4.0 * quartic * ratio_squared * ratio_squared
        )
        return scale, radial

    exponential = math.exp(-(ratio * ratio))
    friction = parameters.dynamic_friction + delta * exponential + viscous * speed
    friction_derivative = viscous - 2.0 * delta * speed * exponential / (
        transition * transition
    )
    tanh_ratio = math.tanh(ratio)
    scale = friction * tanh_ratio / speed
    sech_squared = 1.0 - tanh_ratio * tanh_ratio
    radial = (
        friction_derivative * tanh_ratio + friction * sech_squared / transition - scale
    )
    return scale, radial


def sphere_ground_contact_derivatives(
    center_m: Array,
    velocity_m_s: Array,
    radius_m: float,
    ground: GroundPlane,
    parameters: ContactParameters,
) -> ContactDerivatives:
    """Evaluate contact and its total world-force state Jacobians.

    Matrix columns correspond to world center or velocity components. The
    returned matrices own their storage and are read-only. Inactive branches
    return zero Jacobians. At the penetration and normal-force clipping kinks,
    the function also returns zero Jacobians with ``differentiable=False``;
    off-ground and strictly clipped states remain differentiable. Values within
    eight machine epsilons of either kink, scaled to the relevant state, are
    treated as lying on that boundary to absorb floating-point roundoff.
    """
    sample = sphere_ground_contact(center_m, velocity_m_s, radius_m, ground, parameters)
    center = np.asarray(center_m, dtype=float)
    velocity = np.asarray(velocity_m_s, dtype=float)
    normal = np.asarray(ground.normal, dtype=float)
    zero = np.zeros((3, 3), dtype=float)
    raw_penetration = radius_m + ground.height_m - float(normal @ center)
    penetration_at_kink = _at_roundoff_zero(
        raw_penetration,
        max(abs(radius_m), abs(ground.height_m), float(np.linalg.norm(center))),
    )

    if sample.penetration_m <= 0.0 or penetration_at_kink:
        return ContactDerivatives(
            sample,
            _readonly_matrix(zero),
            _readonly_matrix(zero),
            not penetration_at_kink,
        )

    damping_scale = 1.0 + parameters.dissipation_s_m * sample.penetration_rate_m_s
    normal_magnitude = float(normal @ sample.normal_force_n)
    damping_at_kink = _at_roundoff_zero(
        damping_scale,
        abs(parameters.dissipation_s_m * sample.penetration_rate_m_s),
    )
    if normal_magnitude <= 0.0 or damping_at_kink:
        return ContactDerivatives(
            sample,
            _readonly_matrix(zero),
            _readonly_matrix(zero),
            not damping_at_kink,
        )

    projection = np.eye(3) - np.outer(normal, normal)
    tangential = projection @ velocity
    speed = float(np.linalg.norm(tangential))
    scale, radial = _friction_scale_derivative(speed, parameters)
    force_direction = normal - scale * tangential

    normal_center_gradient = -parameters.stiffness_n_m * damping_scale * normal
    normal_velocity_gradient = (
        -parameters.stiffness_n_m
        * sample.penetration_m
        * parameters.dissipation_s_m
        * normal
    )
    dforce_dcenter = np.outer(force_direction, normal_center_gradient)
    friction_velocity_gradient = scale * projection
    if speed > 0.0:
        tangent_direction = tangential / speed
        friction_velocity_gradient += radial * np.outer(
            tangent_direction, tangent_direction
        )
    dforce_dvelocity = (
        np.outer(force_direction, normal_velocity_gradient)
        - normal_magnitude * friction_velocity_gradient
    )
    return ContactDerivatives(
        sample,
        _readonly_matrix(dforce_dcenter),
        _readonly_matrix(dforce_dvelocity),
        True,
    )
