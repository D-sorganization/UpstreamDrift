"""Differentiable JAX contact law and grip weld for full-body models (epic #11006, #11037).

Provides pure JAX implementations of:
1. Compliant Hunt-Crossley normal ground contact with regularised Coulomb friction,
   matching ``src.shared.python.motion_matching.contact_law.sphere_ground_contact``.
2. Equal-and-opposite weld spring-damper spatial reactions between two sites,
   supporting differentiable closed-chain grip constraints.

This module intentionally imports JAX at module top and is NOT imported by
``src/shared/python/motion_matching/__init__.py`` so that the rest of the package
remains importable in environments without JAX.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.shared.python.core.contracts import require
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
)


class SiteState(NamedTuple):
    """World-frame state of one weld site.

    Fields:
        p: 3-vector site position in world frame (m).
        v: 3-vector site linear velocity in world frame (m/s).
        r: 3x3 rotation matrix from site frame to world frame.
        w: 3-vector site angular velocity in world frame (rad/s).
        com: 3-vector body centre of mass in world frame (m).
    """

    p: jax.Array
    v: jax.Array
    r: jax.Array
    w: jax.Array
    com: jax.Array


class WeldGains(NamedTuple):
    """Gains for the weld spring-damper reaction.

    Fields:
        k: Linear stiffness in N/m (must be positive).
        c: Linear damping in N*s/m (must be nonnegative).
        rot_k: Rotational stiffness in N*m/rad (must be positive).
        rot_c: Rotational damping in N*m*s/rad (must be nonnegative).
    """

    k: float
    c: float
    rot_k: float
    rot_c: float


def sphere_ground_contact_jax(
    center: jax.Array,
    velocity: jax.Array,
    radius_m: float,
    ground: GroundPlane,
    parameters: ContactParameters,
    *,
    speed_floor_m_s: float = 1e-9,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate differentiable sphere-ground contact forces in JAX.

    Matches the Hunt-Crossley normal compliance and regularised Coulomb
    friction law from ``contact_law.sphere_ground_contact``, using a small
    speed floor to maintain finite gradients at zero tangential velocity.

    Args:
        center: 3-vector sphere centre position in world coordinates (m).
        velocity: 3-vector sphere linear velocity in world coordinates (m/s).
        radius_m: Sphere radius in metres (must be positive).
        ground: Ground plane specification.
        parameters: Shared contact parameters.
        speed_floor_m_s: Smoothing term under tangential speed sqrt (m/s).

    Returns:
        Tuple of (normal_force, friction_force) as 3-vectors in world frame (N).
    """
    require(radius_m > 0.0, "Sphere radius must be positive", radius_m)
    require(speed_floor_m_s > 0.0, "Speed floor must be positive", speed_floor_m_s)

    c = jnp.asarray(center)
    v = jnp.asarray(velocity)
    n_ground = jnp.asarray(ground.normal, dtype=c.dtype)

    signed_distance = jnp.dot(n_ground, c) - ground.height_m - radius_m
    penetration = jnp.maximum(0.0, -signed_distance)
    rate = -jnp.dot(n_ground, v)

    raw_magnitude = (
        parameters.stiffness_n_m
        * penetration
        * (1.0 + parameters.dissipation_s_m * rate)
    )
    magnitude = jnp.maximum(0.0, raw_magnitude)
    has_contact = penetration > 0.0
    magnitude = jnp.where(has_contact, magnitude, 0.0)

    tangential = v - n_ground * jnp.dot(n_ground, v)
    speed = jnp.sqrt(jnp.dot(tangential, tangential) + speed_floor_m_s**2)
    ratio = speed / parameters.transition_velocity_m_s
    mu = (
        parameters.dynamic_friction
        + (parameters.static_friction - parameters.dynamic_friction)
        * jnp.exp(-(ratio**2))
        + parameters.viscous_friction * speed
    )
    friction = -tangential / speed * mu * magnitude * jnp.tanh(ratio)

    zero_vec = jnp.zeros_like(c)
    normal_force = jnp.where(has_contact, n_ground * magnitude, zero_vec)
    friction_force = jnp.where(has_contact, friction, zero_vec)

    return normal_force, friction_force


def weld_wrench_jax(
    a: SiteState,
    b: SiteState,
    gains: WeldGains,
) -> tuple[jax.Array, jax.Array]:
    """Compute equal and opposite spatial reaction wrenches between two sites.

    Computes the spring-damper reaction holding closure site b on site a.
    Returns (wrench_a, wrench_b) where each is a 6-vector (force, torque at CoM).

    Args:
        a: State of weld site A on body A.
        b: State of weld site B on body B.
        gains: Spring-damper gains for linear and rotational closure.

    Returns:
        Tuple of (wrench_a, wrench_b), each a 6-vector [fx, fy, fz, tx, ty, tz]
        where the torque is transferred to the respective body centre of mass.
    """
    require(gains.k > 0.0, "Linear stiffness k must be positive", gains.k)
    require(
        gains.rot_k > 0.0, "Rotational stiffness rot_k must be positive", gains.rot_k
    )
    require(gains.c >= 0.0, "Linear damping c must be nonnegative", gains.c)
    require(
        gains.rot_c >= 0.0, "Rotational damping rot_c must be nonnegative", gains.rot_c
    )

    rel = a.r @ b.r.T  # rotation taking b's axes onto a's
    rotvec = 0.5 * jnp.array(
        [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]]
    )
    force_on_b = gains.k * (a.p - b.p) + gains.c * (a.v - b.v)
    torque_on_b = gains.rot_k * rotvec + gains.rot_c * (a.w - b.w)

    lever_b = b.p - b.com
    lever_a = a.p - a.com
    wrench_b = jnp.concatenate(
        [force_on_b, torque_on_b + jnp.cross(lever_b, force_on_b)]
    )
    wrench_a = jnp.concatenate(
        [-force_on_b, -torque_on_b + jnp.cross(lever_a, -force_on_b)]
    )
    return wrench_a, wrench_b
