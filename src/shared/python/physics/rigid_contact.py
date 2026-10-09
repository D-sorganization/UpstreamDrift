"""Finite-inertia two-body point impact with a bounded tangential impulse.

The normal impulse obeys a Newton restitution equality. The impulse minimizes
post-impact kinetic energy within a Coulomb cone. An analytic interior solution
is used when feasible; otherwise the cone boundary is optimized. This
instantaneous law does not resolve the time history
of a compliant ball-face contact or grooves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_MOMENT_OF_INERTIA_KG_M2,
    GOLF_BALL_RADIUS_M,
)


@dataclass(frozen=True)
class RigidContactInput:
    """Pre-impact CG states and contact geometry in one right-handed frame."""

    club_velocity_mps: np.ndarray
    club_angular_velocity_rad_s: np.ndarray
    ball_velocity_mps: np.ndarray
    ball_angular_velocity_rad_s: np.ndarray
    normal: np.ndarray  # Unit vector from club contact toward ball center.
    club_lever_m: np.ndarray  # Club CG to the common contact point.
    club_inertia_world_kg_m2: np.ndarray
    club_mass_kg: float
    restitution: float
    friction_coefficient: float


@dataclass(frozen=True)
class RigidContactResult:
    """Post-impact CG velocities and impulse applied to the ball."""

    club_velocity_mps: np.ndarray
    club_angular_velocity_rad_s: np.ndarray
    ball_velocity_mps: np.ndarray
    ball_angular_velocity_rad_s: np.ndarray
    ball_impulse_n_s: np.ndarray
    pre_relative_normal_mps: float
    post_relative_normal_mps: float
    pre_kinetic_energy_j: float
    post_kinetic_energy_j: float
    friction_regime: str


def _cross_matrix(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _validated_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite three-vector")
    return array


def _validated_inertia(value: np.ndarray) -> np.ndarray:
    inertia = np.asarray(value, dtype=float)
    if (
        inertia.shape != (3, 3)
        or not np.isfinite(inertia).all()
        or not np.allclose(inertia, inertia.T, atol=1e-12, rtol=0)
        or float(np.linalg.eigvalsh(inertia).min()) <= 0
    ):
        raise ValueError("club inertia must be symmetric positive definite")
    return inertia


def _kinetic_energy(
    club_v: np.ndarray,
    club_w: np.ndarray,
    ball_v: np.ndarray,
    ball_w: np.ndarray,
    *,
    club_mass: float,
    club_inertia: np.ndarray,
) -> float:
    return 0.5 * float(
        club_mass * (club_v @ club_v)
        + club_w @ club_inertia @ club_w
        + float(GOLF_BALL_MASS_KG) * (ball_v @ ball_v)
        + float(GOLF_BALL_MOMENT_OF_INERTIA_KG_M2) * (ball_w @ ball_w)
    )


def _sliding_impulse(
    *,
    effective_inverse_mass: np.ndarray,
    relative_velocity: np.ndarray,
    normal: np.ndarray,
    normal_target: float,
    friction: float,
) -> np.ndarray:
    """Find the minimum-energy admissible impulse after sticking fails."""
    k = effective_inverse_mass
    pure_normal = normal * (normal_target / float(normal @ k @ normal))

    def tangential(impulse: np.ndarray) -> np.ndarray:
        return impulse - float(impulse @ normal) * normal

    def cone(impulse: np.ndarray) -> float:
        return friction * float(impulse @ normal) - float(
            np.linalg.norm(tangential(impulse))
        )

    def objective(impulse: np.ndarray) -> float:
        return 0.5 * float(impulse @ k @ impulse) - float(relative_velocity @ impulse)

    def gradient(impulse: np.ndarray) -> np.ndarray:
        return k @ impulse - relative_velocity

    def cone_vector(impulse: np.ndarray) -> np.ndarray:
        return np.array([cone(impulse)], dtype=float)

    solution = minimize(
        objective,
        pure_normal,
        jac=gradient,
        method="SLSQP",
        constraints=(
            LinearConstraint(normal @ k, normal_target, normal_target),
            NonlinearConstraint(cone_vector, 0.0, math.inf),
        ),
        options={"ftol": 1e-12, "maxiter": 150},
    )
    impulse = np.asarray(solution.x, dtype=float)
    if (
        not solution.success
        or not np.isfinite(impulse).all()
        or abs(float(normal @ k @ impulse) - normal_target) > 1e-8
        or cone(impulse) < -1e-8
        or float(impulse @ normal) < -1e-10
    ):
        raise RuntimeError("sliding contact optimization did not converge")
    return impulse


def solve_rigid_contact(state: RigidContactInput) -> RigidContactResult:
    """Conserve both bodies' linear/angular momentum at a shared contact.

    A positive normal relative contact speed means the club approaches the
    ball. The ball contact lever is minus radius times that normal. Purely
    normal impulse is always an admissible fallback for nonnegative friction.
    """
    if not isinstance(state, RigidContactInput):
        raise ValueError("state must be a RigidContactInput")
    vc = _validated_vector(state.club_velocity_mps, "club velocity")
    wc = _validated_vector(state.club_angular_velocity_rad_s, "club angular velocity")
    vb = _validated_vector(state.ball_velocity_mps, "ball velocity")
    wb = _validated_vector(state.ball_angular_velocity_rad_s, "ball angular velocity")
    normal = _validated_vector(state.normal, "contact normal")
    rc = _validated_vector(state.club_lever_m, "club lever")
    inertia = _validated_inertia(state.club_inertia_world_kg_m2)
    if abs(float(normal @ normal) - 1.0) > 1e-10:
        raise ValueError("contact normal must have unit length")
    if (
        not all(
            math.isfinite(v)
            for v in (state.club_mass_kg, state.restitution, state.friction_coefficient)
        )
        or state.club_mass_kg <= 0
        or not 0 <= state.restitution <= 1
        or state.friction_coefficient < 0
    ):
        raise ValueError("mass, restitution, and friction must be physical")

    rb = -float(GOLF_BALL_RADIUS_M) * normal
    relative = vc + np.cross(wc, rc) - vb - np.cross(wb, rb)
    approach = float(relative @ normal)
    if approach <= 0:
        raise ValueError("club and ball must be approaching at contact")

    club_inverse_inertia = np.linalg.inv(inertia)
    ball_inverse_inertia = np.eye(3) / float(GOLF_BALL_MOMENT_OF_INERTIA_KG_M2)
    src = _cross_matrix(rc)
    srb = _cross_matrix(rb)
    effective_inverse_mass = (
        (1.0 / state.club_mass_kg + 1.0 / float(GOLF_BALL_MASS_KG)) * np.eye(3)
        - src @ club_inverse_inertia @ src
        - srb @ ball_inverse_inertia @ srb
    )
    if float(np.linalg.eigvalsh(effective_inverse_mass).min()) <= 0:
        raise RuntimeError("contact inverse mass must be positive definite")
    normal_target = (1.0 + state.restitution) * approach
    # Equality-constrained energy minimizer. A zero-tangential-post-slip
    # impulse is not generally the minimizer when off-center inertia couples
    # normal and tangential response.
    interior_impulse = (
        np.linalg.solve(effective_inverse_mass, relative)
        + state.restitution
        * approach
        / float(normal @ effective_inverse_mass @ normal)
        * normal
    )
    jn = float(interior_impulse @ normal)
    jt = interior_impulse - jn * normal
    if jn >= 0 and float(np.linalg.norm(jt)) <= state.friction_coefficient * jn + 1e-12:
        impulse = interior_impulse
        interior_post_relative = relative - effective_inverse_mass @ impulse
        post_tangent = (
            interior_post_relative - float(interior_post_relative @ normal) * normal
        )
        regime = (
            "stick" if float(np.linalg.norm(post_tangent)) <= 1e-9 else "cone_interior"
        )
    elif state.friction_coefficient == 0:
        impulse = normal * (
            normal_target / float(normal @ effective_inverse_mass @ normal)
        )
        regime = "frictionless"
    else:
        impulse = _sliding_impulse(
            effective_inverse_mass=effective_inverse_mass,
            relative_velocity=relative,
            normal=normal,
            normal_target=normal_target,
            friction=state.friction_coefficient,
        )
        regime = "slide"

    post_vc = vc - impulse / state.club_mass_kg
    post_wc = wc - club_inverse_inertia @ np.cross(rc, impulse)
    post_vb = vb + impulse / float(GOLF_BALL_MASS_KG)
    post_wb = wb + ball_inverse_inertia @ np.cross(rb, impulse)
    post_relative = post_vc + np.cross(post_wc, rc) - post_vb - np.cross(post_wb, rb)
    pre_energy = _kinetic_energy(
        vc, wc, vb, wb, club_mass=state.club_mass_kg, club_inertia=inertia
    )
    post_energy = _kinetic_energy(
        post_vc,
        post_wc,
        post_vb,
        post_wb,
        club_mass=state.club_mass_kg,
        club_inertia=inertia,
    )
    post_normal = float(post_relative @ normal)
    if abs(post_normal + state.restitution * approach) > 1e-8:
        raise RuntimeError("contact restitution postcondition failed")
    if post_energy > pre_energy + 1e-8:
        raise RuntimeError("contact passivity postcondition failed")
    return RigidContactResult(
        club_velocity_mps=post_vc,
        club_angular_velocity_rad_s=post_wc,
        ball_velocity_mps=post_vb,
        ball_angular_velocity_rad_s=post_wb,
        ball_impulse_n_s=impulse,
        pre_relative_normal_mps=approach,
        post_relative_normal_mps=post_normal,
        pre_kinetic_energy_j=pre_energy,
        post_kinetic_energy_j=post_energy,
        friction_regime=regime,
    )
