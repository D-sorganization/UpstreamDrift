"""Generalised pairwise contact law: sphere-capsule and capsule-capsule.

Issue #13300 (HRG-GS-G2).
Provides ContactSphere and ContactCapsule geometric primitives, pairwise distance
queries, and contact wrenches based on the shared Hunt-Crossley / regularised
Coulomb contact law and torsional spin friction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from src.shared.python.grip_contact.bushing_law import BushingState, cross3
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    ContactSample,
    GroundPlane,
    sphere_ground_contact,
    torsional_friction_moment,
)

__all__ = [
    "ContactCapsule",
    "ContactSphere",
    "PairWrench",
    "capsule_capsule",
    "capsule_capsule_distance",
    "capsule_line_nodes",
    "capsule_sphere",
    "closest_point_segment",
    "closest_points_segments",
    "combine_line_nodes",
    "line_contact_node_count",
    "LineNode",
    "sphere_capsule",
    "sphere_capsule_distance",
]


def _finite_vec3(val: np.ndarray, name: str) -> np.ndarray:
    """Validate that val is a finite 1D numpy array of shape (3,).

    Raises:
        TypeError: if val is not an ndarray.
        ValueError: if val is not finite or shape != (3,).
    """
    if not isinstance(val, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(val).__name__}")
    if val.shape != (3,) or not np.isfinite(val).all():
        raise ValueError(f"{name} must be a finite 3-vector, got {val}")
    return val


def _default_state() -> BushingState:
    """Default stationary BushingState at origin with identity orientation."""
    return BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))


def _resolve_state(state: BushingState | None, name: str) -> BushingState:
    """Validate or provide default stationary BushingState."""
    if state is None:
        return _default_state()
    if not isinstance(state, BushingState):
        raise TypeError(f"{name} must be BushingState, got {type(state).__name__}")
    return state


def _to_world(state: BushingState, point: np.ndarray) -> np.ndarray:
    """Transform a point from the body frame of state into world coordinates."""
    return state.position_m + state.rotation @ point


def _point_velocity(state: BushingState, point: np.ndarray) -> np.ndarray:
    """Velocity of a rigid body point in world coordinates."""
    return state.velocity_m_s + cross3(state.omega_rad_s, point - state.position_m)


@dataclass(frozen=True)
class ContactSphere:
    """A spherical contact primitive defined in its body frame.

    ``center_m``: 3-vector center position in body frame.
    ``radius_m``: positive sphere radius.
    """

    center_m: np.ndarray
    radius_m: float

    def __post_init__(self) -> None:
        center = _finite_vec3(self.center_m, "center_m")
        object.__setattr__(self, "center_m", center)
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError(
                f"radius_m must be positive and finite, got {self.radius_m}"
            )


@dataclass(frozen=True)
class ContactCapsule:
    """A capsule contact primitive defined in its body frame.

    ``p0_m``: 3-vector start point of the capsule core segment in body frame.
    ``p1_m``: 3-vector end point of the capsule core segment in body frame.
    ``radius_m``: positive capsule radius.
    """

    p0_m: np.ndarray
    p1_m: np.ndarray
    radius_m: float

    def __post_init__(self) -> None:
        p0 = _finite_vec3(self.p0_m, "p0_m")
        p1 = _finite_vec3(self.p1_m, "p1_m")
        object.__setattr__(self, "p0_m", p0)
        object.__setattr__(self, "p1_m", p1)
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError(
                f"radius_m must be positive and finite, got {self.radius_m}"
            )

    @property
    def start_m(self) -> np.ndarray:
        """Alias for p0_m."""
        return self.p0_m

    @property
    def end_m(self) -> np.ndarray:
        """Alias for p1_m."""
        return self.p1_m


@dataclass(frozen=True)
class PairWrench:
    """Pair wrench between two bodies satisfying action/reaction by construction.

    Forces and moments are expressed in world axes.
    ``force_a_n``: force on body A (world axes).
    ``moment_a_nm``: moment on body A about ``point_a_m``.
    ``force_b_n``: force on body B (world axes), equal to -force_a_n.
    ``moment_b_nm``: moment on body B about ``point_b_m``.
    ``contact_point_m``: contact point in world coordinates.
    ``normal``: unit contact normal pointing from body A toward body B.
    ``penetration_m``: penetration depth in metres (>= 0; 0 when separated).
    ``point_a_m``: origin point about which moment_a_nm is taken.
    ``point_b_m``: origin point about which moment_b_nm is taken.
    ``normal_force_mag_n``: magnitude of normal force (>= 0).
    ``tangential_force_mag_n``: magnitude of friction force (>= 0).
    """

    force_a_n: np.ndarray
    moment_a_nm: np.ndarray
    force_b_n: np.ndarray
    moment_b_nm: np.ndarray
    contact_point_m: np.ndarray
    normal: np.ndarray
    penetration_m: float
    point_a_m: np.ndarray
    point_b_m: np.ndarray
    normal_force_mag_n: float = 0.0
    tangential_force_mag_n: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "force_a_n", _finite_vec3(self.force_a_n, "force_a_n"))
        object.__setattr__(
            self, "moment_a_nm", _finite_vec3(self.moment_a_nm, "moment_a_nm")
        )
        object.__setattr__(self, "force_b_n", _finite_vec3(self.force_b_n, "force_b_n"))
        object.__setattr__(
            self, "moment_b_nm", _finite_vec3(self.moment_b_nm, "moment_b_nm")
        )
        object.__setattr__(
            self,
            "contact_point_m",
            _finite_vec3(self.contact_point_m, "contact_point_m"),
        )
        object.__setattr__(self, "normal", _finite_vec3(self.normal, "normal"))
        object.__setattr__(self, "point_a_m", _finite_vec3(self.point_a_m, "point_a_m"))
        object.__setattr__(self, "point_b_m", _finite_vec3(self.point_b_m, "point_b_m"))
        if not math.isfinite(self.penetration_m) or self.penetration_m < 0.0:
            raise ValueError(
                f"penetration_m must be finite and >= 0, got {self.penetration_m}"
            )
        if not math.isfinite(self.normal_force_mag_n) or self.normal_force_mag_n < 0.0:
            raise ValueError(
                f"normal_force_mag_n must be finite and >= 0, got {self.normal_force_mag_n}"
            )
        if (
            not math.isfinite(self.tangential_force_mag_n)
            or self.tangential_force_mag_n < 0.0
        ):
            raise ValueError(
                f"tangential_force_mag_n must be finite and >= 0, "
                f"got {self.tangential_force_mag_n}"
            )

    @property
    def normal_force_n(self) -> float:
        """Alias for ``normal_force_mag_n``."""
        return self.normal_force_mag_n

    @property
    def tangential_force_n(self) -> float:
        """Alias for ``tangential_force_mag_n``."""
        return self.tangential_force_mag_n

    @property
    def force_n(self) -> np.ndarray:
        """Alias for ``force_b_n`` (force exerted by body A on body B)."""
        return self.force_b_n

    @property
    def moment_nm(self) -> np.ndarray:
        """Alias for ``moment_b_nm`` (moment exerted on body B about ``point_b_m``)."""
        return self.moment_b_nm

    def total_force_n(self) -> np.ndarray:
        """Net force of the pair (identically zero by action/reaction)."""
        return self.force_a_n + self.force_b_n

    def total_moment_about_point(self, point: np.ndarray) -> np.ndarray:
        """Net moment of the pair about ``point`` (zero by action/reaction)."""
        pt = _finite_vec3(point, "point")
        return (
            self.moment_a_nm
            + self.moment_b_nm
            + cross3(self.point_a_m - self.point_b_m, self.force_a_n)
            - cross3(pt, self.force_a_n + self.force_b_n)
        )


def _separated_wrench(
    contact_point: np.ndarray,
    normal: np.ndarray,
    st_a: BushingState,
    st_b: BushingState,
) -> PairWrench:
    """Zero-force PairWrench when bodies are separated."""
    zero = np.zeros(3)
    return PairWrench(
        force_a_n=zero,
        moment_a_nm=zero,
        force_b_n=zero,
        moment_b_nm=zero,
        contact_point_m=contact_point,
        normal=normal,
        penetration_m=0.0,
        point_a_m=st_a.position_m,
        point_b_m=st_b.position_m,
        normal_force_mag_n=0.0,
        tangential_force_mag_n=0.0,
    )


def _check_law_args(
    law: ContactParameters | None,
    patch_radius_m: float,
    spin_transition_rad_s: float,
) -> ContactParameters:
    """Validate shared contact law arguments."""
    if law is None or not isinstance(law, ContactParameters):
        raise TypeError(f"law must be ContactParameters, got {type(law).__name__}")
    if not math.isfinite(patch_radius_m) or patch_radius_m < 0.0:
        raise ValueError(
            f"patch_radius_m must be finite and >= 0, got {patch_radius_m}"
        )
    if not math.isfinite(spin_transition_rad_s) or spin_transition_rad_s <= 0.0:
        raise ValueError(
            f"spin_transition_rad_s must be positive and finite, "
            f"got {spin_transition_rad_s}"
        )
    return law


def _require_sphere_capsule(sphere: object, capsule: object) -> None:
    """Validate that sphere is ContactSphere and capsule is ContactCapsule."""
    if not isinstance(sphere, ContactSphere):
        raise TypeError(f"sphere must be ContactSphere, got {type(sphere).__name__}")
    if not isinstance(capsule, ContactCapsule):
        raise TypeError(f"capsule must be ContactCapsule, got {type(capsule).__name__}")


def _require_capsule_pair(capsule_a: object, capsule_b: object) -> None:
    """Validate that capsule_a and capsule_b are ContactCapsule instances."""
    if not isinstance(capsule_a, ContactCapsule):
        raise TypeError(
            f"capsule_a must be ContactCapsule, got {type(capsule_a).__name__}"
        )
    if not isinstance(capsule_b, ContactCapsule):
        raise TypeError(
            f"capsule_b must be ContactCapsule, got {type(capsule_b).__name__}"
        )


def _contact_plane(normal: np.ndarray, surface_point: np.ndarray) -> GroundPlane:
    """Construct a GroundPlane from unit normal and surface point."""
    return GroundPlane(
        (float(normal[0]), float(normal[1]), float(normal[2])),
        float(normal @ surface_point),
    )


def _sample_contact(
    sphere_center: np.ndarray,
    rel_vel: np.ndarray,
    sphere_radius: float,
    plane: GroundPlane,
    rel_omega: np.ndarray,
    law: ContactParameters,
    patch_radius_m: float,
    spin_transition_rad_s: float,
) -> tuple[ContactSample, np.ndarray]:
    """Sample Hunt-Crossley / Coulomb contact and torsional spin moment."""
    sample = sphere_ground_contact(sphere_center, rel_vel, sphere_radius, plane, law)
    spin = torsional_friction_moment(
        sample.normal_force_n,
        rel_omega,
        plane,
        patch_radius_m=patch_radius_m,
        friction=law.static_friction,
        transition_rad_s=spin_transition_rad_s,
    )
    return sample, spin


def _pair_wrench_from_forces(
    force_a: np.ndarray,
    spin_a: np.ndarray,
    contact_point: np.ndarray,
    normal: np.ndarray,
    penetration: float,
    st_a: BushingState,
    st_b: BushingState,
    sample: ContactSample,
) -> PairWrench:
    """Build a PairWrench from body A force/spin and contact sample."""
    force_b = -force_a
    moment_a = cross3(contact_point - st_a.position_m, force_a) + spin_a
    moment_b = cross3(contact_point - st_b.position_m, force_b) - spin_a
    f_n = float(np.linalg.norm(sample.normal_force_n))
    f_t = float(np.linalg.norm(sample.friction_force_n))
    return PairWrench(
        force_a_n=force_a,
        moment_a_nm=moment_a,
        force_b_n=force_b,
        moment_b_nm=moment_b,
        contact_point_m=contact_point,
        normal=normal,
        penetration_m=penetration,
        point_a_m=st_a.position_m,
        point_b_m=st_b.position_m,
        normal_force_mag_n=f_n,
        tangential_force_mag_n=f_t,
    )


def closest_points_segments(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Find closest points between two line segments [a0, a1] and [b0, b1].

    Preconditions:
        a0, a1, b0, b1 must be finite 3-vectors.

    Postconditions:
        Returns (pa, pb) on [a0, a1] and [b0, b1] respectively, minimizing ||pa - pb||.
        For parallel or coincident segments, returns a documented finite pair
        at the center of their mutual projection overlap.

    The minimum distance is continuous in the endpoints, but the returned
    points are not: when ``sin^2(theta) < 1e-10`` they sit at the overlap
    centre, and just past it at an endpoint.  Use it for distances and
    gaps; the capsule contact law (:func:`capsule_capsule`) integrates along
    the axis instead (#13308).
    """
    p_a0 = _finite_vec3(a0, "a0")
    p_a1 = _finite_vec3(a1, "a1")
    p_b0 = _finite_vec3(b0, "b0")
    p_b1 = _finite_vec3(b1, "b1")

    d1 = p_a1 - p_a0
    d2 = p_b1 - p_b0
    r = p_a0 - p_b0

    a = float(d1 @ d1)
    e = float(d2 @ d2)
    f = float(d2 @ r)

    eps = 1e-14

    if a < eps and e < eps:
        return p_a0.copy(), p_b0.copy()

    if a < eps:
        t = float(np.clip(f / e, 0.0, 1.0))
        return p_a0.copy(), p_b0 + t * d2

    c = float(d1 @ r)
    if e < eps:
        s = float(np.clip(-c / a, 0.0, 1.0))
        return p_a0 + s * d1, p_b0.copy()

    b = float(d1 @ d2)
    denom = a * e - b * b
    if denom < 0.0:
        denom = 0.0

    if denom < 1e-10 * a * e:
        s0 = -c / a
        s1 = (-c + b) / a
        s_min, s_max = min(s0, s1), max(s0, s1)
        overlap_min = max(0.0, s_min)
        overlap_max = min(1.0, s_max)
        if overlap_min <= overlap_max:
            s = 0.5 * (overlap_min + overlap_max)
            t = float(np.clip((a * s + c) / b, 0.0, 1.0))
        else:
            if s_max < 0.0:
                s = 0.0
                t = float(np.clip(f / e, 0.0, 1.0))
            else:
                s = 1.0
                t = float(np.clip((b + f) / e, 0.0, 1.0))
    else:
        s = (b * f - c * e) / denom
        t = (b * s + f) / e
        if s < 0.0:
            s = 0.0
            t = float(np.clip(f / e, 0.0, 1.0))
        elif s > 1.0:
            s = 1.0
            t = float(np.clip((b + f) / e, 0.0, 1.0))
        else:
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))

    return p_a0 + s * d1, p_b0 + t * d2


def closest_point_segment(
    point: np.ndarray, p0: np.ndarray, p1: np.ndarray
) -> tuple[np.ndarray, float]:
    """Find closest point on line segment [p0, p1] to point.

    Preconditions:
        point, p0, p1 must be finite 3-vectors.

    Postconditions:
        Returns (closest_point, t) with t in [0.0, 1.0].
    """
    pt = _finite_vec3(point, "point")
    a0 = _finite_vec3(p0, "p0")
    a1 = _finite_vec3(p1, "p1")
    d = a1 - a0
    d_sq = float(d @ d)
    if d_sq < 1e-14:
        return a0.copy(), 0.0
    t = float(np.clip((pt - a0) @ d / d_sq, 0.0, 1.0))
    return a0 + t * d, t


def _orthogonal_unit_vector(d: np.ndarray) -> np.ndarray:
    """Return a deterministic unit vector orthogonal to d."""
    norm_d = float(np.linalg.norm(d))
    if norm_d < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    u = d / norm_d
    if abs(u[2]) < 0.9:
        v = cross3(u, np.array([0.0, 0.0, 1.0]))
    else:
        v = cross3(u, np.array([1.0, 0.0, 0.0]))
    norm_v = float(np.linalg.norm(v))
    return v / norm_v if norm_v > 1e-12 else np.array([1.0, 0.0, 0.0])


def sphere_capsule_distance(
    sphere: ContactSphere,
    capsule: ContactCapsule,
    state_a: BushingState | None = None,
    state_b: BushingState | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Signed distance and geometry between sphere (body A) and capsule (body B).

    Preconditions:
        sphere: ContactSphere instance.
        capsule: ContactCapsule instance.
        state_a, state_b: optional BushingState for body frames.

    Postconditions:
        Returns (signed_distance_m, contact_point_m, normal, closest_on_axis_m).
        signed_distance_m < 0 indicates penetration.
        normal is a unit vector pointing from body A (sphere) toward body B (capsule).
        contact_point_m is the deepest point of the sphere toward the capsule.
    """
    _require_sphere_capsule(sphere, capsule)
    st_a = _resolve_state(state_a, "state_a")
    st_b = _resolve_state(state_b, "state_b")

    center_w = _to_world(st_a, sphere.center_m)
    b0_w = _to_world(st_b, capsule.p0_m)
    b1_w = _to_world(st_b, capsule.p1_m)

    p_axis, _ = closest_point_segment(center_w, b0_w, b1_w)
    diff = p_axis - center_w
    dist = float(np.linalg.norm(diff))

    normal = diff / dist if dist > 1e-12 else _orthogonal_unit_vector(b1_w - b0_w)
    signed_dist = dist - sphere.radius_m - capsule.radius_m
    contact_point = center_w + sphere.radius_m * normal
    return signed_dist, contact_point, normal, p_axis


def sphere_capsule(
    sphere: ContactSphere,
    capsule: ContactCapsule,
    state_a: BushingState | None = None,
    state_b: BushingState | None = None,
    law: ContactParameters | None = None,
    patch_radius_m: float = 0.0,
    spin_transition_rad_s: float = 1.0,
) -> PairWrench:
    """Shared contact law between a sphere (body A) and a capsule (body B).

    Reuses the Hunt-Crossley normal force, regularised Coulomb/Stribeck friction
    and torsional spin friction moment from motion_matching.contact_law.

    Preconditions:
        sphere: ContactSphere instance.
        capsule: ContactCapsule instance.
        law must be ContactParameters.
        patch_radius_m >= 0.0 and finite.
        spin_transition_rad_s > 0.0 and finite.

    Postconditions:
        Returns PairWrench satisfying action/reaction by construction.
        Zero force and moment when separated (gap >= 0.0).
        Normal force is strictly unilateral / compressive (never pulling).
    """
    _require_sphere_capsule(sphere, capsule)
    law_params = _check_law_args(law, patch_radius_m, spin_transition_rad_s)
    st_a = _resolve_state(state_a, "state_a")
    st_b = _resolve_state(state_b, "state_b")

    signed_dist, contact_point, normal, p_axis = sphere_capsule_distance(
        sphere, capsule, st_a, st_b
    )
    penetration = max(0.0, -signed_dist)
    if penetration <= 0.0:
        return _separated_wrench(contact_point, normal, st_a, st_b)

    center_w = _to_world(st_a, sphere.center_m)
    v_a = _point_velocity(st_a, contact_point)
    v_b = _point_velocity(st_b, contact_point)

    plane = _contact_plane(-normal, p_axis - capsule.radius_m * normal)
    sample, spin_a = _sample_contact(
        center_w,
        v_a - v_b,
        sphere.radius_m,
        plane,
        st_a.omega_rad_s - st_b.omega_rad_s,
        law_params,
        patch_radius_m,
        spin_transition_rad_s,
    )
    force_a = sample.normal_force_n + sample.friction_force_n
    return _pair_wrench_from_forces(
        force_a, spin_a, contact_point, normal, penetration, st_a, st_b, sample
    )


def capsule_capsule_distance(
    capsule_a: ContactCapsule,
    capsule_b: ContactCapsule,
    state_a: BushingState | None = None,
    state_b: BushingState | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Signed distance and geometry between two capsules (body A and body B).

    Preconditions:
        capsule_a, capsule_b: ContactCapsule instances.
        state_a, state_b: optional BushingState for body frames.

    Postconditions:
        Returns (signed_distance_m, contact_point_m, normal, pa_m, pb_m).
        signed_distance_m < 0 indicates penetration.
        normal is a unit vector pointing from body A toward body B.
        contact_point_m is the midpoint of the surface overlap at the closest
        points, which jump for nearly parallel capsules (see
        :func:`closest_points_segments`); :func:`capsule_capsule` does not use
        it to apply force.
    """
    _require_capsule_pair(capsule_a, capsule_b)
    st_a = _resolve_state(state_a, "state_a")
    st_b = _resolve_state(state_b, "state_b")

    a0_w = _to_world(st_a, capsule_a.p0_m)
    a1_w = _to_world(st_a, capsule_a.p1_m)
    b0_w = _to_world(st_b, capsule_b.p0_m)
    b1_w = _to_world(st_b, capsule_b.p1_m)

    pa, pb = closest_points_segments(a0_w, a1_w, b0_w, b1_w)
    diff = pb - pa
    dist = float(np.linalg.norm(diff))

    normal = diff / dist if dist > 1e-12 else _orthogonal_unit_vector(a1_w - a0_w)
    signed_dist = dist - capsule_a.radius_m - capsule_b.radius_m
    sa = pa + capsule_a.radius_m * normal
    sb = pb - capsule_b.radius_m * normal
    contact_point = 0.5 * (sa + sb)
    return signed_dist, contact_point, normal, pa, pb


def line_contact_node_count(half_length_m: float, radius_m: float) -> int:
    """Number of axial nodes of a capsule's line-contact quadrature.

    The single shared rule for every engine that discretises a capsule (the
    OpenSim sphere chain re-exports it as ``capsule_sphere_count``).

    Preconditions:
        ``half_length_m`` must be a nonnegative finite float.
        ``radius_m`` must be a positive finite float.

    Postconditions:
        Returns an integer >= 2 whose node spacing along the core segment is
        about ``radius_m / 2`` (``round(2 L / (r / 2)) + 1`` nodes).
    """
    if not (math.isfinite(half_length_m) and half_length_m >= 0.0):
        raise ValueError(
            f"half_length_m must be nonnegative and finite, got {half_length_m}"
        )
    if not (math.isfinite(radius_m) and radius_m > 0.0):
        raise ValueError(f"radius_m must be positive and finite, got {radius_m}")
    core_len = 2.0 * half_length_m
    return max(2, int(round(core_len / (0.5 * radius_m))) + 1)


def capsule_line_nodes(capsule: ContactCapsule) -> tuple[ContactSphere, ...]:
    """Body-frame node spheres of ``capsule``'s axial line-contact quadrature.

    Preconditions:
        ``capsule`` is a :class:`ContactCapsule`.

    Postconditions:
        Returns ``n = line_contact_node_count(L / 2, r)`` spheres of radius
        ``r`` centred at ``p0 + (k / (n - 1)) (p1 - p0)``, ``k = 0..n-1``: the
        first at ``p0``, the last at ``p1``, equally spaced.
    """
    if not isinstance(capsule, ContactCapsule):
        raise TypeError(f"capsule must be ContactCapsule, got {type(capsule).__name__}")
    axis = capsule.p1_m - capsule.p0_m
    half_length = 0.5 * math.sqrt(_dot3(axis.tolist(), axis.tolist()))
    count = line_contact_node_count(half_length, capsule.radius_m)
    return tuple(
        ContactSphere(capsule.p0_m + (k / (count - 1)) * axis, capsule.radius_m)
        for k in range(count)
    )


Vec3 = tuple[float, float, float]


class LineNode(NamedTuple):
    """One axial node's full-stiffness wrench, as Python floats (world axes).

    Moments are about the pair's ``point_a_m`` (``moment_a_nm``) and
    ``point_b_m`` (``moment_b_nm``); ``normal`` points from A toward B.
    """

    force_a_n: Vec3
    moment_a_nm: Vec3
    force_b_n: Vec3
    moment_b_nm: Vec3
    contact_point_m: Vec3
    normal: Vec3
    penetration_m: float
    normal_force_mag_n: float
    tangential_force_mag_n: float


def _dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _unit3(v: Sequence[float], fallback: Sequence[float]) -> Vec3:
    norm = math.sqrt(_dot3(v, v))
    if norm > 1e-300:
        return (v[0] / norm, v[1] / norm, v[2] / norm)
    return (fallback[0], fallback[1], fallback[2])


def combine_line_nodes(nodes: Sequence[LineNode]) -> LineNode:
    """Resultant of a capsule's axial nodes, each weighted ``1 / n``.

    The penalty load is integrated along the capsule axis with ``n`` equally
    weighted nodes: ``F = (1/n) sum_k F_k``, where ``F_k`` is the
    full-stiffness sphere-capsule wrench of node ``k``.  The law is linear in
    stiffness, so this is the line contact of per-length stiffness ``k / L``.
    It is continuous in the pose because it never selects a closest point.

    Preconditions:
        ``nodes`` holds at least one :class:`LineNode`.

    Postconditions:
        Forces and moments are the left-to-right weighted sums (action and
        reaction hold node by node, so they hold for the sum); the
        ``normal_force_mag_n`` and ``tangential_force_mag_n`` values are the
        weighted sums of the node magnitudes (the line load); ``penetration_m``
        is the deepest node's; ``contact_point_m`` is the normal-force-weighted
        node centroid and ``normal`` the unit normal-force resultant direction.
        With zero total normal force they fall back to the unweighted node
        centroid and mean normal.
    """
    if not nodes:
        raise ValueError("combine_line_nodes needs at least one node")
    w = 1.0 / len(nodes)
    fa = ma = fb = mb = cp = nsum = cp_mean = n_mean = (0.0, 0.0, 0.0)
    f_n = f_t = pen = 0.0
    for node in nodes:
        fa = _axpy(w, node.force_a_n, fa)
        ma = _axpy(w, node.moment_a_nm, ma)
        fb = _axpy(w, node.force_b_n, fb)
        mb = _axpy(w, node.moment_b_nm, mb)
        load = w * node.normal_force_mag_n
        f_n += load
        f_t += w * node.tangential_force_mag_n
        cp = _axpy(load, node.contact_point_m, cp)
        nsum = _axpy(load, node.normal, nsum)
        cp_mean = _axpy(w, node.contact_point_m, cp_mean)
        n_mean = _axpy(w, node.normal, n_mean)
        pen = max(pen, node.penetration_m)
    if f_n > 0.0:
        point = (cp[0] / f_n, cp[1] / f_n, cp[2] / f_n)
        normal = _unit3(nsum, _unit3(n_mean, nodes[0].normal))
    else:
        point = cp_mean
        normal = _unit3(n_mean, nodes[0].normal)
    return LineNode(fa, ma, fb, mb, point, normal, pen, f_n, f_t)


def _axpy(a: float, x: Sequence[float], y: Sequence[float]) -> Vec3:
    """``y + a * x`` on 3-vectors of Python floats."""
    return (y[0] + a * x[0], y[1] + a * x[1], y[2] + a * x[2])


def _line_node(pw: PairWrench) -> LineNode:
    return LineNode(
        _as_vec3(pw.force_a_n),
        _as_vec3(pw.moment_a_nm),
        _as_vec3(pw.force_b_n),
        _as_vec3(pw.moment_b_nm),
        _as_vec3(pw.contact_point_m),
        _as_vec3(pw.normal),
        pw.penetration_m,
        pw.normal_force_mag_n,
        pw.tangential_force_mag_n,
    )


def _as_vec3(v: np.ndarray) -> Vec3:
    a, b, c = v.tolist()
    return (a, b, c)


def _capsule_core_length_m(capsule: ContactCapsule) -> float:
    axis = capsule.p1_m - capsule.p0_m
    return float(math.sqrt(_dot3(axis.tolist(), axis.tolist())))


def _capsule_line_quadrature_key(
    capsule: ContactCapsule,
) -> tuple[float, float, Vec3, Vec3]:
    """Deterministic ordering key for symmetric line-contact integration (#13508)."""
    p0 = _as_vec3(np.asarray(capsule.p0_m, float))
    p1 = _as_vec3(np.asarray(capsule.p1_m, float))
    return (_capsule_core_length_m(capsule), capsule.radius_m, p0, p1)


def _canonical_line_contact_integrator(
    capsule_a: ContactCapsule, capsule_b: ContactCapsule
) -> tuple[ContactCapsule, ContactCapsule, bool]:
    """Pick the capsule whose axis is discretised for :func:`capsule_capsule`.

    The shorter core segment is integrated so a long grip does not dilute the
    load when argument order swaps (#13508).  Equal lengths break ties by
    :func:`_capsule_line_quadrature_key` so ``(A,B)`` and ``(B,A)`` share the
    same integrator.

    Postconditions:
        Returns ``(integrator, other, integrator_is_a)`` with
        ``integrator_is_a`` true when ``integrator`` is ``capsule_a``.
    """
    key_a = _capsule_line_quadrature_key(capsule_a)
    key_b = _capsule_line_quadrature_key(capsule_b)
    if key_a <= key_b:
        return capsule_a, capsule_b, True
    return capsule_b, capsule_a, False


def _line_contact_from_integrator(
    integrator: ContactCapsule,
    other: ContactCapsule,
    integrator_is_a: bool,
    st_a: BushingState,
    st_b: BushingState,
    law_params: ContactParameters,
    patch_radius_m: float,
    spin_transition_rad_s: float,
) -> LineNode:
    st_integ = st_a if integrator_is_a else st_b
    st_other = st_b if integrator_is_a else st_a
    nodes = [
        _line_node(
            sphere_capsule(
                node,
                other,
                st_integ,
                st_other,
                law_params,
                patch_radius_m,
                spin_transition_rad_s,
            )
        )
        for node in capsule_line_nodes(integrator)
    ]
    total = combine_line_nodes(nodes)
    if integrator_is_a:
        return total
    return LineNode(
        total.force_b_n,
        total.moment_b_nm,
        total.force_a_n,
        total.moment_a_nm,
        total.contact_point_m,
        (-total.normal[0], -total.normal[1], -total.normal[2]),
        total.penetration_m,
        total.normal_force_mag_n,
        total.tangential_force_mag_n,
    )


def capsule_capsule(
    capsule_a: ContactCapsule,
    capsule_b: ContactCapsule,
    state_a: BushingState | None = None,
    state_b: BushingState | None = None,
    law: ContactParameters | None = None,
    patch_radius_m: float = 0.0,
    spin_transition_rad_s: float = 1.0,
) -> PairWrench:
    """Shared line-contact law between two capsules (body A and body B) (#13308).

    The shorter capsule's axis is discretised into :func:`capsule_line_nodes`
    (ties broken deterministically when lengths match, #13508); each node
    sphere meets the other capsule through :func:`sphere_capsule` (Hunt-Crossley
    normal, regularised Coulomb/Stribeck friction and torsional spin friction
    from motion_matching.contact_law) and :func:`combine_line_nodes` weights
    every node ``1 / n``.  The law integrates the penetration along the
    overlap instead of selecting a closest point, so it is continuous for
    nearly parallel capsules, where the closest point of
    :func:`closest_points_segments` jumps from the overlap centre to an
    endpoint.  For exactly parallel capsules with a uniform depth ``delta``
    it equals the closest-point law: normal force ``k delta (1 + d ddelta)``
    at the overlap centre.

    Preconditions:
        capsule_a, capsule_b: ContactCapsule instances.
        law must be ContactParameters.
        patch_radius_m >= 0.0 and finite.
        spin_transition_rad_s > 0.0 and finite.

    Postconditions:
        Returns PairWrench satisfying action/reaction by construction.
        Zero force and moment when no node penetrates (``penetration_m == 0``).
        Normal force is strictly unilateral / compressive (never pulling).
        The wrench and contact point are continuous in the poses wherever the
        normal force is positive.
    """
    _require_capsule_pair(capsule_a, capsule_b)
    law_params = _check_law_args(law, patch_radius_m, spin_transition_rad_s)
    st_a = _resolve_state(state_a, "state_a")
    st_b = _resolve_state(state_b, "state_b")
    integrator, other, integrator_is_a = _canonical_line_contact_integrator(
        capsule_a, capsule_b
    )
    total = _line_contact_from_integrator(
        integrator,
        other,
        integrator_is_a,
        st_a,
        st_b,
        law_params,
        patch_radius_m,
        spin_transition_rad_s,
    )
    return PairWrench(
        force_a_n=np.array(total.force_a_n),
        moment_a_nm=np.array(total.moment_a_nm),
        force_b_n=np.array(total.force_b_n),
        moment_b_nm=np.array(total.moment_b_nm),
        contact_point_m=np.array(total.contact_point_m),
        normal=np.array(total.normal),
        penetration_m=total.penetration_m,
        point_a_m=st_a.position_m,
        point_b_m=st_b.position_m,
        normal_force_mag_n=total.normal_force_mag_n,
        tangential_force_mag_n=total.tangential_force_mag_n,
    )


def capsule_sphere(
    capsule: ContactCapsule,
    sphere: ContactSphere,
    state_a: BushingState | None = None,
    state_b: BushingState | None = None,
    law: ContactParameters | None = None,
    patch_radius_m: float = 0.0,
    spin_transition_rad_s: float = 1.0,
) -> PairWrench:
    """Capsule (body A) and sphere (body B) contact; delegates to sphere_capsule."""
    if not isinstance(capsule, ContactCapsule):
        raise TypeError(f"capsule must be ContactCapsule, got {type(capsule).__name__}")
    if not isinstance(sphere, ContactSphere):
        raise TypeError(f"sphere must be ContactSphere, got {type(sphere).__name__}")
    w = sphere_capsule(
        sphere,
        capsule,
        state_a=state_b,
        state_b=state_a,
        law=law,
        patch_radius_m=patch_radius_m,
        spin_transition_rad_s=spin_transition_rad_s,
    )
    return PairWrench(
        force_a_n=w.force_b_n,
        moment_a_nm=w.moment_b_nm,
        force_b_n=w.force_a_n,
        moment_b_nm=w.moment_a_nm,
        contact_point_m=w.contact_point_m,
        normal=-w.normal,
        penetration_m=w.penetration_m,
        point_a_m=w.point_b_m,
        point_b_m=w.point_a_m,
        normal_force_mag_n=w.normal_force_mag_n,
        tangential_force_mag_n=w.tangential_force_mag_n,
    )
