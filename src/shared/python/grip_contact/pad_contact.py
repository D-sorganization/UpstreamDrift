"""Shared sphere-on-cylinder pad contact law (issue #11739, OSV-7 phase 3).

The ``contact`` grip evaluated analytically: spherical pads fixed on a hand
frame press on the club's cylindrical grip.  Each pad is a sphere against the
tangent plane of the cylinder at its nearest point, so the existing shared
law of :mod:`src.shared.python.motion_matching.contact_law` (Hunt-Crossley
normal force, regularised Coulomb friction with a Stribeck blend) is reused
unchanged, plus its spin (pivot) friction moment for the contact patch.

Used directly by the Pinocchio engine (no native contact) and as the reference
for the static checks of the MuJoCo and Drake native contacts.

Conventions.  ``hand`` and ``club`` are :class:`BushingState` grip frames in the
world (``R`` world <- frame, origin velocity, angular velocity).  Forces and
moments are ON THE CLUB; the moment is about the club grip-frame origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.shared.python.grip_contact.bushing_law import BushingState, cross3
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.pad_layout import PadLayout, matched_pad_parameters
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
    sphere_ground_contact,
    torsional_friction_moment,
)

__all__ = [
    "ContactCapsule",
    "ContactSphere",
    "GripCylinder",
    "PadContactModel",
    "PadWrench",
    "PairWrench",
    "build_pad_model",
    "capsule_capsule",
    "capsule_capsule_distance",
    "capsule_sphere",
    "closest_point_segment",
    "closest_points_segments",
    "grip_cylinder",
    "pad_wrench",
    "sphere_capsule",
    "sphere_capsule_distance",
]


@dataclass(frozen=True)
class GripCylinder:
    """The grip cylinder on the club: radius and axial range.

    ``axial_range_m`` is in coordinates along the grip axis measured from the
    *right* grip frame origin; ``left_axial_m`` is the left grip frame's
    coordinate on that axis.
    """

    radius_m: float
    axial_range_m: tuple[float, float]
    left_axial_m: float

    def __post_init__(self) -> None:
        lo, hi = self.axial_range_m
        if self.radius_m <= 0.0 or not hi > lo:
            raise ValueError("cylinder needs a positive radius and hi > lo")

    def origin_axial_m(self, side: str) -> float:
        """Axial coordinate of ``side``'s grip frame origin."""
        return self.left_axial_m if side == "L" else 0.0


def grip_cylinder(
    interface: GripInterface, layout: PadLayout, margin_m: float = 0.08
) -> GripCylinder:
    """Cylinder through both grip points, extended ``margin_m`` beyond them.

    Raises:
        ValueError: if the two grip origins are not ``2 * grip_radius`` apart
            across a common grip axis (the layout would not sit on the shaft).
    """
    rot = np.asarray(interface.right.rotation, dtype=float)
    offset = rot.T @ (
        np.asarray(interface.left.position_m) - np.asarray(interface.right.position_m)
    )
    across = np.array([0.0, 0.0, 2.0 * layout.grip_radius_m])
    if not np.allclose(offset[1:], across[1:], atol=1e-6):
        raise ValueError(
            f"grip origins must be {across[2]:.4f} m apart across the grip "
            f"(got {offset[1]:.4f}, {offset[2]:.4f})"
        )
    left = float(offset[0])
    return GripCylinder(
        layout.grip_radius_m,
        (min(0.0, left) - margin_m, max(0.0, left) + margin_m),
        left,
    )


@dataclass(frozen=True)
class PadContactModel:
    """Pad layout, shared contact-law parameters and the patch for spin friction."""

    layout: PadLayout
    law: ContactParameters
    cylinder: GripCylinder
    patch_radius_m: float
    spin_transition_rad_s: float = 1.0

    def __post_init__(self) -> None:
        if self.patch_radius_m < 0.0 or self.spin_transition_rad_s <= 0.0:
            raise ValueError("patch radius >= 0 and spin transition > 0 required")


def grip_axis(
    interface: GripInterface, pads: PadContactModel
) -> tuple[np.ndarray, np.ndarray]:
    """World grip-cylinder axis direction and the axis point nearest the right hand.

    Postcondition: the direction is a unit vector (the right grip frame's x axis).
    """
    rot_r = np.asarray(interface.right.rotation, dtype=float)
    axis = rot_r[:, 0]
    point = np.asarray(interface.right.position_m) + rot_r @ (
        pads.layout.axis_offset_grip_frame("R")
    )
    return axis, point


#: Friction transition speed [m/s] of the regularised Coulomb law.
FRICTION_TRANSITION_M_S = 1.0e-3


def build_pad_model(
    interface: GripInterface,
    squeeze_per_hand_n: float,
    layout: PadLayout | None = None,
    friction_transition_m_s: float = FRICTION_TRANSITION_M_S,
) -> PadContactModel:
    """Pad model matched to ``interface.bushing`` with the requested squeeze.

    The contact patch radius for spin friction is the Hertz-like half width
    ``sqrt(2 r_eff delta0)`` of a pad sphere pressed ``delta0`` into the
    cylinder, ``1 / r_eff = 1 / r_pad + 1 / r_grip``.
    """
    base = layout or PadLayout()
    matched = matched_pad_parameters(interface.bushing, base, squeeze_per_hand_n)
    layout = replace(base, preload_penetration_m=matched.preload_penetration_m)
    material = interface.contact_material
    law = ContactParameters(
        stiffness_n_m=matched.stiffness_n_m,
        dissipation_s_m=matched.dissipation_s_m,
        static_friction=material.static_friction,
        dynamic_friction=material.dynamic_friction,
        viscous_friction=material.viscous_friction,
        transition_velocity_m_s=friction_transition_m_s,
    )
    r_eff = 1.0 / (1.0 / layout.pad_radius_m + 1.0 / layout.grip_radius_m)
    patch = math.sqrt(2.0 * r_eff * layout.preload_penetration_m)
    return PadContactModel(layout, law, grip_cylinder(interface, layout), patch)


@dataclass(frozen=True)
class PadWrench:
    """Wrench of one hand's pads on the club (world axes, about the grip origin)."""

    force_n: np.ndarray
    moment_nm: np.ndarray
    normal_force_n: np.ndarray  # per pad, nonnegative
    tangential_force_n: np.ndarray  # per pad, magnitude


def pad_wrench(
    model: PadContactModel,
    side: str,
    hand: BushingState,
    club: BushingState,
    positions_grip_frame: np.ndarray | None = None,
) -> PadWrench:
    """Contact wrench of ``side``'s pads on the club grip.

    Postconditions: per-pad normal force is nonnegative; the force is the sum
    of the pad forces and the moment their moment about the club grip origin.
    """
    pads = (
        model.layout.positions_grip_frame(side)
        if positions_grip_frame is None
        else positions_grip_frame
    )
    axis = club.rotation[:, 0]
    axis_point = club.position_m + club.rotation @ model.layout.axis_offset_grip_frame(
        side
    )
    origin_s = model.cylinder.origin_axial_m(side)
    lo, hi = model.cylinder.axial_range_m
    force = np.zeros(3)
    moment = np.zeros(3)
    normal = np.zeros(len(pads))
    tangent = np.zeros(len(pads))
    for i, a in enumerate(pads):
        arm = hand.rotation @ a
        centre = hand.position_m + arm
        rel = centre - axis_point
        s = float(axis @ rel)
        if not lo <= origin_s + s <= hi:
            continue
        rho = rel - s * axis
        dist = math.sqrt(float(rho @ rho))
        if dist < 1e-9:
            continue
        n = rho / dist
        surface = axis_point + s * axis + model.cylinder.radius_m * n
        v_pad = hand.velocity_m_s + cross3(hand.omega_rad_s, arm)
        v_club = club.velocity_m_s + cross3(club.omega_rad_s, surface - club.position_m)
        plane = GroundPlane((float(n[0]), float(n[1]), float(n[2])), float(n @ surface))
        sample = sphere_ground_contact(
            centre, v_pad - v_club, model.layout.pad_radius_m, plane, model.law
        )
        if sample.penetration_m <= 0.0:
            continue
        on_pad = sample.normal_force_n + sample.friction_force_n
        spin = torsional_friction_moment(
            sample.normal_force_n,
            hand.omega_rad_s - club.omega_rad_s,
            plane,
            patch_radius_m=model.patch_radius_m,
            friction=model.law.static_friction,
            transition_rad_s=model.spin_transition_rad_s,
        )
        f_club = -on_pad
        force += f_club
        moment += cross3(sample.contact_point_m - club.position_m, f_club) - spin
        normal[i] = float(np.linalg.norm(sample.normal_force_n))
        tangent[i] = float(np.linalg.norm(sample.friction_force_n))
    return PadWrench(force, moment, normal, tangent)


def _finite_vec3(val: Any, name: str) -> np.ndarray:
    """Validate and return a finite 3-vector.

    Raises:
        ValueError: if val is None, cannot be cast to (3,), or contains non-finite values.
    """
    if val is None:
        raise ValueError(f"{name} must be provided")
    try:
        v = np.asarray(val, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a float array: {exc}") from exc
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f"{name} must be a finite 3-vector, got {val}")
    return v


def _default_state() -> BushingState:
    """Default stationary BushingState at origin with identity orientation."""
    return BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))


@dataclass(frozen=True)
class ContactSphere:
    """A spherical contact primitive in body or world coordinates.

    ``center_m``: 3-vector center position.
    ``radius_m``: positive sphere radius.
    ``frame``: coordinate frame, either 'body' (default) or 'world'.
    """

    center_m: np.ndarray
    radius_m: float
    frame: str = "body"

    def __post_init__(self) -> None:
        center = _finite_vec3(self.center_m, "center_m")
        object.__setattr__(self, "center_m", center)
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError(
                f"radius_m must be positive and finite, got {self.radius_m}"
            )
        if self.frame not in ("body", "world"):
            raise ValueError(f"frame must be 'body' or 'world', got {self.frame!r}")


@dataclass(frozen=True)
class ContactCapsule:
    """A capsule contact primitive defined by a line segment [p0, p1] and radius.

    ``p0_m``: 3-vector start point of the capsule core segment.
    ``p1_m``: 3-vector end point of the capsule core segment.
    ``radius_m``: positive capsule radius.
    ``frame``: coordinate frame, either 'body' (default) or 'world'.
    """

    p0_m: np.ndarray
    p1_m: np.ndarray
    radius_m: float
    frame: str = "body"

    def __post_init__(self) -> None:
        p0 = _finite_vec3(self.p0_m, "p0_m")
        p1 = _finite_vec3(self.p1_m, "p1_m")
        object.__setattr__(self, "p0_m", p0)
        object.__setattr__(self, "p1_m", p1)
        if not math.isfinite(self.radius_m) or self.radius_m <= 0.0:
            raise ValueError(
                f"radius_m must be positive and finite, got {self.radius_m}"
            )
        if self.frame not in ("body", "world"):
            raise ValueError(f"frame must be 'body' or 'world', got {self.frame!r}")

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
                f"tangential_force_mag_n must be finite and >= 0, got {self.tangential_force_mag_n}"
            )

    def total_force_n(self) -> np.ndarray:
        """Net force of the pair (identically zero by action/reaction)."""
        return self.force_a_n + self.force_b_n

    def total_moment_about_point(self, point: np.ndarray) -> np.ndarray:
        """Net moment of the pair about ``point`` (identically zero by action/reaction)."""
        pt = _finite_vec3(point, "point")
        return (
            self.moment_a_nm
            + self.moment_b_nm
            + cross3(self.point_a_m - self.point_b_m, self.force_a_n)
            - cross3(pt, self.force_a_n + self.force_b_n)
        )


def _coerce_sphere(s: Any) -> ContactSphere:
    if isinstance(s, ContactSphere):
        return s
    if hasattr(s, "center_m") and hasattr(s, "radius_m"):
        return ContactSphere(s.center_m, float(s.radius_m))
    if hasattr(s, "center") and hasattr(s, "radius"):
        return ContactSphere(s.center, float(s.radius))
    if isinstance(s, (tuple, list)) and len(s) == 2:
        return ContactSphere(s[0], float(s[1]))
    raise TypeError(f"Cannot coerce {type(s).__name__} to ContactSphere")


def _coerce_capsule(c: Any) -> ContactCapsule:
    if isinstance(c, ContactCapsule):
        return c
    if hasattr(c, "p0_m") and hasattr(c, "p1_m") and hasattr(c, "radius_m"):
        return ContactCapsule(c.p0_m, c.p1_m, float(c.radius_m))
    if hasattr(c, "start_m") and hasattr(c, "end_m") and hasattr(c, "radius_m"):
        return ContactCapsule(c.start_m, c.end_m, float(c.radius_m))
    if hasattr(c, "point_a") and hasattr(c, "point_b") and hasattr(c, "radius"):
        return ContactCapsule(c.point_a, c.point_b, float(c.radius))
    if isinstance(c, (tuple, list)) and len(c) == 3:
        return ContactCapsule(c[0], c[1], float(c[2]))
    raise TypeError(f"Cannot coerce {type(c).__name__} to ContactCapsule")


def closest_points_segments(
    a0: Any, a1: Any, b0: Any, b1: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Find closest points between two line segments [a0, a1] and [b0, b1].

    Preconditions:
        a0, a1, b0, b1 must be finite 3-vectors.

    Postconditions:
        Returns (pa, pb) on [a0, a1] and [b0, b1] respectively, minimizing ||pa - pb||.
        For parallel or coincident segments, returns a documented finite pair
        at the center of their mutual projection overlap.
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
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
        return p_a0.copy(), p_b0 + t * d2

    c = float(d1 @ r)
    if e < eps:
        t = 0.0
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


def closest_point_segment(point: Any, p0: Any, p1: Any) -> tuple[np.ndarray, float]:
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
    sphere: ContactSphere | Any,
    capsule: ContactCapsule | Any,
    state_a: BushingState | None = None,
    state_b: BushingState | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Signed distance and geometry between sphere (body A) and capsule (body B).

    Preconditions:
        sphere: ContactSphere or coercible.
        capsule: ContactCapsule or coercible.
        state_a, state_b: optional BushingState for body frames.

    Postconditions:
        Returns (signed_distance_m, contact_point_m, normal, closest_on_axis_m).
        signed_distance_m < 0 indicates penetration.
        normal is a unit vector pointing from body A (sphere) toward body B (capsule).
        contact_point_m is the deepest point of the sphere toward the capsule.
    """
    sph = _coerce_sphere(sphere)
    cap = _coerce_capsule(capsule)
    st_a = state_a if state_a is not None else _default_state()
    st_b = state_b if state_b is not None else _default_state()
    if not isinstance(st_a, BushingState):
        raise TypeError(f"state_a must be BushingState, got {type(st_a).__name__}")
    if not isinstance(st_b, BushingState):
        raise TypeError(f"state_b must be BushingState, got {type(st_b).__name__}")

    if sph.frame == "world":
        center_w = sph.center_m
    else:
        center_w = st_a.position_m + st_a.rotation @ sph.center_m

    if cap.frame == "world":
        b0_w, b1_w = cap.p0_m, cap.p1_m
    else:
        b0_w = st_b.position_m + st_b.rotation @ cap.p0_m
        b1_w = st_b.position_m + st_b.rotation @ cap.p1_m

    p_axis, _ = closest_point_segment(center_w, b0_w, b1_w)
    diff = p_axis - center_w
    dist = float(np.linalg.norm(diff))

    normal = diff / dist if dist > 1e-12 else _orthogonal_unit_vector(b1_w - b0_w)

    signed_dist = dist - sph.radius_m - cap.radius_m
    contact_point = center_w + sph.radius_m * normal
    return signed_dist, contact_point, normal, p_axis


def sphere_capsule(
    sphere: ContactSphere | Any,
    capsule: ContactCapsule | Any,
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
        law must be ContactParameters.
        patch_radius_m >= 0.0 and finite.
        spin_transition_rad_s > 0.0 and finite.

    Postconditions:
        Returns PairWrench satisfying action/reaction by construction.
        Zero force and moment when separated (gap >= 0.0).
        Normal force is strictly unilateral / compressive (never pulling).
    """
    if law is None or not isinstance(law, ContactParameters):
        raise TypeError(f"law must be ContactParameters, got {type(law).__name__}")
    if not math.isfinite(patch_radius_m) or patch_radius_m < 0.0:
        raise ValueError(
            f"patch_radius_m must be finite and >= 0, got {patch_radius_m}"
        )
    if not math.isfinite(spin_transition_rad_s) or spin_transition_rad_s <= 0.0:
        raise ValueError(
            f"spin_transition_rad_s must be positive and finite, got {spin_transition_rad_s}"
        )

    sph = _coerce_sphere(sphere)
    cap = _coerce_capsule(capsule)
    st_a = state_a if state_a is not None else _default_state()
    st_b = state_b if state_b is not None else _default_state()

    signed_dist, contact_point, normal, p_axis = sphere_capsule_distance(
        sph, cap, st_a, st_b
    )
    penetration = max(0.0, -signed_dist)
    zero = np.zeros(3)

    if penetration <= 0.0:
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

    if sph.frame == "world":
        center_w = sph.center_m
    else:
        center_w = st_a.position_m + st_a.rotation @ sph.center_m

    p_surface = p_axis - cap.radius_m * normal
    arm_a = center_w - st_a.position_m
    arm_b = p_surface - st_b.position_m
    v_a = st_a.velocity_m_s + cross3(st_a.omega_rad_s, arm_a)
    v_b = st_b.velocity_m_s + cross3(st_b.omega_rad_s, arm_b)

    n_legacy = -normal
    plane = GroundPlane(
        (float(n_legacy[0]), float(n_legacy[1]), float(n_legacy[2])),
        float(n_legacy @ p_surface),
    )
    sample = sphere_ground_contact(center_w, v_a - v_b, sph.radius_m, plane, law)

    on_sph = sample.normal_force_n + sample.friction_force_n
    spin = torsional_friction_moment(
        sample.normal_force_n,
        st_a.omega_rad_s - st_b.omega_rad_s,
        plane,
        patch_radius_m=patch_radius_m,
        friction=law.static_friction,
        transition_rad_s=spin_transition_rad_s,
    )

    force_a = on_sph
    force_b = -force_a
    moment_a = cross3(sample.contact_point_m - st_a.position_m, force_a) + spin
    moment_b = cross3(sample.contact_point_m - st_b.position_m, force_b) - spin

    f_n_mag = float(np.linalg.norm(sample.normal_force_n))
    f_t_mag = float(np.linalg.norm(sample.friction_force_n))

    return PairWrench(
        force_a_n=force_a,
        moment_a_nm=moment_a,
        force_b_n=force_b,
        moment_b_nm=moment_b,
        contact_point_m=sample.contact_point_m,
        normal=normal,
        penetration_m=penetration,
        point_a_m=st_a.position_m,
        point_b_m=st_b.position_m,
        normal_force_mag_n=f_n_mag,
        tangential_force_mag_n=f_t_mag,
    )


def capsule_capsule_distance(
    capsule_a: ContactCapsule | Any,
    capsule_b: ContactCapsule | Any,
    state_a: BushingState | None = None,
    state_b: BushingState | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Signed distance and geometry between two capsules (body A and body B).

    Preconditions:
        capsule_a, capsule_b: ContactCapsule or coercible.
        state_a, state_b: optional BushingState for body frames.

    Postconditions:
        Returns (signed_distance_m, contact_point_m, normal, pa_m, pb_m).
        signed_distance_m < 0 indicates penetration.
        normal is a unit vector pointing from body A toward body B.
        contact_point_m is the midpoint of the surface overlap.
    """
    cap_a = _coerce_capsule(capsule_a)
    cap_b = _coerce_capsule(capsule_b)
    st_a = state_a if state_a is not None else _default_state()
    st_b = state_b if state_b is not None else _default_state()
    if not isinstance(st_a, BushingState):
        raise TypeError(f"state_a must be BushingState, got {type(st_a).__name__}")
    if not isinstance(st_b, BushingState):
        raise TypeError(f"state_b must be BushingState, got {type(st_b).__name__}")

    if cap_a.frame == "world":
        a0_w, a1_w = cap_a.p0_m, cap_a.p1_m
    else:
        a0_w = st_a.position_m + st_a.rotation @ cap_a.p0_m
        a1_w = st_a.position_m + st_a.rotation @ cap_a.p1_m

    if cap_b.frame == "world":
        b0_w, b1_w = cap_b.p0_m, cap_b.p1_m
    else:
        b0_w = st_b.position_m + st_b.rotation @ cap_b.p0_m
        b1_w = st_b.position_m + st_b.rotation @ cap_b.p1_m

    pa, pb = closest_points_segments(a0_w, a1_w, b0_w, b1_w)
    diff = pb - pa
    dist = float(np.linalg.norm(diff))

    normal = diff / dist if dist > 1e-12 else _orthogonal_unit_vector(a1_w - a0_w)

    signed_dist = dist - cap_a.radius_m - cap_b.radius_m
    sa = pa + cap_a.radius_m * normal
    sb = pb - cap_b.radius_m * normal
    contact_point = 0.5 * (sa + sb)
    return signed_dist, contact_point, normal, pa, pb


def capsule_capsule(
    capsule_a: ContactCapsule | Any,
    capsule_b: ContactCapsule | Any,
    state_a: BushingState | None = None,
    state_b: BushingState | None = None,
    law: ContactParameters | None = None,
    patch_radius_m: float = 0.0,
    spin_transition_rad_s: float = 1.0,
) -> PairWrench:
    """Shared contact law between two capsules (body A and body B).

    Reuses the Hunt-Crossley normal force, regularised Coulomb/Stribeck friction
    and torsional spin friction moment from motion_matching.contact_law.

    Preconditions:
        law must be ContactParameters.
        patch_radius_m >= 0.0 and finite.
        spin_transition_rad_s > 0.0 and finite.

    Postconditions:
        Returns PairWrench satisfying action/reaction by construction.
        Zero force and moment when separated (gap >= 0.0).
        Normal force is strictly unilateral / compressive (never pulling).
    """
    if law is None or not isinstance(law, ContactParameters):
        raise TypeError(f"law must be ContactParameters, got {type(law).__name__}")
    if not math.isfinite(patch_radius_m) or patch_radius_m < 0.0:
        raise ValueError(
            f"patch_radius_m must be finite and >= 0, got {patch_radius_m}"
        )
    if not math.isfinite(spin_transition_rad_s) or spin_transition_rad_s <= 0.0:
        raise ValueError(
            f"spin_transition_rad_s must be positive and finite, got {spin_transition_rad_s}"
        )

    cap_a = _coerce_capsule(capsule_a)
    cap_b = _coerce_capsule(capsule_b)
    st_a = state_a if state_a is not None else _default_state()
    st_b = state_b if state_b is not None else _default_state()

    signed_dist, contact_point, normal, pa, pb = capsule_capsule_distance(
        cap_a, cap_b, st_a, st_b
    )
    penetration = max(0.0, -signed_dist)
    zero = np.zeros(3)

    if penetration <= 0.0:
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

    v_a = st_a.velocity_m_s + cross3(st_a.omega_rad_s, contact_point - st_a.position_m)
    v_b = st_b.velocity_m_s + cross3(st_b.omega_rad_s, contact_point - st_b.position_m)

    sa = pa + cap_a.radius_m * normal
    plane = GroundPlane(
        (float(normal[0]), float(normal[1]), float(normal[2])),
        float(normal @ sa),
    )
    sample = sphere_ground_contact(pb, v_b - v_a, cap_b.radius_m, plane, law)

    force_b = sample.normal_force_n + sample.friction_force_n
    force_a = -force_b

    spin_b = torsional_friction_moment(
        sample.normal_force_n,
        st_b.omega_rad_s - st_a.omega_rad_s,
        plane,
        patch_radius_m=patch_radius_m,
        friction=law.static_friction,
        transition_rad_s=spin_transition_rad_s,
    )
    spin_a = -spin_b

    moment_a = cross3(contact_point - st_a.position_m, force_a) + spin_a
    moment_b = cross3(contact_point - st_b.position_m, force_b) + spin_b

    f_n_mag = float(np.linalg.norm(sample.normal_force_n))
    f_t_mag = float(np.linalg.norm(sample.friction_force_n))

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
        normal_force_mag_n=f_n_mag,
        tangential_force_mag_n=f_t_mag,
    )


def capsule_sphere(
    capsule: ContactCapsule | Any,
    sphere: ContactSphere | Any,
    state_a: BushingState | None = None,
    state_b: BushingState | None = None,
    law: ContactParameters | None = None,
    patch_radius_m: float = 0.0,
    spin_transition_rad_s: float = 1.0,
) -> PairWrench:
    """Capsule (body A) and sphere (body B) contact; delegates to sphere_capsule."""
    w = sphere_capsule(
        sphere,
        capsule,
        state_b,
        state_a,
        law,
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
