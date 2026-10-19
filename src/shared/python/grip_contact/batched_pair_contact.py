"""Batched one-side evaluation of the sphere/capsule-on-grip law (#13384, #13304 follow-up).

:func:`side_rows_wrench` evaluates every hand-site/grip pair of one side in one
pass, with inputs validated once by the caller.  The per-pair kernels in
:mod:`src.shared.python.grip_contact.pair_contact` stay the reference.

A side has only four to six pairs, so per-call overhead, not arithmetic,
dominates.  A row-vectorised numpy pass costs about 120 array operations of
1-3 us each and measured only ~3x faster than the reference, so this kernel
does the per-row algebra in Python floats and calls numpy once per frame for
the rigid transforms.  It reproduces the reference arithmetic operation by
operation:

- rigid transforms use stacked ``np.matmul`` (one ``R @ x`` per row, the same
  BLAS call as the reference's scalar ``R @ x``) and a Python-float add;
- 3-vector dot products are summed left to right, as numpy's 1-D ``x @ y``;
  cross products use the ``bushing_law.cross3`` component expressions;
- the Hunt-Crossley normal, the regularised Coulomb/Stribeck tangential and the
  torsional spin moment follow
  :func:`src.shared.python.motion_matching.contact_law.sphere_ground_contact`
  and :func:`~src.shared.python.motion_matching.contact_law.torsional_friction_moment`
  term for term, with the same ``math.exp``/``math.tanh``.

The result therefore equals the reference to round-off (tested to <= 1e-12 N).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.pair_contact import _orthogonal_unit_vector
from src.shared.python.motion_matching.contact_law import ContactParameters

__all__ = [
    "PairRow",
    "SideBatch",
    "SideRows",
    "closest_points_segments_floats",
    "rigid_points_floats",
    "side_rows_wrench",
]

Vec3 = tuple[float, float, float]

#: Degenerate-segment threshold of the reference closest-point queries.
_SEGMENT_EPS = 1e-14
#: Coincident-point threshold of the reference distance queries.
_COINCIDENT_EPS = 1e-12
_ZERO: Vec3 = (0.0, 0.0, 0.0)


def _vec(a: Sequence[float]) -> Vec3:
    return (a[0], a[1], a[2])


def _add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(s: float, a: Sequence[float]) -> Vec3:
    return (s * a[0], s * a[1], s * a[2])


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Left-to-right 3-vector dot product (bit-equal to numpy's 1-D ``a @ b``)."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _clip01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def rigid_points_floats(
    rotation: np.ndarray, position_m: np.ndarray, stacked_points_m: np.ndarray
) -> list[Vec3]:
    """World points ``position + rotation @ p`` of stacked body-frame rows.

    ``stacked_points_m`` has shape (K, 1, 3).  The stacked row-vector product
    issues one matrix-vector product per row, so each ``rotation @ p`` is
    bit-identical to the reference's scalar product; the translation is a
    Python-float add, as exact as numpy's.
    """
    x, y, z = position_m.tolist()
    rotated = np.matmul(stacked_points_m, rotation.T).tolist()
    return [(x + a, y + b, z + c) for ((a, b, c),) in rotated]


def _segment_params(
    a: float, b: float, c: float, e: float, f: float
) -> tuple[float, float]:
    """Segment parameters ``(s, t)`` of the reference's non-degenerate branches."""
    denom = max(a * e - b * b, 0.0)
    if denom < 1e-10 * a * e:  # parallel: centre of the projection overlap
        s0 = -c / a
        s1 = (-c + b) / a
        s_min, s_max = min(s0, s1), max(s0, s1)
        lo, hi = max(0.0, s_min), min(1.0, s_max)
        if lo <= hi:
            s = 0.5 * (lo + hi)
            return s, _clip01((a * s + c) / b)
        if s_max < 0.0:
            return 0.0, _clip01(f / e)
        return 1.0, _clip01((b + f) / e)
    s = (b * f - c * e) / denom
    t = (b * s + f) / e
    if s < 0.0:
        return 0.0, _clip01(f / e)
    if s > 1.0:
        return 1.0, _clip01((b + f) / e)
    if t < 0.0:
        return _clip01(-c / a), 0.0
    if t > 1.0:
        return _clip01((b - c) / a), 1.0
    return s, t


def closest_points_segments_floats(
    a0: Sequence[float], a1: Sequence[float], b0: Sequence[float], b1: Sequence[float]
) -> tuple[Vec3, Vec3]:
    """Closest points of segments ``[a0, a1]`` and ``[b0, b1]`` in Python floats.

    Operation-for-operation port of :func:`pair_contact.closest_points_segments`,
    including its degenerate and parallel branches, without per-call
    validation: the caller guarantees finite 3-vectors.
    """
    d1 = _sub(a1, a0)
    d2 = _sub(b1, b0)
    r = _sub(a0, b0)
    a = _dot(d1, d1)
    e = _dot(d2, d2)
    f = _dot(d2, r)
    if a < _SEGMENT_EPS and e < _SEGMENT_EPS:
        return _vec(a0), _vec(b0)
    if a < _SEGMENT_EPS:
        return _vec(a0), _add(b0, _scale(_clip01(f / e), d2))
    c = _dot(d1, r)
    if e < _SEGMENT_EPS:
        return _add(a0, _scale(_clip01(-c / a), d1)), _vec(b0)
    s, t = _segment_params(a, _dot(d1, d2), c, e, f)
    return _add(a0, _scale(s, d1)), _add(b0, _scale(t, d2))


class SideBatch(NamedTuple):
    """Hand-site primitives of one side, stacked in topology pair order.

    ``pair_names``: the N physical pairs, in topology order.
    ``hand_points_m``: (N + Nc, 1, 3) hand-frame points; row ``k < N`` is pair
    ``k``'s sphere centre or capsule start, rows ``N..`` the capsule ends.
    ``capsule_ends``: per pair, the ``hand_points_m`` row of its capsule end,
    or ``-1`` for a sphere site.
    ``site_radii_m``: per-pair site radii.
    ``grip_points_m``: (2, 1, 3) club-frame grip capsule ends.
    ``grip_radius_m``: grip capsule radius.
    """

    pair_names: tuple[str, ...]
    hand_points_m: np.ndarray
    capsule_ends: tuple[int, ...]
    site_radii_m: tuple[float, ...]
    grip_points_m: np.ndarray
    grip_radius_m: float


class PairRow(NamedTuple):
    """One pair's wrench on the club and its contact geometry (world axes).

    ``force_n``/``moment_nm``: force on the club and moment about the club
    frame origin; ``spin_nm``: the spin part of that moment.  ``normal``
    points from the hand site to the grip; ``penetration_m >= 0``.
    """

    force_n: Vec3
    moment_nm: Vec3
    spin_nm: Vec3
    contact_point_m: Vec3
    normal: Vec3
    penetration_m: float
    normal_force_mag_n: float
    tangential_force_mag_n: float


class SideRows(NamedTuple):
    """Per-pair rows in ``SideBatch.pair_names`` order and their resultant."""

    rows: tuple[PairRow, ...]
    force_n: Vec3
    moment_nm: Vec3


class _Frame(NamedTuple):
    """A rigid frame's origin, velocity and angular velocity as Python floats."""

    position_m: Vec3
    velocity_m_s: Vec3
    omega_rad_s: Vec3


class _Side(NamedTuple):
    """Per-call constants shared by every row of one side evaluation."""

    hand: _Frame
    club: _Frame
    #: ``hand.omega - club.omega``; the capsule rows use its exact negation.
    rel_omega: Vec3
    grip_p0_m: Vec3
    grip_p1_m: Vec3
    #: ``grip_p1_m - grip_p0_m`` and its squared length.
    grip_axis_m: Vec3
    grip_axis_sq_m2: float
    grip_radius_m: float
    law: ContactParameters
    patch_radius_m: float
    spin_transition_rad_s: float


#: Normal force, friction force and spin moment on the sampled body.
_Sample = tuple[Vec3, Vec3, Vec3]
_NO_SAMPLE: _Sample = (_ZERO, _ZERO, _ZERO)


def _frame(state: BushingState) -> _Frame:
    return _Frame(
        _vec(state.position_m.tolist()),
        _vec(state.velocity_m_s.tolist()),
        _vec(state.omega_rad_s.tolist()),
    )


def _relative_velocity(side: _Side, point: Vec3) -> Vec3:
    """``v_hand(point) - v_club(point)`` (reference ``_point_velocity`` twice)."""
    (hx, hy, hz), (hvx, hvy, hvz), (hwx, hwy, hwz) = side.hand
    (cx, cy, cz), (cvx, cvy, cvz), (cwx, cwy, cwz) = side.club
    px, py, pz = point
    ax, ay, az = px - hx, py - hy, pz - hz
    bx, by, bz = px - cx, py - cy, pz - cz
    return (
        (hvx + (hwy * az - hwz * ay)) - (cvx + (cwy * bz - cwz * by)),
        (hvy + (hwz * ax - hwx * az)) - (cvy + (cwz * bx - cwx * bz)),
        (hvz + (hwx * ay - hwy * ax)) - (cvz + (cwx * by - cwy * bx)),
    )


def _unit_normal(diff: Vec3, dist: float, axis: Vec3) -> Vec3:
    """``diff / dist``, or the reference's deterministic fallback if coincident."""
    if dist > _COINCIDENT_EPS:
        return (diff[0] / dist, diff[1] / dist, diff[2] / dist)
    return _vec(_orthogonal_unit_vector(np.array(axis)).tolist())


def _sample(
    sphere: tuple[Vec3, float],
    plane: tuple[Vec3, Vec3],
    velocity: Vec3,
    omega: Vec3,
    side: _Side,
) -> _Sample:
    """``sphere_ground_contact`` then ``torsional_friction_moment`` on one plane.

    ``sphere`` is the sampled (centre, radius); ``plane`` is (raw normal,
    surface point), made a plane as ``GroundPlane`` does: the normal is
    re-normalised and the height is ``raw_normal . surface_point``.  Every
    expression keeps the reference's operand order.
    """
    (cx, cy, cz), radius = sphere
    (nx, ny, nz), (sx, sy, sz) = plane
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    ux, uy, uz = nx / norm, ny / norm, nz / norm
    height = nx * sx + ny * sy + nz * sz
    penetration = max(0.0, -(ux * cx + uy * cy + uz * cz - height - radius))
    if penetration <= 0.0:
        return _NO_SAMPLE
    law = side.law
    vx, vy, vz = velocity
    v_n = ux * vx + uy * vy + uz * vz
    magnitude = max(
        0.0, law.stiffness_n_m * penetration * (1.0 + law.dissipation_s_m * -v_n)
    )
    normal = (ux * magnitude, uy * magnitude, uz * magnitude)
    tx, ty, tz = vx - ux * v_n, vy - uy * v_n, vz - uz * v_n
    speed = math.sqrt(tx * tx + ty * ty + tz * tz)
    friction = _ZERO
    if speed > 0.0 and magnitude > 0.0:  # regularised Coulomb/Stribeck
        ratio = speed / law.transition_velocity_m_s
        mu = (
            law.dynamic_friction
            + (law.static_friction - law.dynamic_friction) * math.exp(-(ratio**2))
            + law.viscous_friction * speed
        )
        th = math.tanh(ratio)
        friction = (
            -tx / speed * mu * magnitude * th,
            -ty / speed * mu * magnitude * th,
            -tz / speed * mu * magnitude * th,
        )
    load = max(0.0, ux * normal[0] + uy * normal[1] + uz * normal[2])
    limit = law.static_friction * load * side.patch_radius_m
    wx, wy, wz = omega
    th = math.tanh((ux * wx + uy * wy + uz * wz) / side.spin_transition_rad_s)
    return normal, friction, (-ux * limit * th, -uy * limit * th, -uz * limit * th)


def _club_row(
    geometry: tuple[Vec3, Vec3, float], smp: _Sample, hand_sampled: bool, side: _Side
) -> PairRow:
    """Reaction on the club (``PairWrench.force_b_n``/``moment_b_nm``).

    ``hand_sampled`` is true when the hand site is the sampled body
    (``sphere_capsule``) and false when the grip is (``capsule_capsule``);
    the sign changes between them are exact negations.
    """
    contact, normal, penetration = geometry
    (nx, ny, nz), (tx, ty, tz), (sx, sy, sz) = smp
    fx, fy, fz = nx + tx, ny + ty, nz + tz
    if hand_sampled:
        fx, fy, fz = -fx, -fy, -fz
    else:
        sx, sy, sz = -sx, -sy, -sz
    px, py, pz = side.club.position_m
    lx, ly, lz = contact[0] - px, contact[1] - py, contact[2] - pz
    return PairRow(
        (fx, fy, fz),
        (
            (ly * fz - lz * fy) - sx,
            (lz * fx - lx * fz) - sy,
            (lx * fy - ly * fx) - sz,
        ),
        (-sx, -sy, -sz),
        contact,
        normal,
        penetration,
        math.sqrt(nx * nx + ny * ny + nz * nz),
        math.sqrt(tx * tx + ty * ty + tz * tz),
    )


def _sphere_row(center: Vec3, radius: float, side: _Side) -> PairRow:
    """Reference ``sphere_capsule``: A = hand sphere (sampled), B = grip."""
    (gx, gy, gz), d = side.grip_p0_m, side.grip_axis_m
    r_grip = side.grip_radius_m
    cx, cy, cz = center
    d_sq = side.grip_axis_sq_m2
    if d_sq < _SEGMENT_EPS:
        near = side.grip_p0_m
    else:
        t = _clip01(((cx - gx) * d[0] + (cy - gy) * d[1] + (cz - gz) * d[2]) / d_sq)
        near = (gx + t * d[0], gy + t * d[1], gz + t * d[2])
    diff = (near[0] - cx, near[1] - cy, near[2] - cz)
    dist = math.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
    n = _unit_normal(diff, dist, d)
    contact = (cx + radius * n[0], cy + radius * n[1], cz + radius * n[2])
    penetration = max(0.0, -(dist - radius - r_grip))
    if penetration <= 0.0:
        return PairRow(_ZERO, _ZERO, _ZERO, contact, n, 0.0, 0.0, 0.0)
    surface = (
        near[0] - r_grip * n[0],
        near[1] - r_grip * n[1],
        near[2] - r_grip * n[2],
    )
    smp = _sample(
        (center, radius),
        ((-n[0], -n[1], -n[2]), surface),
        _relative_velocity(side, contact),
        side.rel_omega,
        side,
    )
    return _club_row((contact, n, penetration), smp, True, side)


def _capsule_row(ends: tuple[Vec3, Vec3], radius: float, side: _Side) -> PairRow:
    """Reference ``capsule_capsule``: A = hand capsule, B = grip (sampled)."""
    r_grip = side.grip_radius_m
    pa, pb = closest_points_segments_floats(
        ends[0], ends[1], side.grip_p0_m, side.grip_p1_m
    )
    (ax, ay, az), (bx, by, bz) = pa, pb
    diff = (bx - ax, by - ay, bz - az)
    dist = math.sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
    n = _unit_normal(diff, dist, _sub(ends[1], ends[0]))
    surface_a = (ax + radius * n[0], ay + radius * n[1], az + radius * n[2])
    contact = (
        0.5 * (surface_a[0] + (bx - r_grip * n[0])),
        0.5 * (surface_a[1] + (by - r_grip * n[1])),
        0.5 * (surface_a[2] + (bz - r_grip * n[2])),
    )
    penetration = max(0.0, -(dist - radius - r_grip))
    if penetration <= 0.0:
        return PairRow(_ZERO, _ZERO, _ZERO, contact, n, 0.0, 0.0, 0.0)
    (vx, vy, vz), (wx, wy, wz) = _relative_velocity(side, contact), side.rel_omega
    smp = _sample((pb, r_grip), (n, surface_a), (-vx, -vy, -vz), (-wx, -wy, -wz), side)
    return _club_row((contact, n, penetration), smp, False, side)


def side_rows_wrench(
    batch: SideBatch,
    hand: BushingState,
    club: BushingState,
    law: ContactParameters,
    patch_radius_m: float,
    spin_transition_rad_s: float,
) -> SideRows:
    """Every hand-site/grip pair of one side in one batched pass.

    Preconditions (validated once by the caller): ``hand`` and ``club`` hold a
    proper rotation and finite 3-vectors; ``law``, ``patch_radius_m`` and
    ``spin_transition_rad_s`` are valid contact-law arguments.

    Postconditions:
        ``rows[k]`` equals the reference ``sphere_capsule``/``capsule_capsule``
        wrench on the club of ``batch.pair_names[k]`` to round-off; separated
        pairs carry zero force and moment; the resultant is the left-to-right
        sum of the rows, as in the reference side loop.
    """
    points = rigid_points_floats(hand.rotation, hand.position_m, batch.hand_points_m)
    g0, g1 = rigid_points_floats(club.rotation, club.position_m, batch.grip_points_m)
    hand_f, club_f = _frame(hand), _frame(club)
    axis = _sub(g1, g0)
    side = _Side(
        hand=hand_f,
        club=club_f,
        rel_omega=_sub(hand_f.omega_rad_s, club_f.omega_rad_s),
        grip_p0_m=g0,
        grip_p1_m=g1,
        grip_axis_m=axis,
        grip_axis_sq_m2=_dot(axis, axis),
        grip_radius_m=batch.grip_radius_m,
        law=law,
        patch_radius_m=patch_radius_m,
        spin_transition_rad_s=spin_transition_rad_s,
    )
    rows = []
    fx = fy = fz = mx = my = mz = 0.0
    for k, end in enumerate(batch.capsule_ends):
        if end < 0:
            row = _sphere_row(points[k], batch.site_radii_m[k], side)
        else:
            row = _capsule_row((points[k], points[end]), batch.site_radii_m[k], side)
        rows.append(row)
        (dfx, dfy, dfz), (dmx, dmy, dmz) = row.force_n, row.moment_nm
        fx, fy, fz = fx + dfx, fy + dfy, fz + dfz
        mx, my, mz = mx + dmx, my + dmy, mz + dmz
    return SideRows(tuple(rows), (fx, fy, fz), (mx, my, mz))
