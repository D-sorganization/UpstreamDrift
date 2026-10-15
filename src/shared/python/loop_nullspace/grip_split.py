"""Grip-frame hand-split coordinates and the 6x6 grasp-map kernel (NSX-16).

The two hands exert ``y = (f_L, c_L, f_R, c_R)`` on the bare club (force at
the grip point plus a free couple, grip-frame axes).  The *net* wrench about
the grip origin ``o`` is data-determined by bare-club Newton-Euler
(:func:`net_club_wrench`, ``INVARIANT``); only the split between the hands is
free.  With the grasp map ``G`` (6x12), its kernel ``K`` (12x6) and the split
functional ``Phi``::

    w = G y,   s = Phi y = (1/2 (f_R - f_L), 1/2 (c_R - c_L)),
    y = G# w + K s,        G K = 0,   Phi K = I,   G G# = I.

The columns of ``K`` are the named split coordinates (axial pull/squeeze, two
transverse shear couples, twist couple, two bending couples).  ``K`` and ``s``
do not depend on the reference point, which only transports ``G`` (``T(t) G``).

Non-claims: coordinate values for a real swing are ``FAMILY_DEPENDENT``; this
module chooses no split (NSX-23).  Bare-club Newton-Euler is delegated to
``holding_hand_wrench``; the hand-on-club closing map is the exact affine map
of the specification.  Scaling uses ``C-SCALE-FORCE-MOMENT``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.biomechanics.grip_extraction import holding_hand_wrench
from src.shared.python.loop_nullspace.contracts import ClosedLoopSnapshot
from src.shared.python.loop_nullspace.decomposition import CharacteristicScales
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    GripFrame,
    InputProvenance,
    QuantityRecord,
    check_shape,
    readonly_float,
)

__all__ = [
    "SPLIT_COLUMN_NAMES",
    "closing_hand_map",
    "equal_split_inverse",
    "grasp_map",
    "hand_wrenches",
    "net_club_wrench",
    "scale_split",
    "split_basis",
    "split_coordinates",
    "split_functional",
]

SPLIT_COLUMN_NAMES = (
    "s1_axial_pull_positive_squeeze_negative",
    "s2_transverse_shear_couple_v",
    "s3_transverse_shear_couple_w",
    "s4_twist_couple_about_shaft",
    "s5_bending_couple_about_v",
    "s6_bending_couple_about_w",
)
SPLIT_UNITS = ("N", "N", "N", "N m", "N m", "N m")
NET_WRENCH_UNITS = "N,N,N,N m,N m,N m"
_STRUCT_TOL = 1e-12
_MASS_RTOL = 1e-9
_I3 = np.eye(3)
_Z3 = np.zeros((3, 3))


def _skew(v: Array) -> Array:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=np.float64
    )


def _vec(x: ArrayLike, n: int, name: str) -> Array:
    arr = readonly_float(x, name)
    check_shape(arr, (n,), name)
    return arr


def grasp_map(grip_frame: GripFrame, r_L: ArrayLike, r_R: ArrayLike) -> Array:
    """Grasp map ``G`` (6x12) about the grip origin, in grip-frame axes.

    ``r_L`` and ``r_R`` are the world grip points.  The grip origin must be
    their midpoint (the NSX-00 grip-frame definition).  Row layout is
    ``(F, M)``; column layout is ``y = (f_L, c_L, f_R, c_R)``.

    Raises:
        ValueError: if the points are coincident, non-finite, wrongly shaped,
            or the frame origin is not their midpoint.
    """
    left, right = _vec(r_L, 3, "r_L"), _vec(r_R, 3, "r_R")
    origin = grip_frame.origin
    scale = max(1.0, float(np.linalg.norm(right - left)))
    if float(np.linalg.norm(right - left)) <= 1e-12:
        raise ValueError("r_L and r_R must be distinct grip points")
    if float(np.linalg.norm(0.5 * (left + right) - origin)) > 1e-9 * scale:
        raise ValueError("grip origin must be the midpoint of r_L and r_R")
    rot = grip_frame.rotation
    rho_l, rho_r = rot.T @ (left - origin), rot.T @ (right - origin)
    return np.block([[_I3, _Z3, _I3, _Z3], [_skew(rho_l), _I3, _skew(rho_r), _I3]])


def _check_grasp_map(g: Array) -> Array:
    arr = np.asarray(g, dtype=np.float64)
    check_shape(arr, (6, 12), "G")
    blocks = (arr[:3, :3], arr[:3, 3:6], arr[:3, 6:9], arr[:3, 9:], arr[3:, 3:6])
    expected = (_I3, _Z3, _I3, _Z3, _I3)
    ok = all(
        np.max(np.abs(b - e)) <= _STRUCT_TOL
        for b, e in zip(blocks, expected, strict=True)
    )
    ok = ok and np.max(np.abs(arr[3:, 9:] - _I3)) <= _STRUCT_TOL
    skew_l, skew_r = arr[3:, :3], arr[3:, 6:9]
    ok = ok and np.max(np.abs(skew_l + skew_l.T)) <= _STRUCT_TOL
    ok = ok and np.max(np.abs(skew_r + skew_r.T)) <= _STRUCT_TOL
    ok = ok and np.max(np.abs(skew_l + skew_r)) <= _STRUCT_TOL
    if not ok:
        raise ValueError(
            "G is not a grasp map [[I,0,I,0],[skew(-d),I,skew(d),I]]; "
            "the origin must be the grip midpoint"
        )
    return arr


def split_basis(G: Array) -> Array:  # noqa: N803 - matches the specification
    """Exact kernel ``K`` (12x6) of ``G`` with the named columns.

    Postconditions: ``G @ K == 0`` and ``rank K == 6``; ``K`` depends only on
    the half-separation encoded in ``G`` (not on the reference point).

    Raises:
        ValueError: if ``G`` is not a 6x12 midpoint grasp map.
    """
    g = _check_grasp_map(G)
    skew_d = g[3:, 6:9]
    k = np.block(
        [[-_I3, _Z3], [-skew_d, -_I3], [_I3, _Z3], [-skew_d, _I3]],
    )
    if np.max(np.abs(g @ k)) > 1e-10 * max(1.0, float(np.max(np.abs(g)))):
        raise ValueError("kernel check G K = 0 failed")
    return k


def split_functional() -> Array:
    """``Phi`` (6x12) with ``s = Phi y = (1/2 (f_R - f_L), 1/2 (c_R - c_L))``."""
    h = 0.5 * _I3
    return np.block([[-h, _Z3, h, _Z3], [_Z3, -h, _Z3, h]])


def equal_split_inverse() -> Array:
    """Equal-split right inverse ``G#`` (12x6): each hand carries half the net."""
    h = 0.5 * _I3
    return np.block([[h, _Z3], [_Z3, h], [h, _Z3], [_Z3, h]])


def hand_wrenches(
    net: ArrayLike,
    s: ArrayLike,
    G: Array,  # noqa: N803 - matches the specification
) -> tuple[Array, Array]:
    """Hand wrenches ``(W_L, W_R)`` (each ``(f, c)``) from ``y = G# w + K s``.

    Postconditions: ``G y == net`` and ``Phi y == s`` to rounding.

    Raises:
        ValueError: on wrong shapes, non-finite input or a non-grasp-map ``G``.
    """
    w = _vec(net, 6, "net")
    coords = _vec(s, 6, "s")
    y = equal_split_inverse() @ w + split_basis(G) @ coords
    return y[:6].copy(), y[6:].copy()


def split_coordinates(W_L: ArrayLike, W_R: ArrayLike) -> Array:  # noqa: N803
    """Inverse of :func:`hand_wrenches`: ``s = Phi (W_L, W_R)``.

    Raises:
        ValueError: on wrong shapes or non-finite input.
    """
    y = np.concatenate([_vec(W_L, 6, "W_L"), _vec(W_R, 6, "W_R")])
    return split_functional() @ y


def scale_split(s: ArrayLike, scales: CharacteristicScales | None = None) -> Array:
    """Split coordinates divided by ``F_c`` (forces) and ``M_c`` (couples)."""
    coords = _vec(s, 6, "s")
    sc = scales or CharacteristicScales()
    return coords / np.array([sc.force] * 3 + [sc.moment] * 3)


def net_club_wrench(
    snapshot: ClosedLoopSnapshot, club_ne: Mapping[str, Any]
) -> QuantityRecord:
    """Net hand-on-club wrench about the grip origin, grip-frame axes.

    ``club_ne`` is the bare-club Newton-Euler keyword set of the engine
    (``adapter.club_newton_euler``, solids head, shaft and grip only).  The
    wrench is ``holding_hand_wrench`` with no closing hand, so the "holding
    hand" supplies all of it at the grip origin, then rotated into grip axes.
    ``INVARIANT``: identical for every admissible hand split.

    Raises:
        ValueError: on bad keywords or a mass that differs from
            ``snapshot.bare_club``.
    """
    mass = club_ne.get("mass_kg")
    ref = snapshot.bare_club.mass
    if mass is None or abs(float(mass) - ref) > _MASS_RTOL * ref:
        raise ValueError(
            f"club_ne mass {mass} does not match snapshot bare-club mass {ref}"
        )
    zero = (0.0, 0.0, 0.0)
    force, torque = holding_hand_wrench(
        closing_force_n=zero,
        closing_torque_nm=zero,
        closing_point_m=snapshot.grip_frame.origin,
        holding_point_m=snapshot.grip_frame.origin,
        **dict(club_ne),
    )
    rot = snapshot.grip_frame.rotation
    values = np.concatenate([rot.T @ np.asarray(force), rot.T @ np.asarray(torque)])
    return QuantityRecord(
        values=values,
        units=NET_WRENCH_UNITS,
        allocation_dependence=AllocationDependence.INVARIANT,
        model_provenance=snapshot.model_provenance,
        input_provenance=InputProvenance(
            kind="bare_club_newton_euler",
            source=snapshot.input_provenance.source,
            note="net hand wrench on the bare club about the grip origin",
        ),
        conventions=("C-GRIP-FRAME", "C-SCALE-FORCE-MOMENT"),
    )


def closing_hand_map(hand: str, d: ArrayLike, net: ArrayLike) -> tuple[Array, Array]:
    """Affine map ``s = A lambda + s0`` for the closing hand, ``lambda = -y_h``.

    Right hand: ``A = -Q(d)``, ``s0 = (-F/2, -(M + d x F)/2)``.  Left hand:
    ``A = Q(-d)``, ``s0 = (F/2, (M - d x F)/2)``, with
    ``Q(d) = [[I, 0], [skew(d), I]]`` and ``Q(d)^-1 = Q(-d)``.  ``d`` is the
    half separation in grip axes and ``net = (F, M)`` the net wrench.

    Raises:
        ValueError: if ``hand`` is not ``"L"`` or ``"R"`` or shapes are wrong.
    """
    if hand not in ("L", "R"):
        raise ValueError(f"hand must be 'L' or 'R', got {hand!r}")
    dvec, w = _vec(d, 3, "d"), _vec(net, 6, "net")
    force, moment = w[:3], w[3:]

    def q_of(vec: Array) -> Array:
        return np.block([[_I3, _Z3], [_skew(vec), _I3]])

    if hand == "R":
        return -q_of(dvec), np.concatenate(
            [-0.5 * force, -0.5 * (moment + np.cross(dvec, force))]
        )
    return q_of(-dvec), np.concatenate(
        [0.5 * force, 0.5 * (moment - np.cross(dvec, force))]
    )
