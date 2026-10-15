"""Bare-club versus welded-hand partition of the club body (NSX-01, #12550).

The full-body specification welds the hand solids (``LHand``,
``LHandStandoff``, ``RHand``) into the same rigid body as the club
(``Clubface Vector``): 1.421 kg for the driver spec against a bare club of
0.313 kg.  A hand-on-club wrench from Newton-Euler of that composite body
carries the inertial load of both hands.  The physical hand-on-club interface
loads only the bare club; the hand solids belong to the hand side, wrist ->
hand -> club.  This module is the one place that separates the two.

Classification is by solid provenance name: a solid whose last path segment is
``LHand``, ``LHandStandoff``, ``RHand`` or ``RHandStandoff`` is a hand solid;
every other solid of the closure's club body is bare club (head, shaft, grip
and the zero-mass ``Clubface Vector`` reference solid).

All :class:`RigidInertia` values are expressed in the specification body frame
of the club body.  Engines whose body frame differs (the MuJoCo and Pinocchio
body origin is the wrist-joint follower frame) move them with
:meth:`RigidInertia.expressed_in`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "INTERFACE_BARE_CLUB",
    "INTERFACE_LEGACY_COMPOSITE",
    "ClubSolidsPartition",
    "RigidInertia",
    "combine_inertias",
    "club_solids_partition",
    "newton_euler_keywords",
    "validate_interface",
]

INTERFACE_BARE_CLUB = "hand_on_bare_club"
"""Net wrench the hands exert on the bare club (head, shaft, grip only)."""

INTERFACE_LEGACY_COMPOSITE = "wrist_on_hands_plus_club"
"""Legacy composite value: club body including both welded hands."""

INTERFACES = (INTERFACE_BARE_CLUB, INTERFACE_LEGACY_COMPOSITE)

MASS_CHECK_TOLERANCE_KG = 1e-9
_HAND_SOLID = re.compile(r"(?:^|/)[LR]Hand(?:Standoff)?$")
_SUFFIX = re.compile(r"[^/]+$")


@dataclass(frozen=True)
class RigidInertia:
    """Mass, centre of mass and inertia about the centre of mass (one frame)."""

    mass_kg: float
    com_m: tuple[float, float, float]
    inertia_com_kg_m2: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.mass_kg) or self.mass_kg < 0.0:
            raise ValueError(f"mass_kg must be finite and >= 0, got {self.mass_kg}")
        com = np.asarray(self.com_m, dtype=float)
        inertia = np.asarray(self.inertia_com_kg_m2, dtype=float)
        if com.shape != (3,) or not np.isfinite(com).all():
            raise ValueError("com_m must be a finite 3-vector")
        if inertia.shape != (3, 3) or not np.isfinite(inertia).all():
            raise ValueError("inertia_com_kg_m2 must be a finite 3x3 matrix")
        object.__setattr__(self, "com_m", (float(com[0]), float(com[1]), float(com[2])))
        object.__setattr__(self, "inertia_com_kg_m2", inertia)

    def expressed_in(self, transform: ArrayLike) -> RigidInertia:
        """Same body expressed in a frame given by ``transform`` (4x4, new<-old).

        Postcondition: mass is unchanged, the centre of mass is mapped as a
        point and the inertia about it is rotated, ``R I R^T``.
        """
        t = np.asarray(transform, dtype=float)
        if t.shape != (4, 4) or not np.isfinite(t).all():
            raise ValueError("transform must be a finite 4x4 matrix")
        rot = t[:3, :3]
        com = rot @ np.asarray(self.com_m) + t[:3, 3]
        return RigidInertia(
            self.mass_kg,
            (float(com[0]), float(com[1]), float(com[2])),
            rot @ self.inertia_com_kg_m2 @ rot.T,
        )


class ClubSolidsPartition(NamedTuple):
    """``(bare_club_inertia, welded_hand_inertias)`` of the club body."""

    bare_club: RigidInertia
    welded_hands: Mapping[str, RigidInertia]

    def hands_by_side(self) -> dict[str, RigidInertia]:
        """Welded hand solids combined per side, keys ``"L"`` and ``"R"``.

        Raises:
            ValueError: if a welded solid name does not start with ``L`` or ``R``.
        """
        groups: dict[str, list[RigidInertia]] = {}
        for name, inertia in self.welded_hands.items():
            if name[:1] not in ("L", "R"):
                raise ValueError(f"cannot assign welded solid {name!r} to a hand side")
            groups.setdefault(name[0], []).append(inertia)
        return {side: combine_inertias(items) for side, items in groups.items()}


def combine_inertias(items: Sequence[RigidInertia]) -> RigidInertia:
    """Combine rigid inertias in one frame about their joint centre of mass."""
    return _combine(
        [(i.mass_kg, np.asarray(i.com_m), i.inertia_com_kg_m2) for i in items]
    )


def _placed(solid: Mapping[str, Any]) -> tuple[float, np.ndarray, np.ndarray]:
    t = np.asarray(solid["placement"], dtype=float)
    rot = t[:3, :3]
    com = rot @ np.asarray(solid["com_m"], dtype=float) + t[:3, 3]
    inertia = rot @ np.asarray(solid["inertia_com_kg_m2"], dtype=float) @ rot.T
    return float(solid["mass_kg"]), com, inertia


def _compose(solids: Sequence[Mapping[str, Any]]) -> RigidInertia:
    """Combine solids with the parallel-axis theorem about the combined CoM."""
    return _combine([_placed(s) for s in solids])


def _combine(placed: Sequence[tuple[float, np.ndarray, np.ndarray]]) -> RigidInertia:
    mass = sum(m for m, _, _ in placed)
    if mass <= 0.0:
        return RigidInertia(0.0, (0.0, 0.0, 0.0), np.zeros((3, 3)))
    com = np.sum([m * c for m, c, _ in placed], axis=0) / mass
    inertia = np.zeros((3, 3))
    for m, c, i_c in placed:
        d = c - com
        inertia += i_c + m * ((d @ d) * np.eye(3) - np.outer(d, d))
    return RigidInertia(mass, (float(com[0]), float(com[1]), float(com[2])), inertia)


def club_solids_partition(spec: Mapping[str, Any]) -> ClubSolidsPartition:
    """Split the closure club body of ``spec`` into bare club and welded hands.

    Returns:
        ``(bare_club, welded_hands)`` in the specification body frame.
        ``welded_hands`` maps each hand solid's last name segment to its own
        :class:`RigidInertia`.

    Raises:
        ValueError: if the spec has no closure, the closure body is missing or
            has no bare-club solids, or ``spec["club"]["total_mass_kg"]`` is
            present and differs from the bare mass by more than 1e-9 kg.  A
            spec without a ``club`` record (full-body v1/v2) skips only that
            mass check.
    """
    closure = spec.get("closure")
    if not closure or "body_b" not in closure:
        raise ValueError("spec has no closure body_b to partition")
    name = closure["body_b"]
    body = next((b for b in spec.get("bodies", []) if b["name"] == name), None)
    if body is None:
        raise ValueError(f"closure club body {name!r} not found in spec bodies")
    bare_solids, hands = [], {}
    for solid in body["solids"]:
        if _HAND_SOLID.search(solid["name"]):
            suffix = _SUFFIX.search(solid["name"])
            hands[suffix.group(0) if suffix else solid["name"]] = _compose([solid])
        else:
            bare_solids.append(solid)
    if not bare_solids:
        raise ValueError(f"club body {name!r} has no club solids (head/shaft/grip)")
    bare = _compose(bare_solids)
    total = (spec.get("club") or {}).get("total_mass_kg")
    if total is not None and abs(bare.mass_kg - float(total)) > MASS_CHECK_TOLERANCE_KG:
        raise ValueError(
            f"bare club mass {bare.mass_kg!r} kg differs from club.total_mass_kg "
            f"{float(total)!r} kg by more than {MASS_CHECK_TOLERANCE_KG:g} kg; "
            "the solid partition is wrong or the spec is inconsistent"
        )
    return ClubSolidsPartition(bare, hands)


def newton_euler_keywords(
    inertia: RigidInertia,
    *,
    rot_world_from_body: ArrayLike,
    origin_world: ArrayLike,
    omega: ArrayLike,
    alpha: ArrayLike,
    accel_origin: ArrayLike,
    gravity: ArrayLike,
) -> dict[str, Any]:
    """Club keywords of :func:`holding_hand_wrench` from body-origin kinematics.

    ``inertia`` is in the engine body frame whose origin is ``origin_world``
    with orientation ``rot_world_from_body``; ``omega``, ``alpha`` and the
    classical acceleration ``accel_origin`` of the body origin are world
    vectors.  The centre of mass acceleration is
    ``a_o + alpha x r + omega x (omega x r)`` with ``r = R c``.
    """
    rot = np.asarray(rot_world_from_body, dtype=float)
    origin, w, al, a_o, g = (
        np.asarray(x, dtype=float)
        for x in (origin_world, omega, alpha, accel_origin, gravity)
    )
    r = rot @ np.asarray(inertia.com_m)
    return {
        "mass_kg": float(inertia.mass_kg),
        "gravity_m_s2": g,
        "com_m": origin + r,
        "com_acceleration_m_s2": a_o + np.cross(al, r) + np.cross(w, np.cross(w, r)),
        "inertia_world_kg_m2": rot @ inertia.inertia_com_kg_m2 @ rot.T,
        "angular_velocity_rad_s": w,
        "angular_acceleration_rad_s2": al,
    }


def validate_interface(interface: str) -> str:
    """Return ``interface`` if it is a known hand-on-club interface label.

    Raises:
        ValueError: for any other label.
    """
    if interface not in INTERFACES:
        raise ValueError(f"interface must be one of {INTERFACES}, got {interface!r}")
    return interface
