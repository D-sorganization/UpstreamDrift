"""Shared grip wrench core: per-hand wrench, midpoint net force and couple.

Issue #11713 (GCV-7), epic #11706, ADR-0052.

Sign convention (binding): every wrench is the loading **exerted by the hand
on the club**, expressed in the world frame in SI units.  Hand ``h`` in
``{L, R}`` acts at grip point ``r_h`` with force ``F_h`` and free torque
``tau_h``.  With ``r_M = (r_L + r_R) / 2``::

    R   = F_L + F_R                                  (drawn at r_M)
    M_M = sum_h (r_h - r_M) x F_h + tau_L + tau_R    (equivalent couple)

``contact_force_moment`` is the first sum, ``applied_free_torque`` the second;
``MOF_h = (r_h - r_M) x F_h`` is each hand's moment of force about the
midpoint.  Reference derivation: ``docs/research/proximal_distal_energy_transfer/``
section "Reduction of Two Contacts to an Equivalent Wrench".

Wrench transport is delegated to
:func:`force_overlay.conversions.move_wrench_point`; no second transport
routine lives here.  A quantity that cannot be computed is ``None`` (NaN in
:class:`GripSeries`) with a reason string, never zero.

The left/right split of two rigid welds is indeterminate and is set by the
solver; every result therefore records ``split_method``.  Two compliant grip
models make the split determinate (issue #11739, OSV-7):

* ``"bushing"``: one six-axis ``BushingForce`` per hand; each hand wrench is the
  bushing record on the club (frame2 of the force), in the world frame, with
  the torque taken about that hand's grip point.
* ``"contact"``: reserved for the distributed-contact grip (summed
  ``ElasticFoundationForce`` records per hand); not produced yet.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.force_overlay.contracts import OverlayWrench, WrenchKind
from src.shared.python.force_overlay.conversions import move_wrench_point
from src.shared.python.motion_matching.force_torque import (
    SpatialWrench,
    validate_vec3,
)

Vec3 = tuple[float, float, float]
SplitMethod = Literal[
    "constraint_multiplier",
    "efc_force",
    "allocation",
    "logged",
    "bushing",
    "contact",
    "unavailable",
]
SPLIT_METHODS: tuple[str, ...] = get_args(SplitMethod)

_BODY = "club"
_ZERO: Vec3 = (0.0, 0.0, 0.0)

__all__ = [
    "allocate_min_norm",
    "SPLIT_METHODS",
    "GripAnalysis",
    "GripSeries",
    "HandWrench",
    "SplitMethod",
    "about_axis",
    "analyze_grip",
    "to_contact_reaction_wrench",
    "to_overlay_wrenches",
]


def _tuple3(v: ArrayLike) -> Vec3:
    a = np.asarray(v, dtype=np.float64)
    return (float(a[0]), float(a[1]), float(a[2]))


@dataclass(frozen=True)
class HandWrench:
    """Loading exerted by one hand on the club at its grip point (world frame).

    ``torque_on_club_nm`` is the free torque; ``None`` means the engine does
    not supply it (unavailable, not zero).
    """

    side: str
    point_m: Vec3
    force_on_club_n: Vec3
    torque_on_club_nm: Vec3 | None = None

    def __post_init__(self) -> None:
        if self.side not in ("L", "R"):
            raise ValueError(f"side must be 'L' or 'R', got {self.side!r}")
        object.__setattr__(self, "point_m", validate_vec3(self.point_m, "point_m"))
        object.__setattr__(
            self,
            "force_on_club_n",
            validate_vec3(self.force_on_club_n, "force_on_club_n"),
        )
        if self.torque_on_club_nm is not None:
            object.__setattr__(
                self,
                "torque_on_club_nm",
                validate_vec3(self.torque_on_club_nm, "torque_on_club_nm"),
            )


@dataclass(frozen=True)
class GripAnalysis:
    """Per-hand wrenches, midpoint net force and equivalent couple.

    Unavailable quantities are ``None``; ``unavailable_reason`` explains why
    (empty when everything is available).
    """

    left: HandWrench | None
    right: HandWrench | None
    midpoint_m: Vec3 | None
    net_force_n: Vec3 | None
    couple_at_midpoint_nm: Vec3 | None
    contact_force_moment_nm: Vec3 | None
    applied_free_torque_nm: Vec3 | None
    mof_left_nm: Vec3 | None
    mof_right_nm: Vec3 | None
    split_method: SplitMethod
    couple_local_nm: Vec3 | None = None
    net_force_local_n: Vec3 | None = None
    unavailable_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def net_wrench_at(self, point_m: ArrayLike) -> OverlayWrench:
        """Net wrench moved to ``point_m`` via the shared transport helper.

        Raises:
            ValueError: if the net force or couple is unavailable.
        """
        if (
            self.net_force_n is None
            or self.couple_at_midpoint_nm is None
            or self.midpoint_m is None
        ):
            raise ValueError(
                f"net wrench unavailable: {self.unavailable_reason or 'no data'}"
            )
        at_mid = OverlayWrench(
            kind=WrenchKind.GRIP,
            label="grip:net_wrench",
            body=_BODY,
            point_m=self.midpoint_m,
            force_n=self.net_force_n,
            torque_nm=self.couple_at_midpoint_nm,
            source="grip_wrench:analysis",
        )
        return move_wrench_point(at_mid, _tuple3(point_m))


def about_axis(vector: ArrayLike, axis: ArrayLike) -> float:
    """Component of ``vector`` along ``axis`` (axis is normalised).

    Raises:
        ValueError: for a zero-length or non-finite axis.
    """
    v = np.asarray(validate_vec3(_tuple3(vector), "vector"))
    a = np.asarray(validate_vec3(_tuple3(axis), "axis"))
    n = float(np.linalg.norm(a))
    if n == 0.0:
        raise ValueError("axis must be non-zero")
    return float(v @ a / n)


def _check_rotation(rotation: ArrayLike) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError(f"club_rotation must be a finite 3x3 matrix, got {r.shape}")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6):
        raise ValueError("club_rotation must be orthonormal")
    return r


def _hand_overlay(h: HandWrench, torque: Vec3 | None) -> OverlayWrench:
    return OverlayWrench(
        kind=WrenchKind.GRIP,
        label=f"grip:hand_{'left' if h.side == 'L' else 'right'}",
        body=_BODY,
        point_m=h.point_m,
        force_n=h.force_on_club_n,
        torque_nm=torque,
        source="grip_wrench:input",
    )


def _mof(h: HandWrench, midpoint: Vec3) -> Vec3:
    """Moment of force of one hand about the midpoint (shared transport)."""
    pure_force = _hand_overlay(h, _ZERO)
    moved = move_wrench_point(pure_force, midpoint)
    assert moved.torque_nm is not None
    return moved.torque_nm


def _validate_slot(h: HandWrench | None, side: str) -> None:
    if h is not None and (not isinstance(h, HandWrench) or h.side != side):
        raise ValueError(f"{side} slot requires a HandWrench with side={side!r}")


def _unavailable(
    left: HandWrench | None,
    right: HandWrench | None,
    split_method: SplitMethod,
    metadata: Mapping[str, Any],
) -> GripAnalysis:
    missing = [s for s, h in (("L", left), ("R", right)) if h is None]
    reason = f"hand wrench missing for: {', '.join(missing)}"
    return GripAnalysis(
        left=left,
        right=right,
        midpoint_m=None,
        net_force_n=None,
        couple_at_midpoint_nm=None,
        contact_force_moment_nm=None,
        applied_free_torque_nm=None,
        mof_left_nm=None,
        mof_right_nm=None,
        split_method=split_method,
        unavailable_reason=reason,
        metadata=dict(metadata),
    )


def analyze_grip(
    left: HandWrench | None,
    right: HandWrench | None,
    *,
    split_method: SplitMethod,
    club_rotation: ArrayLike | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GripAnalysis:
    """Reduce two hand wrenches on the club to the midpoint net wrench.

    Args:
        left, right: hand wrenches exerted on the club (or ``None``).
        split_method: how the L/R split was obtained (see ``SPLIT_METHODS``).
        club_rotation: optional world-from-club-local rotation ``R``; the
            club-local couple is ``R^T M_M``.
        metadata: solver metadata copied into the result.

    Postconditions: ``couple == contact_force_moment + applied_free_torque``
    and ``contact_force_moment == mof_left + mof_right``.  Any quantity that
    cannot be computed is ``None`` with ``unavailable_reason`` set.

    Raises:
        ValueError: on invalid ``split_method``, wrong side slot or rotation.
    """
    if split_method not in SPLIT_METHODS:
        raise ValueError(f"split_method must be one of {SPLIT_METHODS}")
    _validate_slot(left, "L")
    _validate_slot(right, "R")
    rot = None if club_rotation is None else _check_rotation(club_rotation)
    meta = metadata or {}
    if left is None or right is None:
        return _unavailable(left, right, split_method, meta)

    mid = _tuple3((np.array(left.point_m) + np.array(right.point_m)) / 2.0)
    net = _tuple3(np.array(left.force_on_club_n) + np.array(right.force_on_club_n))
    mof_l, mof_r = _mof(left, mid), _mof(right, mid)
    cfm = _tuple3(np.array(mof_l) + np.array(mof_r))

    torques = (left.torque_on_club_nm, right.torque_on_club_nm)
    free: Vec3 | None = None
    couple: Vec3 | None = None
    reason = ""
    if torques[0] is None or torques[1] is None:
        reason = "free torque unavailable for at least one hand"
    else:
        free = _tuple3(np.array(torques[0]) + np.array(torques[1]))
        couple = _tuple3(np.array(cfm) + np.array(free))

    couple_local = net_local = None
    if rot is not None:
        net_local = _tuple3(rot.T @ np.array(net))
        if couple is not None:
            couple_local = _tuple3(rot.T @ np.array(couple))
    return GripAnalysis(
        left=left,
        right=right,
        midpoint_m=mid,
        net_force_n=net,
        couple_at_midpoint_nm=couple,
        contact_force_moment_nm=cfm,
        applied_free_torque_nm=free,
        mof_left_nm=mof_l,
        mof_right_nm=mof_r,
        split_method=split_method,
        couple_local_nm=couple_local,
        net_force_local_n=net_local,
        unavailable_reason=reason,
        metadata=dict(meta),
    )


def allocate_min_norm(
    g: GripAnalysis,
) -> tuple[Vec3, Vec3]:
    """Minimum-norm force split ``(F_L, F_R)`` reproducing the net wrench.

    Reference allocation for comparing a determinate grip model (``bushing``)
    with a rigid weld, whose split is solver-defined: with both free torques
    taken as zero, ``F_L + F_R = R`` and ``h x (F_R - F_L) = M`` (``h`` the
    half hand separation) are solved with minimum ``|F_L|^2 + |F_R|^2``.  The
    component of ``M`` along ``h`` cannot be produced by forces and is dropped
    (it needs a free torque); the axial force is split equally.

    Raises:
        ValueError: if the net wrench is unavailable or the hands coincide.
    """
    if g.left is None or g.right is None or g.net_force_n is None:
        raise ValueError("net wrench unavailable; cannot allocate")
    if g.couple_at_midpoint_nm is None:
        raise ValueError("couple unavailable; cannot allocate")
    h = (np.array(g.right.point_m) - np.array(g.left.point_m)) / 2.0
    h2 = float(h @ h)
    if h2 <= 0.0:
        raise ValueError("hand grip points coincide")
    net = np.array(g.net_force_n)
    diff = np.cross(np.array(g.couple_at_midpoint_nm), h) / h2
    return _tuple3((net - diff) / 2.0), _tuple3((net + diff) / 2.0)


def to_overlay_wrenches(g: GripAnalysis, *, source: str) -> list[OverlayWrench]:
    """Build ADR-0052 ``GRIP`` overlay wrenches; unavailable ones are omitted.

    Labels: ``grip:hand_left``, ``grip:hand_right`` (at each grip point),
    ``grip:net_midpoint`` (force only), ``grip:couple_midpoint`` (torque only),
    ``grip:mof_left`` / ``grip:mof_right`` (torque only, about the midpoint).
    """
    out: list[OverlayWrench] = []

    def add(label: str, point: Vec3, f: Vec3 | None, t: Vec3 | None) -> None:
        out.append(
            OverlayWrench(WrenchKind.GRIP, label, _BODY, point, f, t, source=source)
        )

    for hand, name in ((g.left, "left"), (g.right, "right")):
        if hand is not None:
            add(
                f"grip:hand_{name}",
                hand.point_m,
                hand.force_on_club_n,
                hand.torque_on_club_nm,
            )
    mid = g.midpoint_m
    if mid is None:
        return out
    if g.net_force_n is not None:
        add("grip:net_midpoint", mid, g.net_force_n, None)
    if g.couple_at_midpoint_nm is not None:
        add("grip:couple_midpoint", mid, None, g.couple_at_midpoint_nm)
    for name, mof in (("left", g.mof_left_nm), ("right", g.mof_right_nm)):
        if mof is not None:
            add(f"grip:mof_{name}", mid, None, mof)
    return out


def to_contact_reaction_wrench(g: GripAnalysis) -> SpatialWrench | None:
    """Net midpoint wrench for ``ContactReaction.grip_wrench`` (or ``None``)."""
    if g.midpoint_m is None or g.net_force_n is None:
        return None
    if g.couple_at_midpoint_nm is None:
        return None
    return SpatialWrench(
        application_frame="world",
        point_m=g.midpoint_m,
        force_n=g.net_force_n,
        torque_nm=g.couple_at_midpoint_nm,
    )


_NAN3 = (math.nan, math.nan, math.nan)


@dataclass(frozen=True)
class GripSeries:
    """Time series of grip analyses; NaN marks unavailable samples."""

    time_s: np.ndarray
    midpoint_m: np.ndarray
    net_force_n: np.ndarray
    couple_nm: np.ndarray
    contact_force_moment_nm: np.ndarray
    applied_free_torque_nm: np.ndarray
    mof_left_nm: np.ndarray
    mof_right_nm: np.ndarray
    left_force_n: np.ndarray
    right_force_n: np.ndarray
    net_force_local_n: np.ndarray
    couple_local_nm: np.ndarray
    split_method: tuple[str, ...]
    unavailable_reason: tuple[str, ...]

    @classmethod
    def from_analyses(
        cls, time_s: Sequence[float], analyses: Sequence[GripAnalysis]
    ) -> GripSeries:
        """Stack analyses into arrays (per-hand forces, world and club-local).

        Raises:
            ValueError: if lengths differ.
        """
        if len(time_s) != len(analyses):
            raise ValueError("time_s and analyses must have equal length")

        def stack(attr: str) -> np.ndarray:
            rows = [getattr(a, attr) or _NAN3 for a in analyses]
            return np.array(rows, dtype=np.float64).reshape(len(analyses), 3)

        def hand_force(side: str) -> np.ndarray:
            rows = []
            for a in analyses:
                hand = a.left if side == "L" else a.right
                rows.append(_NAN3 if hand is None else hand.force_on_club_n)
            return np.array(rows, dtype=np.float64).reshape(len(analyses), 3)

        return cls(
            time_s=np.asarray(time_s, dtype=np.float64),
            midpoint_m=stack("midpoint_m"),
            net_force_n=stack("net_force_n"),
            couple_nm=stack("couple_at_midpoint_nm"),
            contact_force_moment_nm=stack("contact_force_moment_nm"),
            applied_free_torque_nm=stack("applied_free_torque_nm"),
            mof_left_nm=stack("mof_left_nm"),
            mof_right_nm=stack("mof_right_nm"),
            left_force_n=hand_force("L"),
            right_force_n=hand_force("R"),
            net_force_local_n=stack("net_force_local_n"),
            couple_local_nm=stack("couple_local_nm"),
            split_method=tuple(a.split_method for a in analyses),
            unavailable_reason=tuple(a.unavailable_reason for a in analyses),
        )

    def to_dataframe(self) -> Any:
        """Return a pandas DataFrame with one row per sample."""
        import pandas as pd

        cols: dict[str, Any] = {"time_s": self.time_s}
        blocks = {
            "midpoint": (self.midpoint_m, "m"),
            "net_force": (self.net_force_n, "n"),
            "left_force": (self.left_force_n, "n"),
            "right_force": (self.right_force_n, "n"),
            "couple": (self.couple_nm, "nm"),
            "contact_moment": (self.contact_force_moment_nm, "nm"),
            "free_torque": (self.applied_free_torque_nm, "nm"),
            "mof_left": (self.mof_left_nm, "nm"),
            "mof_right": (self.mof_right_nm, "nm"),
        }
        for name, (arr, unit) in blocks.items():
            for i, axis in enumerate("xyz"):
                cols[f"{name}_{axis}_{unit}"] = arr[:, i]
        cols["split_method"] = list(self.split_method)
        cols["unavailable_reason"] = list(self.unavailable_reason)
        return pd.DataFrame(cols)
