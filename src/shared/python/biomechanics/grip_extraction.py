"""Engine-agnostic helpers that turn solver output into grip analyses.

Issue #11714 (GCV-8), epic #11706, ADR-0052.  Every wrench follows the
:mod:`grip_wrench` convention: exerted by the hand **on the club**, world
frame, SI units.  Nothing here imports an engine; the per-engine adapters
supply the club kinematics and the multiplier-derived closing wrench.

Holding-hand wrench from club Newton-Euler
-------------------------------------------
A club is held by a *closing* hand (the one whose weld closes the loop, its
wrench comes from the constraint multiplier) and a *holding* hand (the one
the kinematic tree already carries, which has no multiplier).  The holding
wrench follows from rigid-body dynamics of the club (mass ``m``, centre of
mass ``c``, world inertia ``I``) with gravity ``g``::

    F_close + F_hold + m g          = m a_c
    tau_close + tau_hold + sum_h (r_h - c) x F_h = I alpha + omega x (I omega)

so ``F_hold = m (a_c - g) - F_close`` and ``tau_hold`` follows likewise about
the club centre of mass, with ``r_h`` the hand grip points.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.biomechanics.grip_wrench import (
    GripAnalysis,
    HandWrench,
    SplitMethod,
    analyze_grip,
)

Vec3 = tuple[float, float, float]

__all__ = [
    "allocation_grip_analysis",
    "closing_side_from_closure",
    "hand_from_arrays",
    "closure_and_club_analysis",
    "holding_hand_wrench",
    "net_only_analysis",
    "unavailable_analysis",
]


def _v3(value: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 3-vector, got shape {arr.shape}")
    return arr


def _t3(arr: np.ndarray) -> Vec3:
    return (float(arr[0]), float(arr[1]), float(arr[2]))


_CLUB_KEYS = (
    "mass_kg",
    "gravity_m_s2",
    "com_m",
    "com_acceleration_m_s2",
    "inertia_world_kg_m2",
    "angular_velocity_rad_s",
    "angular_acceleration_rad_s2",
)


def holding_hand_wrench(
    *,
    closing_force_n: ArrayLike,
    closing_torque_nm: ArrayLike,
    closing_point_m: ArrayLike,
    holding_point_m: ArrayLike,
    **club: Any,
) -> tuple[Vec3, Vec3]:
    """Force and free torque of the holding hand from club Newton-Euler.

    Args:
        closing_force_n, closing_torque_nm, closing_point_m: wrench exerted
            on the club by the closing hand and its application point.
        holding_point_m: grip point of the holding hand.
        **club: the club keywords ``mass_kg`` (positive), ``gravity_m_s2``
            (world ``g``), ``com_m`` and ``com_acceleration_m_s2`` (centre of
            mass and its acceleration), ``inertia_world_kg_m2`` (3x3 about
            the centre of mass, world axes), ``angular_velocity_rad_s`` and
            ``angular_acceleration_rad_s2`` (world ``omega`` and ``alpha``).

    Returns:
        ``(force_n, torque_nm)`` exerted by the holding hand on the club at
        ``holding_point_m``; the torque is a free (couple) torque.

    Raises:
        ValueError: on a missing or unknown club keyword, non-positive mass,
            non-finite input or a bad inertia.
    """
    missing = [k for k in _CLUB_KEYS if k not in club]
    unknown = sorted(set(club) - set(_CLUB_KEYS))
    if missing or unknown:
        raise ValueError(
            f"club keywords missing {missing} / unknown {unknown}; "
            f"expected {list(_CLUB_KEYS)}"
        )
    mass_kg = club["mass_kg"]
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError(f"mass_kg must be positive and finite, got {mass_kg}")
    inertia = np.asarray(club["inertia_world_kg_m2"], dtype=np.float64)
    if inertia.shape != (3, 3) or not np.isfinite(inertia).all():
        raise ValueError("inertia_world_kg_m2 must be a finite 3x3 matrix")
    g = _v3(club["gravity_m_s2"], "gravity_m_s2")
    c = _v3(club["com_m"], "com_m")
    a_c = _v3(club["com_acceleration_m_s2"], "com_acceleration_m_s2")
    omega = _v3(club["angular_velocity_rad_s"], "angular_velocity_rad_s")
    alpha = _v3(club["angular_acceleration_rad_s2"], "angular_acceleration_rad_s2")
    f_close = _v3(closing_force_n, "closing_force_n")
    t_close = _v3(closing_torque_nm, "closing_torque_nm")
    r_close = _v3(closing_point_m, "closing_point_m")
    r_hold = _v3(holding_point_m, "holding_point_m")

    f_hold = mass_kg * (a_c - g) - f_close
    rate_of_angular_momentum = inertia @ alpha + np.cross(omega, inertia @ omega)
    t_hold = (
        rate_of_angular_momentum
        - t_close
        - np.cross(r_close - c, f_close)
        - np.cross(r_hold - c, f_hold)
    )
    return _t3(f_hold), _t3(t_hold)


def net_only_analysis(
    *,
    point_m: ArrayLike,
    force_on_club_n: ArrayLike,
    torque_on_club_nm: ArrayLike | None,
    split_method: SplitMethod,
    reason: str,
    metadata: Mapping[str, Any] | None = None,
) -> GripAnalysis:
    """Analysis carrying only the net wrench at ``point_m`` (no per-hand split).

    Used when an engine supplies one 6-D grip wrench (for example the
    Pinocchio allocation ``lambda_grip``).  ``reason`` records why the
    per-hand values are absent; they stay ``None``, never zero.
    """
    point = _t3(_v3(point_m, "point_m"))
    force = _t3(_v3(force_on_club_n, "force_on_club_n"))
    couple = None if torque_on_club_nm is None else _t3(_v3(torque_on_club_nm, "t"))
    return GripAnalysis(
        left=None,
        right=None,
        midpoint_m=point,
        net_force_n=force,
        couple_at_midpoint_nm=couple,
        contact_force_moment_nm=None,
        applied_free_torque_nm=None,
        mof_left_nm=None,
        mof_right_nm=None,
        split_method=split_method,
        unavailable_reason=reason,
        metadata=dict(metadata or {}),
    )


def unavailable_analysis(
    reason: str, metadata: Mapping[str, Any] | None = None
) -> GripAnalysis:
    """Explicitly unavailable analysis: every quantity ``None`` with a reason."""
    if not reason:
        raise ValueError("an unavailable analysis needs a reason")
    return GripAnalysis(
        left=None,
        right=None,
        midpoint_m=None,
        net_force_n=None,
        couple_at_midpoint_nm=None,
        contact_force_moment_nm=None,
        applied_free_torque_nm=None,
        mof_left_nm=None,
        mof_right_nm=None,
        split_method="unavailable",
        unavailable_reason=reason,
        metadata=dict(metadata or {}),
    )


def hand_from_arrays(
    side: str, point: ArrayLike, force: ArrayLike, torque: ArrayLike | None
) -> HandWrench:
    """Build a :class:`HandWrench` from arrays (world frame, on the club)."""
    return HandWrench(
        side=side,
        point_m=_t3(_v3(point, "point")),
        force_on_club_n=_t3(_v3(force, "force")),
        torque_on_club_nm=None if torque is None else _t3(_v3(torque, "torque")),
    )


def closing_side_from_closure(closure: Mapping[str, Any]) -> str:
    """Hand (``"L"``/``"R"``) whose weld closes the loop, from the spec closure.

    The full-body specification names the closure after the closing hand
    (``.../RightHandOnClubForce`` with body ``.../RHandStandoff``).

    Raises:
        ValueError: if neither hand can be identified.
    """
    text = f"{closure.get('name', '')} {closure.get('body_a', '')}"
    right = "RightHand" in text or "RHand" in text
    left = "LeftHand" in text or "LHand" in text
    if right == left:
        raise ValueError(f"cannot identify the closing hand from closure {text!r}")
    return "R" if right else "L"


def closure_and_club_analysis(
    *,
    closing_side: str,
    closing_point_m: ArrayLike,
    closing_force_n: ArrayLike,
    closing_torque_nm: ArrayLike,
    holding_point_m: ArrayLike,
    club: Mapping[str, Any],
    split_method: SplitMethod,
    metadata: Mapping[str, Any] | None = None,
) -> GripAnalysis:
    """Two-hand analysis from a closure multiplier plus club Newton-Euler.

    ``club`` carries the :func:`holding_hand_wrench` club keywords
    (``mass_kg``, ``gravity_m_s2``, ``com_m``, ``com_acceleration_m_s2``,
    ``inertia_world_kg_m2``, ``angular_velocity_rad_s``,
    ``angular_acceleration_rad_s2``).
    """
    if closing_side not in ("L", "R"):
        raise ValueError(f"closing_side must be 'L' or 'R', got {closing_side!r}")
    f_hold, t_hold = holding_hand_wrench(
        **club,
        closing_force_n=closing_force_n,
        closing_torque_nm=closing_torque_nm,
        closing_point_m=closing_point_m,
        holding_point_m=holding_point_m,
    )
    holding_side = "L" if closing_side == "R" else "R"
    closing = hand_from_arrays(
        closing_side, closing_point_m, closing_force_n, closing_torque_nm
    )
    holding = hand_from_arrays(holding_side, holding_point_m, f_hold, t_hold)
    by_side = {closing_side: closing, holding_side: holding}
    meta = {"closing_hand": closing_side, "holding_hand": holding_side}
    meta.update(metadata or {})
    return analyze_grip(
        by_side["L"], by_side["R"], split_method=split_method, metadata=meta
    )


def allocation_grip_analysis(
    lambda_grip: ArrayLike,
    *,
    point_m: ArrayLike,
    ordering: str = "force_torque",
    load_on: str = "human",
    rotation_world_from_frame: ArrayLike | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GripAnalysis:
    """Net grip analysis from the allocator's single 6-D ``lambda_grip``.

    The Pinocchio allocation (``contact_force_allocator``) solves one 6-D
    closure wrench for the pair of hands, so only the net at ``point_m`` is
    available (``split_method="allocation"``); the per-hand wrenches stay
    ``None`` rather than being guessed.

    Args:
        lambda_grip: 6-vector in ``ordering`` (``"force_torque"`` or
            ``"torque_force"``), in a frame rotated into the world by
            ``rotation_world_from_frame`` (identity if omitted).
        point_m: world point the wrench acts at (typically the grip midpoint).
        load_on: ``"human"`` when ``J.T @ lambda`` is the load on the human
            model (the allocator convention), so the wrench on the club is
            ``-lambda``; ``"club"`` when ``lambda`` is already the club load.

    Raises:
        ValueError: on a bad shape, ordering, ``load_on`` or rotation.
    """
    lam = np.asarray(lambda_grip, dtype=np.float64)
    if lam.shape != (6,) or not np.isfinite(lam).all():
        raise ValueError(f"lambda_grip must be a finite 6-vector, got {lam.shape}")
    if ordering not in ("force_torque", "torque_force"):
        raise ValueError("ordering must be 'force_torque' or 'torque_force'")
    if load_on not in ("human", "club"):
        raise ValueError("load_on must be 'human' or 'club'")
    force, torque = (
        (lam[:3], lam[3:]) if ordering == "force_torque" else (lam[3:], lam[:3])
    )
    if rotation_world_from_frame is not None:
        rot = np.asarray(rotation_world_from_frame, dtype=np.float64)
        if rot.shape != (3, 3) or not np.isfinite(rot).all():
            raise ValueError("rotation_world_from_frame must be a finite 3x3 matrix")
        force, torque = rot @ force, rot @ torque
    sign = -1.0 if load_on == "human" else 1.0
    return net_only_analysis(
        point_m=point_m,
        force_on_club_n=sign * force,
        torque_on_club_nm=sign * torque,
        split_method="allocation",
        reason="allocation yields one net 6-D wrench; left/right split unavailable",
        metadata={"load_on": load_on, **dict(metadata or {})},
    )
