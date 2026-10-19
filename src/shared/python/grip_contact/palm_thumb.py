"""Legacy single-pair palm-thumb allocation API (HRG-H1, #13189; HRG-GS-G3, #13301).

Reporting only. For multi-pair allocation across all hand-hand pairs of a
contact topology, see :mod:`src.shared.python.grip_contact.hand_hand_allocation`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.shared.python.grip_contact.hand_hand_allocation import (
    METHOD,
    MIN_CLOSURE_FORCE_N,
    NORMAL_TOLERANCE,
    AllocatedHandContacts,
    AllocatedPairForce,
    AllocationPrior,
    Vec3,
    _check_rotation,
    _vec,
    allocate_hand_hand_from_closure,
    allocate_pair_force,
    equal_split_prior,
    pair_contact_patch,
    require_allocated_mode,
    unavailable_pair_force,
)

PALM_THUMB_LABEL = "external:palm_thumb_allocated"

#: Alias for backward compatibility with existing HRG-H1 callers.
PalmThumbForce = AllocatedPairForce


def unavailable_palm_thumb(reason: str) -> PalmThumbForce:
    """A result that could not be resolved; ``reason`` must be non-empty."""
    return unavailable_pair_force("trail_palm_lead_thumb", PALM_THUMB_LABEL, reason)


def allocate_palm_thumb(
    *,
    force_on_lead_n: Sequence[float] | np.ndarray,
    torque_on_lead_nm: Sequence[float] | np.ndarray,
    reference_point_m: Sequence[float] | np.ndarray,
    patch_point_m: Sequence[float] | np.ndarray,
    patch_normal: Sequence[float] | np.ndarray,
    static_friction: float,
    pair_name: str = "trail_palm_lead_thumb",
    label: str = PALM_THUMB_LABEL,
) -> PalmThumbForce:
    """Thin wrapper around :func:`allocate_pair_force` for backward compatibility."""
    return allocate_pair_force(
        force_on_lead_n=force_on_lead_n,
        torque_on_lead_nm=torque_on_lead_nm,
        reference_point_m=reference_point_m,
        patch=(patch_point_m, patch_normal),
        static_friction=static_friction,
        pair_name=pair_name,
        label=label,
    )


def allocate_from_closure(
    closure: Any,
    *,
    club_rotation: Sequence[Sequence[float]] | np.ndarray,
    club_origin: Sequence[float] | np.ndarray,
    patch: Any,
    static_friction: float,
) -> PalmThumbForce:
    """Allocate the palm-thumb force of a world-frame ``GripClosure``.

    The load on the lead side (the club) is the closure's second wrench,
    ``-lambda`` at the club site. ``patch`` (``point_club_m``, ``normal_club``) is
    carried to the world by ``club_rotation`` and ``club_origin``. An
    unavailable closure, or one without a torque, gives an unavailable result
    with the reason (never a zero force). Preconditions: ``club_rotation`` a
    proper 3x3 rotation, ``club_origin`` a finite 3-vector (``ValueError``).
    """
    rot = _check_rotation(club_rotation)
    origin = _vec("club_origin", club_origin)
    if not closure.available:
        return unavailable_palm_thumb(closure.reason or "closure unavailable")
    on_club = closure.wrenches[-1]
    if on_club.force_n is None or on_club.torque_nm is None:
        return unavailable_palm_thumb("closure wrench has no force or torque half")
    return allocate_palm_thumb(
        force_on_lead_n=on_club.force_n,
        torque_on_lead_nm=on_club.torque_nm,
        reference_point_m=on_club.point_m,
        patch_point_m=rot @ _vec("patch.point_club_m", patch.point_club_m) + origin,
        patch_normal=rot @ _vec("patch.normal_club", patch.normal_club),
        static_friction=static_friction,
    )


__all__ = [
    "METHOD",
    "MIN_CLOSURE_FORCE_N",
    "NORMAL_TOLERANCE",
    "PALM_THUMB_LABEL",
    "AllocatedHandContacts",
    "AllocatedPairForce",
    "AllocationPrior",
    "PalmThumbForce",
    "Vec3",
    "allocate_from_closure",
    "allocate_hand_hand_from_closure",
    "allocate_pair_force",
    "allocate_palm_thumb",
    "equal_split_prior",
    "pair_contact_patch",
    "require_allocated_mode",
    "unavailable_pair_force",
    "unavailable_palm_thumb",
]
