"""Engine contact wrenches to the ground-reaction breakdown (GCV-2, #11708).

Every engine source already emits one ``CONTACT`` :class:`OverlayWrench` per
contact, with the force the ground applies *to* the body at the contact point
(and an optional contact torque).  This module groups those wrenches by foot
(:func:`foot_of_body`) and runs the one shared analysis
(:func:`analyze_ground_reaction`) so each engine reports per-foot and net GRF,
CoP, free moment and moment about the centre of mass identically.

Callers pass **ground** contacts only; foot-versus-club contacts are not
ground reaction and must be filtered by the engine adapter that knows the
contact partner.  A contact wrench without a torque contributes zero torque
(point contacts exert none), which differs from an unavailable CoP: a foot
with no loaded contact has no CoP and reports no overlay.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.biomechanics.foot_membership import (
    FOOT_SIDES,
    Side,
    foot_of_body,
)
from src.shared.python.biomechanics.ground_reaction import (
    COP_MIN_FZ_N,
    ContactSet,
    GroundReactionBreakdown,
    analyze_ground_reaction,
    to_overlay_wrenches,
)
from src.shared.python.force_overlay.contracts import OverlayWrench, WrenchKind

__all__ = [
    "foot_contact_sets",
    "ground_reaction_breakdown",
    "ground_reaction_overlay",
]

FootOf = Callable[[str], Side | None]


def foot_contact_sets(
    wrenches: Iterable[OverlayWrench], *, foot_of: FootOf = foot_of_body
) -> dict[str, ContactSet]:
    """Group ``CONTACT`` wrenches by foot into world-frame contact sets.

    Postcondition: the result always has both ``"left"`` and ``"right"`` keys
    (an empty set for a foot with no wrenches); wrenches on bodies that are not
    feet, or that are not ``CONTACT`` kind or carry no force, are dropped.
    """
    rows: dict[str, list[tuple[tuple, tuple, tuple]]] = {s: [] for s in FOOT_SIDES}
    for w in wrenches:
        if not isinstance(w, OverlayWrench):
            raise TypeError("wrenches must be OverlayWrench instances")
        if w.kind is not WrenchKind.CONTACT or w.force_n is None:
            continue
        side = foot_of(w.body)
        if side is None:
            continue
        rows[side].append((w.force_n, w.point_m, w.torque_nm or (0.0, 0.0, 0.0)))
    out: dict[str, ContactSet] = {}
    for side, items in rows.items():
        if not items:
            out[side] = ContactSet.empty()
            continue
        f, p, t = (np.array(col, dtype=float) for col in zip(*items, strict=True))
        out[side] = ContactSet(f, p, t)
    return out


def ground_reaction_breakdown(
    wrenches: Iterable[OverlayWrench],
    com_m: ArrayLike,
    *,
    ground_height_m: float = 0.0,
    cop_min_fz_n: float = COP_MIN_FZ_N,
    foot_of: FootOf = foot_of_body,
) -> GroundReactionBreakdown:
    """Full breakdown of the foot ground contacts among ``wrenches``.

    Preconditions: ``com_m`` is a finite world 3-vector; ``wrenches`` are
    ground contacts (see the module docstring).
    """
    return analyze_ground_reaction(
        foot_contact_sets(wrenches, foot_of=foot_of),
        com_m,
        ground_height_m=ground_height_m,
        cop_min_fz_n=cop_min_fz_n,
    )


def ground_reaction_overlay(
    wrenches: Iterable[OverlayWrench],
    com_m: ArrayLike,
    *,
    source: str,
    ground_height_m: float = 0.0,
    cop_min_fz_n: float = COP_MIN_FZ_N,
    foot_of: FootOf = foot_of_body,
) -> tuple[OverlayWrench, ...]:
    """``contact:grf_*`` / ``free_moment_*`` / ``moment_com_*`` overlay wrenches.

    Empty when no foot carries load (unavailable, never zero arrows).
    """
    breakdown = ground_reaction_breakdown(
        wrenches,
        com_m,
        ground_height_m=ground_height_m,
        cop_min_fz_n=cop_min_fz_n,
        foot_of=foot_of,
    )
    return tuple(to_overlay_wrenches(breakdown, source=source))
