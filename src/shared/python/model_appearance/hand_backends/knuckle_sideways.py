"""How far a finger may swing sideways at its knuckle (#15278, epic #15150).

A finger moves almost wholly in flexion and extension. Its knuckle also lets
it spread sideways, but only while it is fairly straight: as the knuckle
flexes its collateral ligaments tighten and the sideways play all but goes.
A fit that swings a bent knuckle sideways puts a kink in the finger that no
hand can make. The range here is widest with the knuckle straight and falls
linearly to a share of that by a set flexion; every number is a setting
(``side_close`` in ``scripts/config/skinned_human_hand.json``).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Protocol

#: The fingers a hand may be set to abduct (``side_close.knuckle_abduction_deg``).
ABDUCTED_FINGERS = ("middle", "ring", "little")


class SidewaysSettings(Protocol):
    """The settings the sideways range is read from."""

    @property
    def side_close_knuckle_max_deg(self) -> float: ...
    @property
    def side_close_flexed_share(self) -> float: ...
    @property
    def side_close_flexed_at_deg(self) -> float: ...
    @property
    def side_close_limit_by_flexion(self) -> bool: ...


class _Knuckles(Protocol):
    """A posable hand whose fingers swing sideways at their knuckles."""

    side: dict[str, float]

    @property
    def settings(self) -> Any: ...

    def swing_sideways(self, bone: str | None, swing: float = 0.0) -> None: ...

    def joint_position(self, bone: str, angles: Mapping[str, float]) -> Any: ...


def sideways_limit_rad(flexion_rad: float, settings: SidewaysSettings) -> float:
    """The most a knuckle flexed ``flexion_rad`` may swing to either side.

    Precondition: ``flexion_rad`` is finite. Postconditions: the full range at
    or below zero flexion, ``flexed_share`` of it at or beyond
    ``flexed_at_deg``, linear between; never negative.
    """
    flexion_rad = float(flexion_rad)
    if not math.isfinite(flexion_rad):
        raise ValueError("knuckle flexion must be finite")
    bent = flexion_rad / math.radians(settings.side_close_flexed_at_deg)
    lost = (1.0 - settings.side_close_flexed_share) * min(1.0, max(0.0, bent))
    return math.radians(settings.side_close_knuckle_max_deg) * (1.0 - lost)


def held_swing_rad(
    swing_rad: float, flexion_rad: float, settings: SidewaysSettings
) -> float:
    """``swing_rad`` as a knuckle flexed ``flexion_rad`` can take it.

    Postconditions: with ``side_close.limit_by_flexion`` on, the swing held
    inside :func:`sideways_limit_rad`, on the side asked; with it off, the
    swing as asked, as the fit was first tuned (#15278 records why that is
    still the default).
    """
    if not settings.side_close_limit_by_flexion:
        return swing_rad
    most = sideways_limit_rad(flexion_rad, settings)
    return min(most, max(-most, swing_rad))


def abduct_knuckles(
    hand: _Knuckles, angles: Mapping[str, float], abduction_deg: Mapping[str, float]
) -> None:
    """Swing each finger of ``abduction_deg`` toward the little finger by its
    degrees, on top of the swing the sideways closing left it (#15329).

    Preconditions: every finger is one of :data:`ABDUCTED_FINGERS` and its
    angle is finite and not negative (``ValueError``). Postconditions: an empty
    mapping leaves the hand untouched; a finger's swing is its swing on entry
    plus its angle, toward the side that carries its end joint along the line
    from the forefinger's knuckle to the little finger's.
    """
    for finger, degrees in abduction_deg.items():
        if finger not in ABDUCTED_FINGERS:
            raise ValueError(f"knuckle abduction is of {ABDUCTED_FINGERS}: {finger!r}")
        if not (math.isfinite(degrees) and degrees >= 0.0):
            raise ValueError("knuckle abduction must be finite and not negative")
    if not abduction_deg:
        return
    probe = hand.settings.curl.step_rad
    across = hand.joint_position("little_prox", angles)
    across = across - hand.joint_position("index_prox", angles)
    for finger, degrees in abduction_deg.items():
        bone, closed = f"{finger}_prox", hand.side.get(f"{finger}_prox", 0.0)

        def along(swing: float, bone: str = bone, finger: str = finger) -> float:
            hand.swing_sideways(bone, swing)
            return float(hand.joint_position(f"{finger}_dist", angles) @ across)

        toward = 1.0 if along(closed + probe) > along(closed - probe) else -1.0
        hand.swing_sideways(bone, closed + toward * math.radians(degrees))


def hold_knuckles_in_range(
    hand: _Knuckles, angles: Mapping[str, float], fingers: Iterable[str]
) -> bool:
    """Bring the swing of each of ``fingers`` inside the range its knuckle has
    as ``angles`` flex it, less ``side_close.abduction_hold_margin_deg``.

    Postconditions: no named finger swings further than
    :func:`sideways_limit_rad` of its knuckle less the margin (never below
    zero); a swing already inside is untouched; ``True`` if any was moved,
    after which the fingers want wrapping again.
    """
    settings = hand.settings
    margin = math.radians(settings.side_close_abduction_hold_margin_deg)
    moved = False
    for finger in fingers:
        bone = f"{finger}_prox"
        swing = hand.side.get(bone, 0.0)
        most = max(0.0, sideways_limit_rad(angles[bone], settings) - margin)
        if abs(swing) > most:
            hand.swing_sideways(bone, math.copysign(most, swing))
            moved = True
    return moved


__all__ = [
    "ABDUCTED_FINGERS",
    "SidewaysSettings",
    "abduct_knuckles",
    "held_swing_rad",
    "hold_knuckles_in_range",
    "sideways_limit_rad",
]
