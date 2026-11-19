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
from typing import Protocol


class SidewaysSettings(Protocol):
    """The settings the sideways range is read from."""

    @property
    def side_close_knuckle_max_deg(self) -> float: ...
    @property
    def side_close_flexed_share(self) -> float: ...
    @property
    def side_close_flexed_at_deg(self) -> float: ...
    @property
    def side_close_spread_in_palm(self) -> bool: ...


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


def split_swing_rad(
    swing_rad: float, flexion_rad: float, settings: SidewaysSettings
) -> tuple[float, float]:
    """``swing_rad`` as ``(lean, turn)`` for a knuckle flexed ``flexion_rad``.

    ``turn`` swings the finger about the axis the knuckle spreads on, which is
    fixed in the palm: a straight finger moves sideways by all of it, a finger
    flexed a right angle only turns the plane it bends in, which is how fingers
    converge as they curl. ``lean`` tips the flexed finger sideways off that
    path; it is only ever the correction that keeps the sideways movement,
    ``turn * cos(flexion) + lean``, inside :func:`sideways_limit_rad`.
    Postconditions: ``turn`` is ``swing_rad`` held to the straight-finger
    range; ``lean`` is zero for a straight finger, so it swings as asked.
    With ``side_close.spread_in_palm`` off the whole swing is a lean, as the
    fit was first tuned (#15278 records why that is still the default).
    """
    if not settings.side_close_spread_in_palm:
        return swing_rad, 0.0
    full = math.radians(settings.side_close_knuckle_max_deg)
    turn = min(full, max(-full, swing_rad))
    most = sideways_limit_rad(flexion_rad, settings)
    sideways = turn * math.cos(flexion_rad)
    return min(most, max(-most, sideways)) - sideways, turn


__all__ = ["SidewaysSettings", "sideways_limit_rad", "split_swing_rad"]
