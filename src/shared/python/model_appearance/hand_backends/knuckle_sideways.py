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
    def side_close_limit_by_flexion(self) -> bool: ...


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


__all__ = ["SidewaysSettings", "held_swing_rad", "sideways_limit_rad"]
