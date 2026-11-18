"""The trigger finger: a forefinger held apart, down the shaft (#15278).

A golfer's trail forefinger does not close up against the middle finger. It
hangs a little down the shaft like a finger on a trigger, which carries its
first joint down to about where the thumb's tip lies. The spread is a setting
per hand, ``side_close.trigger_spread_deg``; zero leaves the finger closed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

TRIGGER_BONE = "index_prox"
NEIGHBOUR_TIP_BONE = "middle_dist"
TRIGGER_TIP_BONE = "index_dist"


class _SwingableHand(Protocol):
    side: dict[str, float]

    @property
    def settings(self) -> Any: ...

    def swing_sideways(self, bone: str | None, swing: float = 0.0) -> None: ...

    def joint_position(
        self, bone: str, angles: Mapping[str, float]
    ) -> NDArray[np.float64]: ...


def spread_trigger_finger(
    hand: _SwingableHand, angles: Mapping[str, float], hand_side: str
) -> None:
    """Swing the forefinger of ``hand_side`` away from the middle finger.

    Preconditions: ``hand_side`` is ``"lead"`` or ``"trail"``; the hand's
    settings give that side a finite spread of zero or more degrees.
    Postconditions: with zero spread the hand is untouched; otherwise the
    forefinger's sideways swing is its swing on entry plus the spread, in the
    direction that takes its end joint away from the middle finger's.
    """
    spreads = dict(hand.settings.side_close_trigger_spread_deg)
    if hand_side not in spreads:
        raise ValueError(f"hand_side must be one of {sorted(spreads)}")
    spread_rad = math.radians(spreads[hand_side])
    if not (math.isfinite(spread_rad) and spread_rad >= 0.0):
        raise ValueError("trigger spread must be finite and not negative")
    if spread_rad == 0.0:
        return
    probe_rad = hand.settings.curl.step_rad  # the curl's own step, a setting
    closed = hand.side.get(TRIGGER_BONE, 0.0)

    def apart(swing: float) -> float:
        hand.swing_sideways(TRIGGER_BONE, swing)
        mine = hand.joint_position(TRIGGER_TIP_BONE, angles)
        theirs = hand.joint_position(NEIGHBOUR_TIP_BONE, angles)
        return float(np.linalg.norm(mine - theirs))

    away = 1.0 if apart(closed + probe_rad) > apart(closed - probe_rad) else -1.0
    hand.swing_sideways(TRIGGER_BONE, closed + away * spread_rad)


__all__ = ["spread_trigger_finger"]
