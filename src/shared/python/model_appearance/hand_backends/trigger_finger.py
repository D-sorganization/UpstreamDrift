"""The trigger finger: a forefinger held apart, down the shaft (#15278).

A golfer's trail forefinger does not close up against the middle finger. It
hangs a little down the shaft like a finger on a trigger, which carries its
first joint down to about where the thumb's tip lies. The spread is a setting
per hand, ``side_close.trigger_spread_deg``; zero leaves the finger closed. It
is the most the finger is held apart: a finger whose tip will not seat on the
grip at that spread is brought back, ``side_close.trigger_back_off_deg`` at a
time (:func:`wrap_trigger_finger`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
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

    def points(self, angles: Mapping[str, float]) -> NDArray[np.float64]: ...

    def segment_ids(self, bone: str) -> NDArray[np.int64]: ...


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
    curl = hand.settings.curl
    probe_rad = curl.step_rad  # the curl's own step, a setting
    closed = hand.side.get(TRIGGER_BONE, 0.0)

    def apart(swing: float) -> float:
        hand.swing_sideways(TRIGGER_BONE, swing)
        mine = hand.joint_position(TRIGGER_TIP_BONE, angles)
        theirs = hand.joint_position(NEIGHBOUR_TIP_BONE, angles)
        return float(np.linalg.norm(mine - theirs))

    away = 1.0 if apart(closed + probe_rad) > apart(closed - probe_rad) else -1.0
    hand.swing_sideways(TRIGGER_BONE, closed + away * spread_rad)


def wrap_trigger_finger(
    hand: _SwingableHand,
    hand_side: str,
    closed_angles: Mapping[str, float],
    obstacle: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    wrap: Callable[[], dict[str, float]],
) -> dict[str, float]:
    """Wrap the fingers with the forefinger of ``hand_side`` held apart.

    Preconditions: ``closed_angles`` is the pose the fingers were closed
    sideways on, empty when side closing is off (there is then no trigger
    finger); ``wrap()`` curls the fingers at the hand's current sideways swings
    and returns the joint angles; ``obstacle`` gives each skin point's distance
    to what the fingers close on. Postconditions: the angles returned are those
    of ``wrap()`` at the spread that was kept, which is the configured spread
    less a whole number of back-off steps and never less than zero; either the
    forefinger's tip is within the curl's contact tolerance of the obstacle or
    the finger is back where the sideways closing left it.
    """
    if not closed_angles:
        return wrap()
    closed = hand.side.get(TRIGGER_BONE, 0.0)
    spread_trigger_finger(hand, closed_angles, hand_side)
    signed = hand.side.get(TRIGGER_BONE, 0.0) - closed
    angles = wrap()
    settings = hand.settings
    step = math.radians(settings.side_close_trigger_back_off_deg)
    tolerance = settings.curl.contact_tolerance_m
    tip = hand.segment_ids(TRIGGER_TIP_BONE)
    left = abs(signed)
    while left > 0.0 and float(obstacle(hand.points(angles))[tip].min()) > tolerance:
        left = max(left - step, 0.0)
        hand.swing_sideways(TRIGGER_BONE, closed + math.copysign(left, signed))
        angles = wrap()
    return angles


__all__ = ["spread_trigger_finger", "wrap_trigger_finger"]
