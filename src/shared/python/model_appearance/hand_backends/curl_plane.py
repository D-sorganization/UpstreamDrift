"""Fingers whose curl plane is turned from the palm's (epic #15150).

A club lies diagonally across the trail palm, from the base of the little
finger toward the forefinger's middle joint, and the fingers still close on
it nearly square: each is turned at its knuckle, the hand as a whole is not.
Here that is one turn per finger about the palm's normal, taken at the
knuckle before the finger flexes, so the finger's flexion axis turns with it
and the palm, the wrist and the forearm stay where they were.

The knuckle is the only joint of a finger with sideways play, about 20 to 30
degrees in all with the finger straight (recalled from anatomy texts, NOT
verified against a source); ``preshape.curl_plane_turn_max_deg`` in
``scripts/config/skinned_human_hand.json`` is the most any one turn may be.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.hand_backends.joint_angles import FINGERS
from src.shared.python.model_appearance.hand_backends.skinning import (
    rotation_about_axis,
)

_SIDES = ("lead", "trail")
_NAME = "preshape.curl_plane_turn_deg"
#: The palm's normal in the hand's rest frame (x radial, z palmar).
_PALMAR = np.array([0.0, 0.0, 1.0])

#: Per hand, each finger's turn in degrees.
CurlPlaneTurns = tuple[tuple[str, tuple[tuple[str, float], ...]], ...]


def curl_plane_turns(raw: Mapping[str, Any], most_deg: float) -> CurlPlaneTurns:
    """Validate ``preshape.curl_plane_turn_deg``.

    Preconditions: both hands, each with a finite turn for every finger and
    no other key, none past ``most_deg`` either way (``ValueError``
    otherwise). Postcondition: the turns, per hand, in finger order.
    """
    if not most_deg >= 0.0 or set(raw) != set(_SIDES):
        raise ValueError(f"{_NAME} needs a lead and a trail hand and a range >= 0")
    out = []
    for side in _SIDES:
        per = raw[side]
        if set(per) != set(FINGERS):
            raise ValueError(f"{_NAME}.{side} needs exactly {FINGERS}")
        turns = tuple((finger, float(per[finger])) for finger in FINGERS)
        if not all(math.isfinite(t) and abs(t) <= most_deg for _, t in turns):
            raise ValueError(f"{_NAME}.{side} must be within {most_deg} degrees")
        out.append((side, turns))
    return tuple(out)


def turn_curl_planes(
    fixed: MutableMapping[str, NDArray[np.float64]], turns_deg: Mapping[str, float]
) -> None:
    """Turn each named finger's knuckle rotation in ``fixed`` about the palm's
    normal, ahead of its flexion.

    Precondition: every name is a finger (``ValueError`` otherwise).
    Postcondition: a finger with no turn keeps the very rotation it had.
    """
    unknown = sorted(set(turns_deg) - set(FINGERS))
    if unknown:
        raise ValueError(f"not fingers: {unknown}")
    for finger, turn in turns_deg.items():
        if turn:
            bone = f"{finger}_prox"
            fixed[bone] = fixed[bone] @ rotation_about_axis(_PALMAR, math.radians(turn))


__all__ = ["CurlPlaneTurns", "curl_plane_turns", "turn_curl_planes"]
