"""Golf words for wrist angles, aligned with the golf-term registry.

Cupped/bowed and hinge/cock describe positions of the wrist. "Release" is an
event and a description of the motion, not a sign of ulnar deviation, so there
is deliberately no alias for it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.anatomical_angles.wrist import WristAngles


@dataclass(frozen=True)
class GolfAlias:
    """A golf word: positive alias value = the word's sense of ``component``."""

    word: str
    component: str  # "flexion" or "ulnar_deviation"
    sign: float
    description: str


_CUPPED = GolfAlias("cupped", "flexion", -1.0, "wrist extension (dorsiflexion)")
_BOWED = GolfAlias("bowed", "flexion", 1.0, "wrist flexion (palmar flexion)")
_HINGE = GolfAlias("hinge", "ulnar_deviation", -1.0, "radial deviation (cock)")
WRIST_GOLF_ALIASES: dict[str, GolfAlias] = {
    "cupped": _CUPPED,
    "bowed": _BOWED,
    "hinge": _HINGE,
    "cock": _HINGE,
}


def alias_series(angles: WristAngles, word: str) -> np.ndarray:
    """Degrees of the golf ``word`` (positive = that wrist state) for one hand.

    The hand is whichever the series was computed for; choose lead or trail by
    the frames passed to ``wrist_angles`` (``core.handedness.lead_side``).
    ``ValueError`` for an unknown word; "release" is rejected explicitly.
    """
    if word == "release":
        raise ValueError(
            "'release' is an event and motion description, not a signed wrist "
            "angle; it has no alias"
        )
    try:
        alias = WRIST_GOLF_ALIASES[word]
    except KeyError:
        raise ValueError(
            f"unknown wrist alias {word!r}; choose from {sorted(WRIST_GOLF_ALIASES)}"
        ) from None
    values = (
        angles.flexion_deg
        if alias.component == "flexion"
        else angles.ulnar_deviation_deg
    )
    return np.asarray(alias.sign * values)
