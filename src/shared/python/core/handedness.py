"""Lead/trail hand to anatomical side, for a right- or left-handed golfer.

The one place that decides which side is the lead hand: the left hand of a
right-handed golfer, the right hand of a left-handed one (HRG-G1, #13186).
"""

from __future__ import annotations

from typing import Literal

Handedness = Literal["right", "left"]
Side = Literal["L", "R"]
DEFAULT_HANDEDNESS: Handedness = "right"
HANDEDNESS_CHOICES: tuple[Handedness, ...] = ("right", "left")


def validate_handedness(value: object) -> Handedness:
    """``value`` as a :data:`Handedness`; ``ValueError`` for anything else."""
    if value == "right":
        return "right"
    if value == "left":
        return "left"
    raise ValueError(f"handedness must be 'right' or 'left', got {value!r}")


def lead_side(handedness: object) -> Side:
    """Side of the lead hand: ``"L"`` for a right-handed golfer."""
    return "L" if validate_handedness(handedness) == "right" else "R"


def trail_side(handedness: object) -> Side:
    """Side of the trail hand: ``"R"`` for a right-handed golfer."""
    return "R" if lead_side(handedness) == "L" else "L"
