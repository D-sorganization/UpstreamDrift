"""Lead/trail hand to anatomical side, for a right- or left-handed golfer.

The one place that decides which side is the lead hand: the left hand of a
right-handed golfer, the right hand of a left-handed one (HRG-G1, #13186).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

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


def subject_handedness(spec: Mapping[str, Any]) -> Handedness:
    """Handedness declared by a spec's ``subject`` block, the one accessor.

    A spec without ``subject.handedness`` (every spec built before HRG-GR0,
    #13201) resolves to :data:`DEFAULT_HANDEDNESS`; this is a declared default,
    not an inference. ``ValueError`` for a declared value that is not
    ``"right"`` or ``"left"``.
    """
    subject = spec.get("subject") or {}
    return validate_handedness(subject.get("handedness", DEFAULT_HANDEDNESS))
