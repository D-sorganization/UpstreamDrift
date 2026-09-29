"""Body-skeleton segment connectivity for full-body marker targets.

Defines the canonical pair-of-marker segments that turn a marker cloud
into a stick figure. The connectivity is data-driven — it targets the
"anatomical 28" marker subset used by the full-body C3D loader (a
Plug-in-Gait subset: pelvis / back / head / shoulder / elbow / wrist /
knee / ankle / toe).

The default segment table is filtered at runtime to those segments whose
endpoints are actually present in the supplied marker name list, so the
helpers are always safe to call with a partial marker set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, get_args

BodySegmentGroup = Literal[
    "torso",
    "head",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "pelvis",
]

_VALID_GROUPS: frozenset[str] = frozenset(get_args(BodySegmentGroup))


@dataclass(frozen=True)
class BodySegment:
    """A directed pair of marker names connected by a single bone-like line.

    ``a`` and ``b`` are marker names (non-empty, distinct strings). ``group``
    is one of the seven anatomical body-region literals.
    """

    a: str
    b: str
    group: BodySegmentGroup

    def __post_init__(self) -> None:
        if not isinstance(self.a, str) or not self.a:
            raise ValueError(
                f"BodySegment.a must be a non-empty string, got {self.a!r}"
            )
        if not isinstance(self.b, str) or not self.b:
            raise ValueError(
                f"BodySegment.b must be a non-empty string, got {self.b!r}"
            )
        if self.a == self.b:
            raise ValueError(
                f"BodySegment.a and BodySegment.b must differ, got {self.a!r}"
            )
        if self.group not in _VALID_GROUPS:
            raise ValueError(
                f"BodySegment.group must be one of {sorted(_VALID_GROUPS)}, "
                f"got {self.group!r}"
            )


# Canonical 26-segment table for the anatomical 28-marker subset.
# Tuples are (a, b, group). Ordering is anatomically grouped for readability;
# downstream consumers must not depend on iteration order.
_CANONICAL_SEGMENTS: tuple[tuple[str, str, BodySegmentGroup], ...] = (
    # pelvis (4)
    ("WaistLeft", "WaistRight", "pelvis"),
    ("WaistLeft", "WaistLBack", "pelvis"),
    ("WaistRight", "WaistRBack", "pelvis"),
    ("WaistLBack", "WaistRBack", "pelvis"),
    # torso (4) — BackTop acts as upper-spine proxy
    ("BackTop", "BackLeft", "torso"),
    ("BackTop", "BackRight", "torso"),
    ("BackTop", "WaistLBack", "torso"),
    ("BackTop", "WaistRBack", "torso"),
    # head (2)
    ("HeadTop", "HeadFront", "head"),
    ("HeadTop", "HeadSide", "head"),
    # left arm (4)
    ("LShoulderTop", "LShoulderBack", "left_arm"),
    ("LShoulderTop", "LUArmHigh", "left_arm"),
    ("LUArmHigh", "LElbowOut", "left_arm"),
    ("LElbowOut", "LWristTop", "left_arm"),
    # right arm (4)
    ("RShoulderTop", "RShoulderBack", "right_arm"),
    ("RShoulderTop", "RUArmHigh", "right_arm"),
    ("RUArmHigh", "RElbowOut", "right_arm"),
    ("RElbowOut", "RWristTop", "right_arm"),
    # left leg (4)
    ("WaistLeft", "LKneeOut", "left_leg"),
    ("LKneeOut", "LAnkleOut", "left_leg"),
    ("LAnkleOut", "LToeIn", "left_leg"),
    ("LToeIn", "LToeOut", "left_leg"),
    # right leg (4)
    ("WaistRight", "RKneeOut", "right_leg"),
    ("RKneeOut", "RAnkleOut", "right_leg"),
    ("RAnkleOut", "RToeIn", "right_leg"),
    ("RToeIn", "RToeOut", "right_leg"),
)


def default_body_segments(
    marker_names: Sequence[str],
) -> tuple[BodySegment, ...]:
    """Return the canonical segment list filtered to present markers.

    Only segments whose BOTH endpoints appear in ``marker_names`` are
    returned. Always safe — a missing marker just drops the segments that
    reference it, leaving the remainder of the figure intact.
    """
    if marker_names is None:
        raise TypeError("marker_names must be a sequence of strings, got None")
    available = frozenset(marker_names)
    return tuple(
        BodySegment(a=a, b=b, group=group)
        for (a, b, group) in _CANONICAL_SEGMENTS
        if a in available and b in available
    )


__all__ = [
    "BodySegment",
    "BodySegmentGroup",
    "default_body_segments",
]
