"""Tape-measured segment lengths: as many as the user has, both sides at once.

The fit needs one measured length for scale (the gauge) and profits from
every additional one: each measured segment replaces a 5 cm anthropometric
prior with a 3 mm measurement. Users measure body parts, not joint names, so
this module maps everyday names (``forearm``, ``shank``, ``shoulder_width``)
onto the skeleton's segments and applies bilateral symmetry: one ``forearm``
measurement constrains both forearms. Side-specific names are accepted when
the two sides really differ.

Segments are named by their child joint in :data:`~.skeleton.PARENTS`
(``left_wrist`` is elbow -> wrist). Widths are split in half onto the two
half-segments from the midline joint.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from src.shared.python.core.contracts import require

from .skeleton import PARENTS

#: Everyday name -> (segment child joints, factor applied to the measured length).
SEGMENT_ALIASES: dict[str, tuple[tuple[str, ...], float]] = {
    "torso": (("neck",), 1.0),
    "head": (("nose",), 1.0),
    "upper_arm": (("left_elbow", "right_elbow"), 1.0),
    "forearm": (("left_wrist", "right_wrist"), 1.0),
    "thigh": (("left_knee", "right_knee"), 1.0),
    "shank": (("left_ankle", "right_ankle"), 1.0),
    "shoulder_width": (("left_shoulder", "right_shoulder"), 0.5),
    "hip_width": (("left_hip", "right_hip"), 0.5),
    "left_upper_arm": (("left_elbow",), 1.0),
    "right_upper_arm": (("right_elbow",), 1.0),
    "left_forearm": (("left_wrist",), 1.0),
    "right_forearm": (("right_wrist",), 1.0),
    "left_thigh": (("left_knee",), 1.0),
    "right_thigh": (("right_knee",), 1.0),
    "left_shank": (("left_ankle",), 1.0),
    "right_shank": (("right_ankle",), 1.0),
}

#: How to put the tape on, per everyday name (rendered in the tile and guide).
TAPE_GUIDE: dict[str, str] = {
    "shank": "lateral knee joint line to the lateral ankle bone (malleolus), leg straight",
    "forearm": "lateral elbow crease to the wrist bone (styloid), arm straight",
    "upper_arm": "tip of the shoulder (acromion) to the lateral elbow joint line",
    "thigh": "greater trochanter (hip bone at the side) to the lateral knee joint line",
    "shoulder_width": "acromion to acromion across the back",
    "hip_width": "greater trochanter to greater trochanter",
    "torso": "mid-hip (between the trochanters) to the base of the neck (C7)",
    "head": "base of the neck (C7) to the tip of the nose",
}

#: Best value first: bony landmarks, long, rigid, reliably detected.
RECOMMENDED: tuple[str, ...] = ("shank", "forearm", "upper_arm", "thigh")


@dataclass(frozen=True)
class Measurement:
    """One tape reading: an everyday or segment name and its length in metres."""

    name: str
    metres: float

    def __post_init__(self) -> None:
        require(self.name.strip() != "", "measurement needs a name")
        require(0.02 <= self.metres <= 2.5, "length must be in metres", self.metres)
        require(
            self.name in SEGMENT_ALIASES or self.name in PARENTS,
            "unknown segment; use an alias or a joint name",
            self.name,
        )

    def segments(self) -> dict[str, float]:
        """``{segment child joint: metres}`` this reading constrains."""
        if self.name in SEGMENT_ALIASES:
            joints, factor = SEGMENT_ALIASES[self.name]
            return dict.fromkeys(joints, self.metres * factor)
        return {self.name: self.metres}


def parse_measurement(text: str) -> Measurement:
    """``NAME=METRES`` (``shank=0.42``); precondition: both parts present."""
    name, sep, value = text.partition("=")
    require(sep == "=" and name.strip() != "", "measurement must be NAME=METRES", text)
    try:
        metres = float(value)
    except ValueError as exc:
        raise ValueError(f"length is not a number: {value!r}") from exc
    return Measurement(name.strip().lower(), metres)


def expand_measurements(
    items: Iterable[Measurement | str],
) -> dict[str, float]:
    """Every constrained segment with its length; repeats of a segment average.

    Postcondition: keys are child joints of :data:`PARENTS`; insertion order
    follows the first mention, so the first item names the scale gauge.
    """
    sums: dict[str, list[float]] = {}
    for item in items:
        m = parse_measurement(item) if isinstance(item, str) else item
        for joint, metres in m.segments().items():
            sums.setdefault(joint, []).append(metres)
    return {j: sum(v) / len(v) for j, v in sums.items()}


def gauge(lengths: Mapping[str, float]) -> tuple[str, float]:
    """The scale anchor: the first measured segment. Precondition: non-empty."""
    require(bool(lengths), "at least one measured segment is needed for scale")
    first = next(iter(lengths))
    return first, lengths[first]


def describe(lengths: Mapping[str, float]) -> str:
    """``left_wrist=0.260, right_wrist=0.260`` for logs and the tile."""
    return ", ".join(f"{k}={v:.3f}" for k, v in lengths.items())
