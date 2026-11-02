"""Putt classes (distance, slope, break orientation, stimp) with encounter weights."""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.contracts import require

DISTANCE_RANGE_FT = (2.0, 60.0)
SLOPE_RANGE_PCT = (0.0, 4.0)
FLAT = "flat"
ORIENTATIONS = (FLAT, "uphill", "downhill", "left_to_right", "right_to_left", "oblique")
DEFAULT_DISTANCES_FT = (3.0, 6.0, 10.0, 15.0, 25.0, 40.0, 60.0)
DEFAULT_SLOPES_PCT = (0.0, 2.0, 4.0)
DEFAULT_STIMPS = (10.0, 12.0)

#: Fall line of each break orientation: the direction the green falls toward,
#: in degrees CCW of the ball-to-hole line (the ``aspect`` of
#: ``HeightField.planar``). A right-to-left putt falls toward the golfer's left
#: (+90 deg). "oblique" is taken as a quartering downhill putt that breaks right
#: to left (45 deg); the issue leaves its exact angle open.
ORIENTATION_ASPECT_DEG: dict[str, float] = {
    FLAT: 0.0,
    "uphill": 180.0,
    "downhill": 0.0,
    "left_to_right": 270.0,
    "right_to_left": 90.0,
    "oblique": 45.0,
}


@dataclass(frozen=True)
class PuttClass:
    """One putt class; a flat green (zero slope) has no break orientation."""

    distance_ft: float
    slope_pct: float
    orientation: str
    stimp: float

    def __post_init__(self) -> None:
        lo, hi = DISTANCE_RANGE_FT
        require(lo <= self.distance_ft <= hi, f"distance_ft must be in [{lo}, {hi}]")
        lo, hi = SLOPE_RANGE_PCT
        require(lo <= self.slope_pct <= hi, f"slope_pct must be in [{lo}, {hi}]")
        require(self.orientation in ORIENTATIONS, f"orientation in {ORIENTATIONS}")
        require(self.stimp > 0.0, "stimp must be positive")
        require(
            (self.slope_pct == 0.0) == (self.orientation == FLAT),
            "orientation is 'flat' exactly when slope_pct is 0",
        )


@dataclass(frozen=True)
class ClassLibrary:
    """Putt classes with encounter weights that sum to 1."""

    classes: tuple[PuttClass, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        require(len(set(self.classes)) == len(self.classes), "duplicate class")
        require(len(self.weights) == len(self.classes), "weights/classes mismatch")
        require(all(w > 0.0 for w in self.weights), "weights must be > 0")
        require(abs(sum(self.weights) - 1.0) <= 1e-9, "weights must sum to 1")

    @classmethod
    def from_counts(
        cls, classes: Sequence[PuttClass], counts: Sequence[float]
    ) -> ClassLibrary:
        """Normalise positive encounter ``counts`` into weights."""
        require(all(c > 0.0 for c in counts), "counts must be positive")
        total = float(sum(counts))
        return cls(tuple(classes), tuple(float(c) / total for c in counts))


def default_library(
    distances_ft: Iterable[float] = DEFAULT_DISTANCES_FT,
    slopes_pct: Iterable[float] = DEFAULT_SLOPES_PCT,
    orientations: Iterable[str] = ORIENTATIONS[1:],
    stimps: Iterable[float] = DEFAULT_STIMPS,
) -> ClassLibrary:
    """Uniformly weighted cross product; zero slope gives only the ``flat`` class."""
    orients, slopes = tuple(orientations), tuple(slopes_pct)
    classes = [
        PuttClass(float(d), float(s), o, float(v))
        for d, s, v in itertools.product(distances_ft, slopes, stimps)
        for o in ((FLAT,) if s == 0.0 else orients)
    ]
    return ClassLibrary.from_counts(classes, [1.0] * len(classes))


def load_frequency_table(path: str | Path) -> ClassLibrary:
    """Read a CSV frequency table into a library; its rows define the classes.

    Columns: ``distance_ft, slope_pct, orientation, stimp, count``. Repeated
    classes add their counts; zero-count rows are dropped.
    """
    counts: defaultdict[PuttClass, float] = defaultdict(float)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            d, s, v = (float(row[k]) for k in ("distance_ft", "slope_pct", "stimp"))
            counts[PuttClass(d, s, row["orientation"], v)] += float(row["count"])
    seen = {k: c for k, c in counts.items() if c > 0.0}
    return ClassLibrary.from_counts(list(seen), list(seen.values()))
