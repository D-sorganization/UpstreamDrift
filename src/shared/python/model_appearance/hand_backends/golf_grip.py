"""The golf grip the visual hands hold (epic #15150, #15185).

A rubber golf grip is not a cylinder: it narrows from the butt cap toward the
club head. The published size of a grip is its outside diameter at two
stations below the edge of the cap; this module reads those from
``scripts/config/golf_grip.json`` and treats the grip as an even taper through
them. The cap and the length are typical values, declared unverified there.
This is a visual model only; contact physics keeps its own grip radius.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GRIP_PATH = Path(__file__).resolve().parents[5] / "scripts/config/golf_grip.json"


def _positive(value: Any, name: str) -> float:
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be positive and finite")
    return number


@dataclass(frozen=True)
class GolfGrip:
    """Outside shape of a grip; stations are metres below the cap's edge."""

    edge_radius_m: float
    taper: float
    length_m: float
    cap_radius_m: float
    cap_height_m: float
    cap_round_m: float
    vent_radius_m: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> GolfGrip:
        """Build from the settings mapping.

        Preconditions: the chosen size has exactly two diameter stations, the
        second farther from the cap and narrower; every dimension is positive.
        """
        stations = raw["sizes"][raw["size"]]["diameter_stations_m"]
        if len(stations) != 2:
            raise ValueError("a grip size needs exactly two diameter stations")
        (near, wide), (far, narrow) = (
            (float(s), 0.5 * _positive(d, "diameter")) for s, d in stations
        )
        if not 0.0 <= near < far:
            raise ValueError("diameter stations must run from the cap to the head")
        if narrow > wide:
            raise ValueError("a grip must not widen toward the club head")
        taper = (narrow - wide) / (far - near)
        length = _positive(raw["length_m"], "length_m")
        grip = cls(
            edge_radius_m=wide - taper * near,
            taper=taper,
            length_m=length,
            cap_radius_m=0.5 * _positive(raw["cap_diameter_m"], "cap_diameter_m"),
            cap_height_m=_positive(raw["cap_height_m"], "cap_height_m"),
            cap_round_m=_positive(raw["cap_round_m"], "cap_round_m"),
            vent_radius_m=0.5 * _positive(raw["vent_diameter_m"], "vent_diameter_m"),
        )
        if grip.edge_radius_m + taper * length <= 0.0:
            raise ValueError("the grip tapers to nothing before its mouth")
        if grip.cap_radius_m < grip.edge_radius_m:
            raise ValueError("the cap must be at least as wide as the grip")
        return grip

    def radius_at_m(self, below_cap_edge_m: float) -> float:
        """Outside radius ``below_cap_edge_m`` from the cap toward the head.

        Raises ``ValueError`` for a station that is not on the grip.
        """
        if not 0.0 <= below_cap_edge_m <= self.length_m:
            raise ValueError("station must lie on the grip")
        return self.edge_radius_m + self.taper * below_cap_edge_m

    def drawing_knots(
        self, hand_below_cap_edge_m: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """``(z, radius)`` at the cap edge and at the mouth, in the frame of a
        hand whose origin is ``hand_below_cap_edge_m`` below the cap's edge."""
        self.radius_at_m(hand_below_cap_edge_m)
        return (
            (-hand_below_cap_edge_m, self.edge_radius_m),
            (self.length_m - hand_below_cap_edge_m, self.radius_at_m(self.length_m)),
        )


def load_golf_grip() -> GolfGrip:
    """The grip in ``scripts/config/golf_grip.json``."""
    if not GRIP_PATH.is_file():
        raise ValueError(f"missing golf grip settings: {GRIP_PATH}")
    return GolfGrip.from_mapping(json.loads(GRIP_PATH.read_text(encoding="utf-8")))
