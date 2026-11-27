"""Display-neutral readout numbers derived from a coherent ``BunkerFlight``.

Pure geometry over the already-validated SI traces (x forward, y left, z up).
Desktop, API and web displays share this instead of each re-deriving apex,
descent and rollout. No physics lives here, and a missing flight (a refused or
unflown shot, ADR-0032) yields no metrics rather than invented numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from src.tools.bunker_shot_gui.flight import BunkerFlight


@dataclass(frozen=True)
class FlightMetrics:
    """SI readout of one flight; ``lateral_m`` is left-positive at the rest point."""

    carry_m: float
    apex_height_m: float
    apex_downrange_m: float
    descent_angle_deg: float | None  # None when the last flight segment is degenerate
    rollout_m: float  # ground-plane path length, ignoring bounce height
    total_m: float  # straight-line distance from the origin to the rest point
    lateral_m: float


def flight_metrics(flight: BunkerFlight | None) -> FlightMetrics | None:
    """Summarise a flight, or return ``None`` when there is no flight to read out."""
    if flight is None:
        return None
    if not isinstance(flight, BunkerFlight):
        raise TypeError("flight must be a BunkerFlight or None")
    apex = max(flight.flight_m, key=lambda p: p[2])
    before, landing = flight.flight_m[-2], flight.flight_m[-1]
    horizontal = math.hypot(landing[0] - before[0], landing[1] - before[1])
    drop = before[2] - landing[2]
    descent = math.degrees(math.atan2(drop, horizontal)) if drop > 0.0 else None
    rest = flight.rollout_m[-1]
    rollout = sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(flight.rollout_m, flight.rollout_m[1:], strict=False)
    )
    return FlightMetrics(
        carry_m=flight.carry_m,
        apex_height_m=apex[2],
        apex_downrange_m=math.hypot(apex[0], apex[1]),
        descent_angle_deg=descent,
        rollout_m=rollout,
        total_m=math.hypot(rest[0], rest[1]),
        lateral_m=rest[1],
    )
