"""Choose the default gaze weight from a sweep (OSV-3b, #11729).

The soft gaze residual trades marker fidelity for a quieter, ball-fixating
head. The default weight is not chosen by taste: among the sweep points that
keep the marker RMS within ``marker_tolerance`` of the weight-0 run and the
OSV-10 face-angle error at or below ``face_cap_deg``, take the knee of the
Pareto front of (marker RMS, gaze-error RMS), the point farthest from the
chord joining the front's two ends in normalised coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

MARKER_TOLERANCE = 0.10
FACE_CAP_DEG = 5.0


@dataclass(frozen=True)
class SweepPoint:
    """One sweep run: weight and the three measured trade-off quantities."""

    weight: float
    marker_rms_mm: float
    gaze_rms_deg: float
    face_error_deg: float

    def __post_init__(self) -> None:
        for name in ("weight", "marker_rms_mm", "gaze_rms_deg", "face_error_deg"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative, got {value}")


def _baseline(points: Sequence[SweepPoint]) -> SweepPoint:
    for p in points:
        if p.weight == 0:
            return p
    raise ValueError("the sweep needs its weight 0 baseline")


def feasible(
    points: Sequence[SweepPoint],
    marker_tolerance: float = MARKER_TOLERANCE,
    face_cap_deg: float = FACE_CAP_DEG,
) -> list[SweepPoint]:
    """Points within the marker tolerance of weight 0 and under the face cap."""
    limit = _baseline(points).marker_rms_mm * (1.0 + marker_tolerance)
    return [
        p
        for p in points
        if p.marker_rms_mm <= limit and p.face_error_deg <= face_cap_deg
    ]


def pareto_front(points: Sequence[SweepPoint]) -> list[SweepPoint]:
    """Non-dominated points (lower marker RMS and lower gaze RMS), by weight."""
    front = [
        p
        for p in points
        if not any(
            q is not p
            and q.marker_rms_mm <= p.marker_rms_mm
            and q.gaze_rms_deg <= p.gaze_rms_deg
            and (q.marker_rms_mm < p.marker_rms_mm or q.gaze_rms_deg < p.gaze_rms_deg)
            for q in points
        )
    ]
    return sorted(front, key=lambda p: p.weight)


def select_knee(
    points: Sequence[SweepPoint],
    marker_tolerance: float = MARKER_TOLERANCE,
    face_cap_deg: float = FACE_CAP_DEG,
) -> SweepPoint:
    """Knee of the feasible Pareto front; weight 0 if nothing else qualifies."""
    front = pareto_front(feasible(points, marker_tolerance, face_cap_deg))
    if len(front) < 3:
        return min(front, key=lambda p: p.gaze_rms_deg)
    by_marker = sorted(front, key=lambda p: p.marker_rms_mm)
    lo, hi = by_marker[0], by_marker[-1]
    m_span = (hi.marker_rms_mm - lo.marker_rms_mm) or 1.0
    g_span = (lo.gaze_rms_deg - hi.gaze_rms_deg) or 1.0

    def coords(p: SweepPoint) -> tuple[float, float]:
        return (
            (p.marker_rms_mm - lo.marker_rms_mm) / m_span,
            (p.gaze_rms_deg - hi.gaze_rms_deg) / g_span,
        )

    # Normalised chord runs (0, 1) -> (1, 0); distance below it is 1 - x - y.
    return max(front, key=lambda p: 1.0 - sum(coords(p)))
