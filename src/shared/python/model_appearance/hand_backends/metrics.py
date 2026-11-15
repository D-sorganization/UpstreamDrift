"""Contact-quality record every hand backend's evidence must report.

Gap is the positive clearance. Penetration is how far a capsule has crossed
the grip surface. ``within_tolerance`` uses the intended contact (the grip,
or a neighbour when a style lifts a finger off the grip) and still rejects
grip penetration on every capsule.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.shared.python.model_appearance.hand_backends.report import (
    HandPose,
    SegmentPose,
)
from src.shared.python.model_appearance.hand_backends.settings import PoseSettings

SCHEMA = "hand_contact_quality_v1"
CONTACT_QUALITY_METRIC_KEYS = (
    "schema",
    "contact_gap_tolerance_m",
    "penetration_tolerance_m",
    "per_segment_gap_m",
    "per_segment_grip_gap_m",
    "per_segment_penetration_m",
    "per_segment_neighbour_gap_m",
    "wrap_angle_rad",
    "palm_contact_area_m2",
    "max_gap_m",
    "max_penetration_m",
    "within_tolerance",
)


def palm_contact_area_m2(depth: float, settings: PoseSettings) -> float:
    """Area of a uniform cylindrical palm patch at ``depth``.

    The patch is not a mesh. Every cell shares one gap,
    ``palm_standoff_m - depth * palm_arch_travel_m``. Depth 0 is outside the
    band (area 0). Depth 1 is inside it (the full grid).
    """
    if isinstance(depth, bool) or not isinstance(depth, (int, float)):
        raise TypeError(f"depth must be a number, got {depth!r}")
    if not math.isfinite(float(depth)) or not 0.0 <= float(depth) <= 1.0:
        raise ValueError(f"depth must be in [0, 1], got {depth!r}")
    gap = settings.palm_standoff_m - float(depth) * settings.palm_arch_travel_m
    if -settings.penetration_tolerance_m <= gap <= settings.contact_gap_tolerance_m:
        cells = settings.palm_grid_nu * settings.palm_grid_nv
        return float(cells * settings.palm_cell_m * settings.palm_cell_m)
    return 0.0


def contact_quality_metrics(
    lead: HandPose,
    trail: HandPose,
    settings: PoseSettings,
) -> dict[str, Any]:
    """Metrics for a posed pair of hands.

    Preconditions: ``lead`` and ``trail`` are HandPose values from this poser.
    Postcondition: ``schema`` is ``hand_contact_quality_v1`` and
    ``within_tolerance`` is true only when every required contact is inside
    the configured band and no capsule penetrates the grip.
    """
    segments = tuple(lead.segments) + tuple(trail.segments)
    return _from_segments(
        segments,
        {"lead": lead.palm_contact_area_m2, "trail": trail.palm_contact_area_m2},
        _wraps(lead, trail),
        settings,
    )


def metrics_from_tables(
    *,
    gaps_m: Mapping[str, float],
    penetrations_m: Mapping[str, float],
    wrap_rad: Mapping[str, float],
    palm_area_m2: Mapping[str, float],
    settings: PoseSettings,
    grip_gaps_m: Mapping[str, float] | None = None,
    neighbour_gaps_m: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Same schema for a backend that measured its own mesh.

    ``gaps_m`` is the intended contact gap. A backend with no neighbour rule
    passes the grip gap there. A gap key with no penetration key is rejected.
    """
    if set(gaps_m) != set(penetrations_m):
        raise ValueError("gaps_m and penetrations_m must name the same segments")
    grip = dict(gaps_m if grip_gaps_m is None else grip_gaps_m)
    neighbour = {} if neighbour_gaps_m is None else dict(neighbour_gaps_m)
    required_gap = [float(value) for value in gaps_m.values()]
    penetration = [float(value) for value in penetrations_m.values()]
    neighbour_pen = [max(0.0, -float(value)) for value in neighbour.values()]
    within = _within(required_gap, penetration, neighbour_pen, settings)
    return _record(
        gaps=dict(gaps_m),
        grip_gaps=grip,
        penetrations=dict(penetrations_m),
        neighbours=neighbour,
        wraps=dict(wrap_rad),
        palm=dict(palm_area_m2),
        max_gap=max(required_gap, default=0.0),
        max_penetration=max([*penetration, *neighbour_pen], default=0.0),
        within=within,
        settings=settings,
    )


def _from_segments(
    segments: tuple[SegmentPose, ...],
    palm: dict[str, float],
    wraps: dict[str, float],
    settings: PoseSettings,
) -> dict[str, Any]:
    gaps: dict[str, float] = {}
    grip_gaps: dict[str, float] = {}
    penetrations: dict[str, float] = {}
    neighbours: dict[str, float] = {}
    required: list[float] = []
    pens: list[float] = []
    neighbour_pens: list[float] = []
    for segment in segments:
        gaps[segment.name] = segment.gap_m
        grip_gaps[segment.name] = segment.grip_gap_m
        penetrations[segment.name] = segment.penetration_m
        pens.append(segment.penetration_m)
        if segment.neighbour_gap_m is not None:
            neighbours[segment.name] = segment.neighbour_gap_m
            neighbour_pens.append(max(0.0, -segment.neighbour_gap_m))
        if segment.check_grip:
            required.append(segment.grip_gap_m)
        if segment.check_neighbour:
            if segment.neighbour_gap_m is None:
                raise ValueError(f"{segment.name} is missing its neighbour gap")
            required.append(segment.neighbour_gap_m)
    return _record(
        gaps=gaps,
        grip_gaps=grip_gaps,
        penetrations=penetrations,
        neighbours=neighbours,
        wraps=wraps,
        palm=palm,
        max_gap=max(required, default=0.0),
        max_penetration=max([*pens, *neighbour_pens], default=0.0),
        within=_within(required, pens, neighbour_pens, settings),
        settings=settings,
    )


def _wraps(lead: HandPose, trail: HandPose) -> dict[str, float]:
    totals: dict[str, float] = {}
    for pose in (lead, trail):
        for segment in pose.segments:
            if segment.finger == "thumb":
                continue
            key = f"{pose.side}_{segment.finger}"
            totals[key] = totals.get(key, 0.0) + segment.wrap_rad
    return totals


def _within(
    required: list[float],
    penetrations: list[float],
    neighbour_penetrations: list[float],
    settings: PoseSettings,
) -> bool:
    gap_tol = settings.contact_gap_tolerance_m
    pen_tol = settings.penetration_tolerance_m
    for gap in required:
        if gap < -pen_tol or gap > gap_tol:
            return False
    for pen in penetrations:
        if pen > pen_tol:
            return False
    return all(pen <= pen_tol for pen in neighbour_penetrations)


def _record(
    *,
    gaps: dict[str, float],
    grip_gaps: dict[str, float],
    penetrations: dict[str, float],
    neighbours: dict[str, float],
    wraps: dict[str, float],
    palm: dict[str, float],
    max_gap: float,
    max_penetration: float,
    within: bool,
    settings: PoseSettings,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "contact_gap_tolerance_m": settings.contact_gap_tolerance_m,
        "penetration_tolerance_m": settings.penetration_tolerance_m,
        "per_segment_gap_m": gaps,
        "per_segment_grip_gap_m": grip_gaps,
        "per_segment_penetration_m": penetrations,
        "per_segment_neighbour_gap_m": neighbours,
        "wrap_angle_rad": wraps,
        "palm_contact_area_m2": palm,
        "max_gap_m": max_gap,
        "max_penetration_m": max_penetration,
        "within_tolerance": within,
    }
    missing = set(CONTACT_QUALITY_METRIC_KEYS) - set(record)
    if missing:
        raise RuntimeError(f"contact metrics missing {sorted(missing)}")
    return record


__all__ = [
    "CONTACT_QUALITY_METRIC_KEYS",
    "SCHEMA",
    "contact_quality_metrics",
    "metrics_from_tables",
    "palm_contact_area_m2",
]
