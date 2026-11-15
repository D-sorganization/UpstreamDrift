"""Tangent wrap, shaft-parallel hinge bisection, and capsule clearances.

The tangent construction is the one in ``grip_hand_mesh._finger``: each
phalanx is tangent to a circle, and the proximal tangent point is the
midpoint. It is reimplemented here so this package does not call a private
mesh helper.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
)
from src.shared.python.model_appearance.hand_backends.grip_solid import GripSolid
from src.shared.python.model_appearance.hand_backends.settings import PoseSettings

Point = tuple[float, float, float]
_SHAFT_AXIS_TOL = 1e-6


def tangent_finger(
    seat_radius_m: float,
    azimuth_rad: float,
    z_m: float,
    lengths_m: Sequence[float],
    curl_sign: int,
) -> list[tuple[Point, Point, float, float, float]]:
    """Capsules tangent to the circle of radius ``seat_radius_m``.

    Returns ``(p0, p1, flexion_rad, wrap_rad, tangent_azimuth_rad)`` per
    segment. Preconditions: positive seat radius and lengths, and
    ``curl_sign`` of -1 or 1. Postcondition: each segment's nearest approach
    to the shaft is ``seat_radius_m`` at the reported tangent azimuth, and
    both endpoints share ``z_m``.
    """
    if curl_sign not in (-1, 1):
        raise ValueError(f"curl_sign must be -1 or 1, got {curl_sign!r}")
    if not math.isfinite(seat_radius_m) or seat_radius_m <= 0.0:
        raise ValueError(f"seat radius must be positive, got {seat_radius_m}")
    if not lengths_m:
        raise ValueError("tangent finger needs at least one segment")
    before = float(lengths_m[0]) / 2.0
    theta = azimuth_rad + curl_sign * math.atan2(before, seat_radius_m)
    posed: list[tuple[Point, Point, float, float, float]] = []
    for length in lengths_m:
        bone = float(length)
        if not math.isfinite(bone) or bone <= 0.0:
            raise ValueError(f"phalanx length must be positive, got {length!r}")
        after = bone - before
        if after <= 1e-9:
            raise ValueError(
                f"phalanx length {bone} cannot pass the tangent point at {before}"
            )
        along = (math.cos(theta), math.sin(theta), 0.0)
        tangent = (
            curl_sign * -math.sin(theta),
            curl_sign * math.cos(theta),
            0.0,
        )
        start = _add(_scale(along, seat_radius_m), _scale(tangent, -before), z_m)
        end = _add(_scale(along, seat_radius_m), _scale(tangent, after), z_m)
        flexion = math.atan2(before, seat_radius_m) + math.atan2(after, seat_radius_m)
        wrap = 2.0 * math.atan2(after, seat_radius_m)
        posed.append((start, end, flexion, wrap, theta))
        theta += curl_sign * wrap
        before = after
    return posed


def axial_segments(
    seat_radius_m: float,
    azimuth_rad: float,
    z_m: float,
    lengths_m: Sequence[float],
) -> list[tuple[Point, Point]]:
    """Segments parallel to +Z at a fixed radius. Used for a straight thumb."""
    if seat_radius_m <= 0.0:
        raise ValueError(f"seat radius must be positive, got {seat_radius_m}")
    radial = (math.cos(azimuth_rad), math.sin(azimuth_rad), 0.0)
    station = z_m
    posed: list[tuple[Point, Point]] = []
    for length in lengths_m:
        bone = float(length)
        if bone <= 0.0:
            raise ValueError(f"thumb length must be positive, got {length!r}")
        start = _add(_scale(radial, seat_radius_m), (0.0, 0.0, 0.0), station)
        station += bone
        end = _add(_scale(radial, seat_radius_m), (0.0, 0.0, 0.0), station)
        posed.append((start, end))
    return posed


def project_outside(point: Point, seat_radius_m: float) -> Point:
    """Push a point radially out to at least ``seat_radius_m`` from the shaft."""
    radial = math.hypot(point[0], point[1])
    if radial >= seat_radius_m or radial <= 1e-12:
        return point
    scale = seat_radius_m / radial
    return (point[0] * scale, point[1] * scale, point[2])


def blend_segment(
    straight: tuple[Point, Point],
    curled: tuple[Point, Point],
    straightness: float,
    seat_radius_m: float,
) -> tuple[Point, Point]:
    """Blend a straight thumb toward the tangent curl without entering the grip."""
    weight = min(1.0, max(0.0, straightness))
    start = project_outside(
        _lerp(curled[0], straight[0], weight),
        seat_radius_m,
    )
    end = project_outside(_lerp(curled[1], straight[1], weight), seat_radius_m)
    return start, end


def seat_radius(
    grip: GripSolid,
    z_m: float,
    finger_radius_m: float,
    target_gap_m: float,
    settings: PoseSettings,
) -> float:
    """Curl a clearance down from ``open_gap_m`` until it meets ``target_gap_m``.

    The last sample is the closed form ``radius_at + finger + target``. The
    loop is the curl the flat hand actually takes; it is not a decorative
    counter.
    """
    radius = grip.radius_at(z_m)
    if radius <= 0.0:
        raise ValueError(f"grip radius at z={z_m:.4f} m is {radius}, not positive")
    desired = radius + finger_radius_m + target_gap_m
    if settings.open_gap_m <= target_gap_m:
        return desired
    current = radius + finger_radius_m + settings.open_gap_m
    step = (current - desired) / (settings.curl_samples - 1)
    for _ in range(settings.curl_samples):
        gap = current - radius - finger_radius_m
        if gap <= target_gap_m + 1e-12:
            return current
        current = max(desired, current - step)
    return desired


def target_clearance(
    extra_m: float,
    depth: float,
    arch_weight: float,
    settings: PoseSettings,
    *,
    apply_arch: bool,
) -> float:
    """Flat clearance, optionally reduced by the metacarpal arch."""
    extra = extra_m
    if apply_arch:
        extra -= depth * arch_weight * settings.arch_travel_m
    return settings.base_clearance_m + max(0.0, extra)


def roll_point(point: Point, roll_deg: float) -> Point:
    """Right-hand rotation about the appearance shaft (+Z, butt toward head)."""
    if abs(roll_deg) <= 1e-15:
        return point
    rotation = apply_shaft_roll(np.eye(3), APPEARANCE_GRIP_SHAFT_AXIS, roll_deg)
    x = (
        rotation[0][0] * point[0]
        + rotation[0][1] * point[1]
        + rotation[0][2] * point[2]
    )
    y = (
        rotation[1][0] * point[0]
        + rotation[1][1] * point[1]
        + rotation[1][2] * point[2]
    )
    z = (
        rotation[2][0] * point[0]
        + rotation[2][1] * point[1]
        + rotation[2][2] * point[2]
    )
    return (float(x), float(y), float(z))


def shaft_angle_deg(start: Point, end: Point) -> float:
    """Angle between a segment and the shaft. 90 means the bone crosses it."""
    direction = (end[0] - start[0], end[1] - start[1], end[2] - start[2])
    length = math.dist(direction, (0.0, 0.0, 0.0))
    if length <= 1e-12:
        raise ValueError("segment length is zero")
    cosine = abs(direction[2]) / length
    cosine = min(1.0, max(-1.0, cosine))
    return math.degrees(math.acos(cosine))


def segment_azimuth_rad(start: Point, end: Point) -> float:
    """Azimuth of the segment midpoint about the shaft."""
    mid = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
    return math.atan2(mid[1], mid[0])


def grip_gap_m(
    start: Point,
    end: Point,
    radius_m: float,
    grip: GripSolid,
    samples: int,
) -> float:
    """Minimum signed surface gap from a capsule to the grip (metres)."""
    if samples < 2:
        raise ValueError(f"samples must be >= 2, got {samples}")
    best = math.inf
    for station in _sample_ts(start, end, samples):
        point = _lerp(start, end, station)
        radial = math.hypot(point[0], point[1])
        gap = radial - grip.radius_at(point[2]) - radius_m
        if gap < best:
            best = gap
    return float(best)


def surface_gap_m(
    start: Point,
    end: Point,
    radius_m: float,
    other_start: Point,
    other_end: Point,
    other_radius_m: float,
    samples: int,
) -> float:
    """Minimum sampled surface gap between two capsules (metres)."""
    if samples < 2:
        raise ValueError(f"samples must be >= 2, got {samples}")
    best = math.inf
    for left in _sample_ts(start, end, samples):
        a = _lerp(start, end, left)
        for right in _sample_ts(other_start, other_end, samples):
            b = _lerp(other_start, other_end, right)
            gap = math.dist(a, b) - radius_m - other_radius_m
            if gap < best:
                best = gap
    return float(best)


def require_shaft_parallel(axis: Point, name: str) -> None:
    """This partial hinge solver only bends about the shaft axis."""
    length = math.dist(axis, (0.0, 0.0, 0.0))
    if length <= 1e-12:
        raise ValueError(f"{name} axis must be non-zero")
    unit = (axis[0] / length, axis[1] / length, axis[2] / length)
    drift = abs(abs(unit[2]) - 1.0)
    if drift > _SHAFT_AXIS_TOL or abs(unit[0]) > _SHAFT_AXIS_TOL or abs(unit[1]) > 1e-5:
        raise ValueError(
            f"{name} hinge axis {axis} is not parallel to the shaft. "
            "This poser only bisects shaft-parallel hinges; a finger axis in "
            "another frame needs the non-Z solver (see the posing handoff)."
        )


def hinge_endpoints(
    origin: Point,
    rest_direction: Point,
    length_m: float,
    angle_rad: float,
    ancestor_rad: float,
    parent_origin: Point | None,
) -> tuple[Point, Point]:
    """Bone from a shaft-parallel hinge, including ancestor rotation about Z."""
    if parent_origin is None:
        start = origin
    else:
        offset = (
            origin[0] - parent_origin[0],
            origin[1] - parent_origin[1],
            origin[2] - parent_origin[2],
        )
        spun = _rotate_z(offset, ancestor_rad)
        start = (
            parent_origin[0] + spun[0],
            parent_origin[1] + spun[1],
            parent_origin[2] + spun[2],
        )
    direction = _rotate_z(rest_direction, ancestor_rad + angle_rad)
    norm = math.dist(direction, (0.0, 0.0, 0.0))
    if norm <= 1e-12:
        raise ValueError("hinge rest direction vanished")
    end = (
        start[0] + direction[0] / norm * length_m,
        start[1] + direction[1] / norm * length_m,
        start[2] + direction[2] / norm * length_m,
    )
    return start, end


def hinge_child_endpoints(
    origin: Point,
    rest_direction: Point,
    length_m: float,
    angle_rad: float,
    ancestor_rad: float,
    parent_world: Point | None,
    parent_rest: Point | None,
) -> tuple[Point, Point]:
    """Hinge bone whose parent origin may already have moved.

    ``parent_rest`` is the parent origin with every hinge at zero.
    ``parent_world`` is that origin after the ancestor hinges have moved it.
    A root hinge passes both parents as ``None``.
    """
    if parent_world is None or parent_rest is None:
        return hinge_endpoints(origin, rest_direction, length_m, angle_rad, 0.0, None)
    offset = (
        origin[0] - parent_rest[0],
        origin[1] - parent_rest[1],
        origin[2] - parent_rest[2],
    )
    spun = _rotate_z(offset, ancestor_rad)
    start = (
        parent_world[0] + spun[0],
        parent_world[1] + spun[1],
        parent_world[2] + spun[2],
    )
    direction = _rotate_z(rest_direction, ancestor_rad + angle_rad)
    norm = math.dist(direction, (0.0, 0.0, 0.0))
    if norm <= 1e-12:
        raise ValueError("hinge rest direction vanished")
    end = (
        start[0] + direction[0] / norm * length_m,
        start[1] + direction[1] / norm * length_m,
        start[2] + direction[2] / norm * length_m,
    )
    return start, end


def bisect_contact(
    gap_at,
    limit_rad: float,
    settings: PoseSettings,
) -> float:
    """Least angle in ``[0, limit]`` whose gap is inside the contact band.

    ``gap_at`` must decrease as the angle increases. A coarse grid can step
    over the 2 mm band, so this is bisection.
    """
    if limit_rad < 0.0:
        raise ValueError(f"hinge limit must be non-negative, got {limit_rad}")
    band_hi = settings.contact_gap_tolerance_m
    band_lo = -settings.penetration_tolerance_m
    opening = float(gap_at(0.0))
    if band_lo <= opening <= band_hi:
        return 0.0
    if opening < band_lo:
        raise ValueError(f"hinge rest pose already penetrates (gap {opening} m)")
    lo = 0.0
    hi = limit_rad
    for _ in range(settings.hinge_bisection_steps):
        mid = 0.5 * (lo + hi)
        if float(gap_at(mid)) > band_hi:
            lo = mid
        else:
            hi = mid
    gap = float(gap_at(hi))
    if not band_lo <= gap <= band_hi:
        raise ValueError(
            "hinge bisection missed the contact window: "
            f"gap {gap} m at {hi} rad (limit {limit_rad} rad)"
        )
    return hi


def _sample_ts(start: Point, end: Point, samples: int) -> list[float]:
    fractions = [index / (samples - 1) for index in range(samples)]
    radial = _closest_axis_fraction(start, end)
    if radial is not None:
        fractions.append(radial)
    return fractions


def _closest_axis_fraction(start: Point, end: Point) -> float | None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denom = dx * dx + dy * dy
    if denom <= 1e-18:
        return None
    fraction = -(start[0] * dx + start[1] * dy) / denom
    if fraction < 0.0 or fraction > 1.0:
        return None
    return fraction


def _rotate_z(point: Point, angle: float) -> Point:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        cosine * point[0] - sine * point[1],
        sine * point[0] + cosine * point[1],
        point[2],
    )


def _scale(point: Point, factor: float) -> Point:
    return (point[0] * factor, point[1] * factor, point[2] * factor)


def _add(radial: Point, tangent: Point, z_m: float) -> Point:
    return (radial[0] + tangent[0], radial[1] + tangent[1], z_m)


def _lerp(start: Point, end: Point, fraction: float) -> Point:
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[1] + (end[1] - start[1]) * fraction,
        start[2] + (end[2] - start[2]) * fraction,
    )


__all__ = [
    "axial_segments",
    "bisect_contact",
    "blend_segment",
    "grip_gap_m",
    "hinge_child_endpoints",
    "hinge_endpoints",
    "project_outside",
    "require_shaft_parallel",
    "roll_point",
    "seat_radius",
    "segment_azimuth_rad",
    "shaft_angle_deg",
    "surface_gap_m",
    "tangent_finger",
    "target_clearance",
]
