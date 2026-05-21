"""Geometry utilities for coordinate math, bounding boxes, and collision detection.

DRY: Centralises the point/rect operations that were scattered across generator.py.
"""

from __future__ import annotations  # noqa: E402, F404

import math  # noqa: E402
from collections.abc import Sequence  # noqa: E402

from programmatic_pid.types import BBox, Point  # noqa: E402


def to_float(value: object, default: float = 0.0) -> float:
    """Safely convert any value to float.

    Precondition: default must be convertible to float.
    Postcondition: always returns a finite float.
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* into [lo, hi].

    Precondition: lo <= hi.
    """
    return max(lo, min(hi, value))


def closest_point_on_rect(
    point: tuple[float, float], rect: tuple[float, float, float, float]
) -> Point:
    """Return the closest point on *rect* to *point*."""
    if point is None:
        raise ValueError("point must be provided")
    px, py = to_float(point[0]), to_float(point[1])
    x1, y1, x2, y2 = rect
    return Point(clamp(px, x1, x2), clamp(py, y1, y2))


def rects_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    pad: float = 0.0,
) -> bool:
    """Return True if axis-aligned rectangles *a* and *b* overlap (with padding)."""
    if a is None:
        raise ValueError("a must be provided")
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (
        ax2 + pad <= bx1 or bx2 + pad <= ax1 or ay2 + pad <= by1 or by2 + pad <= ay1
    )


def text_box(
    text: str, x: float, y: float, h: float, align: str = "MIDDLE_CENTER"
) -> tuple[float, float, float, float]:
    """Estimate the bounding box of a text entity given its alignment."""
    if text is None:
        raise ValueError("text must be provided")
    text = str(text)
    h = max(to_float(h, 1.0), 0.1)
    width = max(len(text), 1) * h * 0.55
    height = h * 1.2
    align = str(align or "MIDDLE_CENTER").upper()

    if "LEFT" in align:
        x1, x2 = x, x + width
    elif "RIGHT" in align:
        x1, x2 = x - width, x
    else:
        x1, x2 = x - width / 2, x + width / 2

    if "TOP" in align:
        y1, y2 = y - height, y
    elif "BOTTOM" in align:
        y1, y2 = y, y + height
    else:
        y1, y2 = y - height / 2, y + height / 2
    return (x1, y1, x2, y2)


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def dedupe_points(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    """Remove consecutive duplicate points (within tolerance)."""
    cleaned: list[tuple[float, float]] = []
    for p in points:
        qx, qy = to_float(p[0]), to_float(p[1])
        if not cleaned:
            cleaned.append((qx, qy))
            continue
        px, py = cleaned[-1]
        if abs(px - qx) < 1e-9 and abs(py - qy) < 1e-9:
            continue
        cleaned.append((qx, qy))
    return cleaned


def find_free_region(
    occupied: list[BBox],
    width: float,
    height: float,
    search_origin: Point | None = None,
) -> BBox | None:
    """Find a free rectangular region that does not overlap any occupied box.

    Searches in a spiral pattern outward from *search_origin* (default 0,0).
    Returns None if nothing found within a reasonable search radius.
    """
    if occupied is None:
        raise ValueError("occupied must be provided")
    ox = search_origin.x if search_origin else 0.0
    oy = search_origin.y if search_origin else 0.0

    for radius in range(0, 500, 5):
        for dx in range(-radius, radius + 1, max(int(width), 5)):
            for dy in range(-radius, radius + 1, max(int(height), 5)):
                candidate = BBox(ox + dx, oy + dy, ox + dx + width, oy + dy + height)
                if not any(candidate.overlaps(o, pad=2.0) for o in occupied):
                    return candidate
    return None
