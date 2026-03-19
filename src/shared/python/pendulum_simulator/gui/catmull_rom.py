"""
Catmull-Rom spline interpolation — pure math, no Qt dependency.

Extracted from base_pendulum_widget.py so it can be tested headlessly.
Used by trail rendering to smooth the path of pendulum endpoints.
"""

from __future__ import annotations


def catmull_rom_smooth(
    points: list[tuple[float, float]],
    n_sub: int = 4,
) -> list[tuple[float, float]]:
    """Catmull-Rom spline interpolation over trail points (#1116).

    Preconditions:
        n_sub >= 1
    Postconditions:
        len(result) >= len(points)
        result[-1] == points[-1]  (endpoint preserved)

    If fewer than 4 points are provided, the input is returned unchanged
    (not enough control points for Catmull-Rom).
    """
    assert n_sub >= 1, f"n_sub must be >= 1, got {n_sub}"
    n = len(points)
    if n < 4:
        return points

    # Duplicate first/last for boundary tangents
    pts = [points[0]] + list(points) + [points[-1]]
    result: list[tuple[float, float]] = []

    for i in range(1, len(pts) - 2):
        p0 = pts[i - 1]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[i + 2]
        for j in range(n_sub):
            t = j / n_sub
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            result.append((x, y))
    result.append(points[-1])
    return result
