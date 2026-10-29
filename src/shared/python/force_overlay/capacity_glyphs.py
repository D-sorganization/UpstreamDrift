"""Wireframe capacity ellipsoid glyphs (CAP-5b, #14045).

MuJoCo has no wireframe ellipsoid; capacity kinds are drawn as line primitives:
three principal-axis segments with end ticks plus a thin equatorial rim. Inertial
kinds stay translucent shells (``EllipsoidGlyph`` in #14003).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.shared.python.force_overlay.ellipsoid_validation import (
    validate_oriented_ellipsoid_fields,
)

__all__ = [
    "CapacityWireGlyph",
    "LineSegmentGlyph",
    "capacity_wire_segments",
]


def _vec3(point: np.ndarray) -> tuple[float, float, float]:
    return (float(point[0]), float(point[1]), float(point[2]))


@dataclass(frozen=True)
class LineSegmentGlyph:
    """One world-space line segment for a renderer-neutral wireframe."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    width_m: float = 0.0025
    dashed: bool = False


@dataclass(frozen=True)
class CapacityWireGlyph:
    """Oriented capacity ellipsoid drawn as a wireframe, not a solid shell."""

    centre: tuple[float, float, float]
    axes: tuple[tuple[float, float, float], ...]
    radii: tuple[float, float, float]
    rgba: tuple[float, float, float, float]
    dashed: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        centre_t, axes_mat, radii_t, rgba_t = validate_oriented_ellipsoid_fields(
            self.centre, self.axes, self.radii, self.rgba
        )
        object.__setattr__(self, "centre", centre_t)
        object.__setattr__(
            self,
            "axes",
            tuple(tuple(float(axes_mat[j, i]) for j in range(3)) for i in range(3)),
        )
        object.__setattr__(self, "radii", radii_t)
        object.__setattr__(self, "rgba", rgba_t)


def _tick_segment(
    tip: np.ndarray,
    axis: np.ndarray,
    tick_m: float,
    rgba: tuple[float, float, float, float],
    *,
    width_m: float,
    dashed: bool,
) -> LineSegmentGlyph:
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    perp = np.cross(axis, ref)
    perp /= np.linalg.norm(perp)
    half = 0.5 * tick_m * perp
    return LineSegmentGlyph(
        start=_vec3(tip - half),
        end=_vec3(tip + half),
        rgba=rgba,
        width_m=width_m,
        dashed=dashed,
    )


def capacity_wire_segments(
    glyph: CapacityWireGlyph,
    *,
    rim_samples: int = 32,
    tick_length_m: float = 0.012,
    line_width_m: float = 0.0025,
) -> tuple[LineSegmentGlyph, ...]:
    """Principal axes, end ticks, and an equatorial rim in the a0-a1 plane."""
    if rim_samples < 8:
        raise ValueError("rim_samples must be at least 8")
    if tick_length_m <= 0 or line_width_m <= 0:
        raise ValueError("tick_length_m and line_width_m must be positive")
    centre = np.asarray(glyph.centre, dtype=float)
    axis_vectors = [np.asarray(axis, dtype=float) for axis in glyph.axes]
    radii = np.asarray(glyph.radii, dtype=float)
    rgba = glyph.rgba
    dashed = glyph.dashed
    segments: list[LineSegmentGlyph] = []
    axis_segments: list[LineSegmentGlyph] = []
    for i in range(3):
        direction = axis_vectors[i]
        radius = float(radii[i])
        start = centre - radius * direction
        end = centre + radius * direction
        axis_segments.append(
            LineSegmentGlyph(
                start=_vec3(start),
                end=_vec3(end),
                rgba=rgba,
                width_m=line_width_m,
                dashed=dashed,
            )
        )
        segments.append(
            _tick_segment(
                end, direction, tick_length_m, rgba, width_m=line_width_m, dashed=dashed
            )
        )
        segments.append(
            _tick_segment(
                start,
                direction,
                tick_length_m,
                rgba,
                width_m=line_width_m,
                dashed=dashed,
            )
        )
    segments = axis_segments + segments
    a0, a1 = axis_vectors[0], axis_vectors[1]
    r0, r1 = float(radii[0]), float(radii[1])
    angles = np.linspace(0.0, 2.0 * math.pi, rim_samples + 1)
    rim_pts = [centre + r0 * math.cos(t) * a0 + r1 * math.sin(t) * a1 for t in angles]
    for k in range(rim_samples):
        segments.append(
            LineSegmentGlyph(
                start=_vec3(rim_pts[k]),
                end=_vec3(rim_pts[k + 1]),
                rgba=rgba,
                width_m=0.75 * line_width_m,
                dashed=dashed,
            )
        )
    return tuple(segments)
