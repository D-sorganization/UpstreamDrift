"""Wireframe capacity ellipsoid glyphs (CAP-5b, #14045).

MuJoCo has no wireframe ellipsoid; capacity kinds are drawn as line primitives:
three principal-axis segments with end ticks plus a thin equatorial rim. Inertial
kinds stay translucent shells (``EllipsoidGlyph`` in #14003).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CapacityWireGlyph",
    "LineSegmentGlyph",
    "capacity_wire_segments",
]


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
        centre = np.asarray(self.centre, dtype=float)
        if centre.shape != (3,) or not np.isfinite(centre).all():
            raise ValueError(f"centre must be a finite 3-vector, got {self.centre!r}")
        axes = np.asarray(self.axes, dtype=float)
        if axes.shape != (3, 3) or not np.isfinite(axes).all():
            raise ValueError("axes must be a finite 3x3 matrix")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6):
            raise ValueError("axes must have orthonormal columns (within 1e-6)")
        if np.linalg.det(axes) < 0.0:
            axes = axes.copy()
            axes[:, 2] *= -1.0
        radii = np.asarray(self.radii, dtype=float)
        if radii.shape != (3,) or not np.isfinite(radii).all() or (radii <= 0).any():
            raise ValueError(
                f"radii must be 3 finite positive values, got {self.radii!r}"
            )
        rgba = np.asarray(self.rgba, dtype=float)
        if rgba.shape != (4,) or not ((rgba >= 0.0) & (rgba <= 1.0)).all():
            raise ValueError(f"rgba must be 4 values in [0, 1], got {self.rgba!r}")
        object.__setattr__(self, "centre", tuple(float(v) for v in centre))
        object.__setattr__(
            self,
            "axes",
            tuple(tuple(float(axes[j, i]) for j in range(3)) for i in range(3)),
        )
        object.__setattr__(self, "radii", tuple(float(v) for v in radii))
        object.__setattr__(self, "rgba", tuple(float(v) for v in rgba))


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
        start=tuple(float(v) for v in tip - half),
        end=tuple(float(v) for v in tip + half),
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
                start=tuple(start),
                end=tuple(end),
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
                start=tuple(rim_pts[k]),
                end=tuple(rim_pts[k + 1]),
                rgba=rgba,
                width_m=0.75 * line_width_m,
                dashed=dashed,
            )
        )
    return tuple(segments)
