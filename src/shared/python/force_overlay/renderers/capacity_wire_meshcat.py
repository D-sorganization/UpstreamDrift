"""MeshCat renderer for capacity wireframe ellipsoids (#14045, CAP-5b)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.shared.python.force_overlay.capacity_glyphs import LineSegmentGlyph
from src.shared.python.force_overlay.renderers.meshcat_glyphs import (
    MeshcatSink,
    align_y_to,
)


def draw_capacity_wire_segments(
    sink: MeshcatSink,
    root: str,
    segments: Sequence[LineSegmentGlyph],
) -> set[str]:
    """Render line segments as thin cylinders; returns active paths."""
    active: set[str] = set()
    for index, seg in enumerate(segments):
        start = np.asarray(seg.start, dtype=np.float64)
        end = np.asarray(seg.end, dtype=np.float64)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            continue
        path = f"{root}/capacity_wire/{index}"
        sink.set_cylinder(
            path,
            length,
            seg.width_m,
            seg.width_m,
            seg.rgba,
        )
        rot = align_y_to(direction)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rot
        transform[:3, 3] = start + 0.5 * direction
        sink.set_transform(path, transform)
        active.add(path)
    return active
