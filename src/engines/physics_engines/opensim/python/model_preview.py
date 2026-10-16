"""Primitive (capsule) preview of an OpenSim model for the explorer (#12919).

Draws one thick rounded segment per joint between the parent and child body
origins, so a model whose body meshes are unavailable still shows its skeleton
instead of a blank viewport.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.engine_core.opensim_runtime import import_opensim

__all__ = ["Segment", "body_segments", "draw_segments"]


@dataclass(frozen=True)
class Segment:
    """A capsule from ``start`` to ``end`` (metres, ground frame, Y up)."""

    name: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]


def _origin(frame: Any, state: Any) -> tuple[float, float, float]:
    v = frame.findBaseFrame().getPositionInGround(state)
    return (v.get(0), v.get(1), v.get(2))


def body_segments(model_path: Path | str) -> list[Segment]:
    """Joint segments of the model in its default pose."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"OpenSim model file not found: {path}")
    opensim = import_opensim()
    model = opensim.Model(str(path))
    state = model.initSystem()
    model.realizePosition(state)
    segments = []
    joints = model.getJointSet()
    for i in range(joints.getSize()):
        joint = joints.get(i)
        start = _origin(joint.getParentFrame(), state)
        end = _origin(joint.getChildFrame(), state)
        if start != end:
            segments.append(Segment(joint.getName(), start, end))
    return segments


def draw_segments(fig: Any, segments: list[Segment], title: str = "") -> int:
    """Clear ``fig`` and draw ``segments`` as capsules; returns segments drawn."""
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    pts = []
    for seg in segments:
        a, b = np.array(seg.start), np.array(seg.end)
        # OpenSim is Y-up; matplotlib is Z-up, so plot (x, z, y).
        ax.plot(
            [a[0], b[0]],
            [a[2], b[2]],
            [a[1], b[1]],
            color="#2f6fb0",
            linewidth=5,
            solid_capstyle="round",
        )
        pts.extend([a, b])
    if pts:
        arr = np.array(pts)
        centre = (arr.min(axis=0) + arr.max(axis=0)) / 2
        half = max(float((arr.max(axis=0) - arr.min(axis=0)).max()) / 2, 0.1)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[2] - half, centre[2] + half)
        ax.set_zlim(centre[1] - half, centre[1] + half)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Y (m)")
    if title:
        ax.set_title(title)
    return len(segments)
