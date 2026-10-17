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

from src.engines.physics_engines.opensim.python import geometry_resolver
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.motion_matching.loaders._marker_clusters import y_up_to_z_up

__all__ = [
    "Segment",
    "body_segments",
    "draw_segments",
    "missing_mesh_bodies",
    "to_plot_frame",
    "up_axis_from_gravity",
]


@dataclass(frozen=True)
class Segment:
    """A capsule from ``start`` to ``end`` (metres, plot frame, Z up)."""

    name: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    body: str = ""


def up_axis_from_gravity(gravity: tuple[float, float, float]) -> int:
    """Index (1 = Y, 2 = Z) of the ground axis that points up.

    OpenSim's stock convention is Y up, but generated golf models are Z up; the
    model's own gravity vector is the authority, not an assumption.
    """
    g = np.asarray(gravity, dtype=float)
    if g.shape != (3,) or not np.any(g):
        raise ValueError(f"gravity must be a non-zero 3-vector, got {gravity!r}")
    axis = int(np.argmax(np.abs(g)))
    if axis == 0:
        raise ValueError(f"unsupported X-up gravity {gravity!r}")
    return axis


def to_plot_frame(
    points: np.ndarray, gravity: tuple[float, float, float]
) -> np.ndarray:
    """Express ground-frame ``points`` in the Z-up plot frame (single place)."""
    if up_axis_from_gravity(gravity) == 1:
        return y_up_to_z_up(np.asarray(points, dtype=float))
    return np.asarray(points, dtype=float)


def missing_mesh_bodies(model: Any, model_path: Path) -> frozenset[str]:
    """Bodies with no mesh, or any mesh that cannot be found (need a capsule)."""
    opensim = import_opensim()
    dirs = geometry_resolver.candidate_geometry_dirs()
    missing = set()
    bodies = model.getBodySet()
    for i in range(bodies.getSize()):
        body = bodies.get(i)
        names = []
        for j in range(body.getPropertyByName("attached_geometry").size()):
            mesh = opensim.Mesh.safeDownCast(body.get_attached_geometry(j))
            if mesh is not None:
                names.append(mesh.get_mesh_file())
        if not names or not all(
            geometry_resolver.mesh_resolves(n, model_path.parent, dirs) for n in names
        ):
            missing.add(body.getName())
    return frozenset(missing)


def _origin(frame: Any, state: Any) -> tuple[float, float, float]:
    v = frame.findBaseFrame().getPositionInGround(state)
    return (v.get(0), v.get(1), v.get(2))


def body_segments(
    model_path: Path | str, only_bodies: frozenset[str] | None = None
) -> list[Segment]:
    """Z-up joint segments in the default pose, for bodies needing a capsule.

    ``only_bodies`` limits output to those child bodies; ``None`` means every
    body whose meshes are missing (see :func:`missing_mesh_bodies`).
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"OpenSim model file not found: {path}")
    opensim = import_opensim()
    model = opensim.Model(str(path))
    state = model.initSystem()
    model.realizePosition(state)
    if only_bodies is None:
        only_bodies = missing_mesh_bodies(model, path)
    g = model.get_gravity()
    gravity = (g.get(0), g.get(1), g.get(2))
    segments = []
    joints = model.getJointSet()
    for i in range(joints.getSize()):
        joint = joints.get(i)
        child = joint.getChildFrame().findBaseFrame().getName()
        if child not in only_bodies:
            continue
        pts = to_plot_frame(
            np.array(
                [
                    _origin(joint.getParentFrame(), state),
                    _origin(joint.getChildFrame(), state),
                ]
            ),
            gravity,
        )
        start = (float(pts[0, 0]), float(pts[0, 1]), float(pts[0, 2]))
        end = (float(pts[1, 0]), float(pts[1, 1]), float(pts[1, 2]))
        if start != end:
            segments.append(Segment(joint.getName(), start, end, child))
    return segments


def draw_segments(fig: Any, segments: list[Segment], title: str = "") -> int:
    """Clear ``fig`` and draw ``segments`` as capsules; returns segments drawn."""
    fig.clear()
    ax = fig.add_subplot(111, projection="3d")
    pts = []
    for seg in segments:
        a, b = np.array(seg.start), np.array(seg.end)
        # Segments are already in the Z-up plot frame (see ``to_plot_frame``).
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
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
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z up (m)")
    if title:
        ax.set_title(title)
    return len(segments)
