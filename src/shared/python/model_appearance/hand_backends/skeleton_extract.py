"""Joint polylines from the capsule placement skeleton."""

from __future__ import annotations

import numpy as np

from src.shared.python.model_appearance.grip_hand_mesh import (
    FINGER_NAMES,
    GripHandMesh,
    PHALANX_NAMES,
)


def _capsule_endpoints(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0] / np.linalg.norm(vh[0])
    proj = centered @ axis
    return (
        centroid + axis * float(proj.min()),
        centroid + axis * float(proj.max()),
    )


def finger_polyline(baked: GripHandMesh, finger: str) -> np.ndarray:
    """Ordered joint centres for one finger (MCP → tip)."""
    chains: list[np.ndarray] = []
    for bone in PHALANX_NAMES:
        name = f"{finger}_{bone}"
        part = next((c for c in baked.components if c.name == name), None)
        if part is None:
            continue
        start, end = _capsule_endpoints(part.mesh.vertices)
        if not chains:
            chains.append(start)
        chains.append(end)
    if len(chains) < 2:
        raise ValueError(f"missing phalanx chain for {finger!r}")
    return np.asarray(chains, dtype=float)


def thumb_polyline(baked: GripHandMesh) -> np.ndarray:
    chains: list[np.ndarray] = []
    for part in baked.components:
        if part.finger != "thumb":
            continue
        start, end = _capsule_endpoints(part.mesh.vertices)
        if not chains:
            chains.append(start)
        chains.append(end)
    if len(chains) < 2:
        raise ValueError("missing thumb chain")
    return np.asarray(chains, dtype=float)


def palm_z_extent(baked: GripHandMesh) -> tuple[float, float]:
    palm = next(c for c in baked.components if c.finger == "palm")
    verts = palm.mesh.vertices
    return float(verts[:, 2].min()), float(verts[:, 2].max())


def scale_points_about_centre(
    points: list[np.ndarray], centre: np.ndarray, scale: float
) -> list[np.ndarray]:
    return [centre + (p - centre) * scale for p in points]


def uniform_scale_skeleton(
    points_list: list[np.ndarray],
    *,
    target_length_m: float,
    measured_length_m: float,
) -> list[np.ndarray]:
    if measured_length_m < 1e-6:
        return points_list
    s = target_length_m / measured_length_m
    all_pts = np.vstack(points_list)
    centre = all_pts.mean(axis=0)
    return scale_points_about_centre(points_list, centre, s)


def middle_finger_length(points: np.ndarray) -> float:
    total = 0.0
    for i in range(len(points) - 1):
        total += float(np.linalg.norm(points[i + 1] - points[i]))
    return total


__all__ = [
    "finger_polyline",
    "middle_finger_length",
    "palm_z_extent",
    "thumb_polyline",
    "uniform_scale_skeleton",
]
