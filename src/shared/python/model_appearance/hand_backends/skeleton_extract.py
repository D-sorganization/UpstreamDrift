"""Joint polylines from the capsule placement skeleton."""

from __future__ import annotations

import numpy as np

from src.shared.python.model_appearance.grip_hand_mesh import (
    GripHandMesh,
    PHALANX_NAMES,
)


def _capsule_endpoints(
    vertices: np.ndarray,
    *,
    hint_dir: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the inner barrel endpoints ``(start, end)`` of one capsule."""
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0] / max(float(np.linalg.norm(vh[0])), 1e-12)
    if hint_dir is not None and float(np.dot(axis, hint_dir)) < 0.0:
        axis = -axis
    proj = centered @ axis
    perp = np.linalg.norm(centered - np.outer(proj, axis), axis=1)
    cap_radius = float(perp.max()) if len(perp) else 0.0
    lo = float(proj.min()) + cap_radius
    hi = float(proj.max()) - cap_radius
    if lo >= hi:
        lo, hi = float(proj.min()), float(proj.max())
    return centroid + axis * lo, centroid + axis * hi


def _chain_connected_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Order segment endpoint pairs proximally to distally by shared-joint matching."""
    if len(segments) < 2:
        raise ValueError("need at least 2 segments to chain a digit polyline")
    best_dist = float("inf")
    best_pair = (0, 0)
    a0, b0 = segments[0]
    a1, b1 = segments[1]
    for i, u in enumerate((a0, b0)):
        for j, v in enumerate((a1, b1)):
            d = float(np.linalg.norm(u - v))
            if d < best_dist:
                best_dist = d
                best_pair = (i, j)
    start_0 = b0 if best_pair[0] == 0 else a0
    joint_1 = 0.5 * ((a0, b0)[best_pair[0]] + (a1, b1)[best_pair[1]])
    prev_end = b1 if best_pair[1] == 0 else a1
    chain = [start_0, joint_1, prev_end]
    for a_next, b_next in segments[2:]:
        d_a = float(np.linalg.norm(a_next - prev_end))
        d_b = float(np.linalg.norm(b_next - prev_end))
        if d_a <= d_b:
            chain[-1] = 0.5 * (prev_end + a_next)
            prev_end = b_next
        else:
            chain[-1] = 0.5 * (prev_end + b_next)
            prev_end = a_next
        chain.append(prev_end)
    return np.asarray(chain, dtype=float)


def finger_polyline(baked: GripHandMesh, finger: str) -> np.ndarray:
    """Ordered joint centres for one finger (MCP -> PIP -> DIP -> tip)."""
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for bone in PHALANX_NAMES:
        name = f"{finger}_{bone}"
        part = next((c for c in baked.components if c.name == name), None)
        if part is not None:
            segments.append(_capsule_endpoints(part.mesh.vertices))
    if len(segments) < 2:
        raise ValueError(f"missing phalanx chain for {finger!r}")
    return _chain_connected_segments(segments)


def thumb_polyline(baked: GripHandMesh) -> np.ndarray:
    """Ordered joint centres for the thumb (MCP -> IP -> tip)."""
    segments = [
        _capsule_endpoints(part.mesh.vertices)
        for bone in ("proximal", "distal")
        for part in baked.components
        if part.name == f"thumb_{bone}"
    ]
    if len(segments) < 2:
        raise ValueError("missing thumb chain")
    return _chain_connected_segments(segments)


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
    "scale_points_about_centre",
    "thumb_polyline",
    "uniform_scale_skeleton",
]
