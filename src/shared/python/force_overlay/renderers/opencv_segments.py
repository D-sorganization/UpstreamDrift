"""OpenCV segment mesh renderer for filled, shaded segment volumes (FTO-26, #11311).

Draws filled, shaded segment meshes onto video frames, coloured by tension
and compression axial loads when available, using the painter's algorithm
for depth sorting and Lambertian reflectance for shading.
"""

from __future__ import annotations

import functools
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import numpy.typing as npt

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.body_part_viz.shapes import CapsuleShape
from src.shared.python.core.contracts import require
from src.shared.python.force_overlay.conversions import SegmentAxis

if TYPE_CHECKING:
    from src.shared.python.body_part_viz.asset_library import ShapeLibrary

logger = logging.getLogger(__name__)

__all__ = [
    "SegmentDrawReceipt",
    "SegmentPose",
    "SegmentShading",
    "draw_segment_meshes_on_frame",
    "segment_poses_from_axes",
]

_NEAR_Z_TOL: float = 1e-6


@dataclass(frozen=True)
class SegmentPose:
    """Rigid pose and scale for a single body segment.

    Parameters
    ----------
    name : str
        Canonical segment name matching axial load labels.
    mesh_id : str
        Identifier in ShapeLibrary (e.g. 'forearm', 'thigh') or 'capsule'.
    T_world_segment : np.ndarray
        4x4 homogeneous transformation matrix from segment-local to world frame.
    scale : tuple[float, float, float]
        Per-axis scale factor (sx, sy, sz), all positive.
    base_color : str
        Default color hex string (e.g. '#1f77b4') when loads are unmeasured.
    """

    name: str
    mesh_id: str
    T_world_segment: np.ndarray
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    base_color: str = "#1f77b4"

    def __post_init__(self) -> None:
        require(
            isinstance(self.name, str) and bool(self.name.strip()),
            "name must be a non-empty string",
        )
        require(
            isinstance(self.mesh_id, str) and bool(self.mesh_id.strip()),
            "mesh_id must be a non-empty string",
        )
        t = np.asarray(self.T_world_segment, dtype=float)
        require(t.shape == (4, 4), f"T_world_segment must be 4x4; got {t.shape}")
        require(bool(np.all(np.isfinite(t))), "T_world_segment must be finite")
        r = t[:3, :3]
        require(
            bool(np.allclose(r.T @ r, np.eye(3), atol=1e-4)),
            "rotation matrix must be orthonormal",
        )
        det = float(np.linalg.det(r))
        require(abs(det - 1.0) < 1e-4, f"rotation determinant must be +1; got {det}")
        require(
            bool(np.allclose(t[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6)),
            "bottom row must be [0, 0, 0, 1]",
        )

        require(len(self.scale) == 3, "scale must have 3 components")
        for s in self.scale:
            require(
                float(s) > 0.0 and math.isfinite(float(s)),
                "scale components must be positive and finite",
            )
        object.__setattr__(self, "T_world_segment", t)
        object.__setattr__(
            self,
            "scale",
            (float(self.scale[0]), float(self.scale[1]), float(self.scale[2])),
        )


@dataclass(frozen=True)
class SegmentShading:
    """Visual shading and compositing parameters."""

    ambient: float = 0.35
    opacity: float = 0.55
    max_triangles: int = 20_000
    backface_culling: bool = True

    def __post_init__(self) -> None:
        require(0.0 <= self.ambient <= 1.0, "ambient must be in [0.0, 1.0]")
        require(0.0 <= self.opacity <= 1.0, "opacity must be in [0.0, 1.0]")
        require(self.max_triangles > 0, "max_triangles must be positive")


@dataclass(frozen=True)
class SegmentDrawReceipt:
    """Execution receipt recording triangle counts and timings."""

    triangles_drawn: int
    triangles_culled: int
    segments_rendered: int
    segments_without_loads: int
    render_time_ms: float
    total_triangles: int = 0


@functools.lru_cache(maxsize=128)
def _get_cached_capsule(length: float, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Cache procedural capsule meshes by discrete dimensions."""
    cap = CapsuleShape(length=length, radius=radius, n_facets=16, n_lat=8)
    return cap.vertices_at_rest(), cap.faces()


def _resolve_mesh(
    mesh_id: str,
    scale: tuple[float, float, float],
    shape_library: ShapeLibrary | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve rest vertices and faces for a segment mesh."""
    if shape_library is not None and mesh_id in shape_library.names():
        shape = shape_library.get(mesh_id)
        v_rest = shape.vertices_at_rest() * np.asarray(scale, dtype=float)
        return v_rest, shape.faces()
    # Procedural capsule fallback
    length = max(float(scale[0]), 1e-4)
    radius = max(float(scale[1]), 1e-4)
    return _get_cached_capsule(round(length, 6), round(radius, 6))


def _hex_to_bgr(color_hex: str) -> tuple[int, int, int]:
    """Convert '#rrggbb' to BGR tuple."""
    hex_str = color_hex.lstrip("#")
    if len(hex_str) == 6:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return (b, g, r)
    return (180, 119, 31)


def _camera_space_points(projector: Any, points_world: np.ndarray) -> np.ndarray:
    """Project world points into camera space (N, 3)."""
    if hasattr(projector, "camera"):
        cam = projector.camera
        if hasattr(cam, "camera_from_world"):
            return cam.camera_from_world(points_world)
        if hasattr(cam, "extrinsics"):
            ext = cam.extrinsics
            r_wc = ext.rotation_world_from_camera
            t_wc = ext.translation_world_from_camera_m
            return (points_world - t_wc) @ r_wc
    return points_world


def _axis_to_rotation(axis: np.ndarray) -> np.ndarray:
    """Compute 3x3 rotation aligning local X with the given unit axis."""
    axis_unit = axis / float(np.linalg.norm(axis))
    z_dot = abs(float(axis_unit @ np.array([0.0, 0.0, 1.0])))
    world_up = (
        np.array([0.0, 1.0, 0.0])
        if z_dot > 1.0 - _NEAR_Z_TOL
        else np.array([0.0, 0.0, 1.0])
    )

    proj = float(axis_unit @ world_up) * axis_unit
    up_perp = world_up - proj
    up_unit = up_perp / float(np.linalg.norm(up_perp))
    side = np.cross(up_unit, axis_unit)

    rot = np.zeros((3, 3), dtype=float)
    rot[:, 0] = axis_unit
    rot[:, 1] = side
    rot[:, 2] = up_unit
    return rot


def segment_poses_from_axes(
    axes: Sequence[SegmentAxis], radius_m: float
) -> list[SegmentPose]:
    """Build capsule SegmentPose instances from proximal/distal endpoints.

    Parameters
    ----------
    axes : Sequence[SegmentAxis]
        Segment axes defining proximal and distal 3D world endpoints.
    radius_m : float
        Radius of each capsule segment in metres.

    Returns
    -------
    list[SegmentPose]
        SegmentPose instances aligned with proximal->distal along local X.
    """
    require(
        float(radius_m) > 0.0 and math.isfinite(float(radius_m)),
        "radius_m must be positive and finite",
    )
    poses: list[SegmentPose] = []
    for axis in axes:
        p = np.asarray(axis.proximal_m, dtype=float)
        d = np.asarray(axis.distal_m, dtype=float)
        delta = d - p
        length = float(np.linalg.norm(delta))
        require(
            length > 1e-9,
            f"Segment '{axis.segment}' has degenerate coincident endpoints",
        )

        rot = _axis_to_rotation(delta)
        T = np.eye(4, dtype=float)
        T[:3, :3] = rot
        T[:3, 3] = p

        poses.append(
            SegmentPose(
                name=axis.segment,
                mesh_id="capsule",
                T_world_segment=T,
                scale=(length, float(radius_m), float(radius_m)),
            )
        )
    return poses


def _extract_segment_triangles(
    pose: SegmentPose,
    projector: Any,
    shading: SegmentShading,
    loads: AxialLoadFrame | None,
    color_scale: ForceColorScale,
    shape_library: ShapeLibrary | None,
) -> tuple[list[tuple[float, np.ndarray, tuple[int, int, int]]], int, bool]:
    """Process a single segment and return its candidate screen triangles."""
    v_rest, faces = _resolve_mesh(pose.mesh_id, pose.scale, shape_library)
    r = pose.T_world_segment[:3, :3]
    t = pose.T_world_segment[:3, 3]
    v_world = v_rest @ r.T + t

    pixels, valid_mask = projector.project(v_world)
    v_cam = _camera_space_points(projector, v_world)

    # Determine base segment color
    has_load = False
    if loads is not None and pose.name in loads.values_n:
        load_val = loads.values_n[pose.name]
        color_hex = color_scale.color(load_val, pose.base_color)
        has_load = load_val is not None
    else:
        color_hex = pose.base_color
    base_bgr = _hex_to_bgr(color_hex)

    triangles: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
    culled = 0

    for face in faces:
        if not valid_mask[face].all():
            culled += 1
            continue

        p0, p1, p2 = pixels[face]
        # Back-face culling via 2D winding
        w = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if shading.backface_culling and w >= 0:
            culled += 1
            continue

        cam_xyz = v_cam[face]
        depth = float(cam_xyz[:, 2].mean())
        if depth <= 0.0:
            culled += 1
            continue

        # Lambertian shading: n_cam dot light_cam (light along camera view dir)
        n_cam = np.cross(cam_xyz[1] - cam_xyz[0], cam_xyz[2] - cam_xyz[0])
        n_norm = float(np.linalg.norm(n_cam))
        if n_norm <= 1e-12:
            culled += 1
            continue
        n_unit = n_cam / n_norm
        # Front-facing surface has normal pointing back towards camera (-Z_cam)
        n_dot_l = max(0.0, float(-n_unit[2]))
        intensity = shading.ambient + (1.0 - shading.ambient) * n_dot_l

        shaded_bgr = (
            int(np.clip(round(base_bgr[0] * intensity), 0, 255)),
            int(np.clip(round(base_bgr[1] * intensity), 0, 255)),
            int(np.clip(round(base_bgr[2] * intensity), 0, 255)),
        )
        triangles.append((depth, np.array([p0, p1, p2], dtype=float), shaded_bgr))

    return triangles, culled, has_load


def draw_segment_meshes_on_frame(
    frame_bgr: np.ndarray,
    segments: Sequence[SegmentPose],
    projector: Any,
    *,
    shading: SegmentShading | None = None,
    loads: AxialLoadFrame | None = None,
    color_scale: ForceColorScale | None = None,
    shape_library: ShapeLibrary | None = None,
) -> tuple[np.ndarray, SegmentDrawReceipt]:
    """Draw filled, shaded segment meshes onto a video frame.

    Parameters
    ----------
    frame_bgr : np.ndarray
        Input image array in BGR format (H, W, 3).
    segments : Sequence[SegmentPose]
        Segment poses to project and render.
    projector : Any
        Image projector (e.g. PinholeProjector).
    shading : SegmentShading, optional
        Visual shading options. Defaults to SegmentShading().
    loads : AxialLoadFrame, optional
        Measured axial reaction loads per segment.
    color_scale : ForceColorScale, optional
        Force-to-color mapping policy.
    shape_library : ShapeLibrary, optional
        Shape library instance; if None, loads default library lazily.

    Returns
    -------
    tuple[np.ndarray, SegmentDrawReceipt]
        (composited_frame_bgr, receipt)
    """
    t_start = time.perf_counter()
    opts = shading or SegmentShading()
    scale = color_scale or ForceColorScale(enabled=False)

    if shape_library is None:
        try:
            from src.shared.python.body_part_viz.asset_library import ShapeLibrary

            shape_library = ShapeLibrary.default()
        except Exception as exc:  # noqa: BLE001
            logger.debug("ShapeLibrary default could not be loaded: %s", exc)
            shape_library = None

    if opts.opacity <= 0.0 or len(segments) == 0:
        total_time_ms = (time.perf_counter() - t_start) * 1000.0
        return frame_bgr.copy(), SegmentDrawReceipt(
            triangles_drawn=0,
            triangles_culled=0,
            segments_rendered=0,
            segments_without_loads=len(segments),
            render_time_ms=total_time_ms,
        )

    all_triangles: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
    total_culled = 0
    rendered_segments = 0
    segments_without_loads = 0

    for pose in segments:
        triangles, culled, has_load = _extract_segment_triangles(
            pose, projector, opts, loads, scale, shape_library
        )
        total_culled += culled
        if not has_load:
            segments_without_loads += 1
        if triangles:
            rendered_segments += 1
            all_triangles.extend(triangles)

    # Budget cap
    if len(all_triangles) > opts.max_triangles:
        total_culled += len(all_triangles) - opts.max_triangles
        all_triangles = all_triangles[: opts.max_triangles]

    # Painter's algorithm: sort triangles far to near (descending depth)
    all_triangles.sort(key=lambda item: item[0], reverse=True)

    drawn = frame_bgr.copy()
    for _, poly, color in all_triangles:
        pts = np.rint(poly).astype(np.int32)
        cv2.fillConvexPoly(drawn, pts, color, lineType=cv2.LINE_AA)

    if opts.opacity >= 1.0:
        result = drawn
    else:
        result = cv2.addWeighted(
            drawn, opts.opacity, frame_bgr, 1.0 - opts.opacity, 0.0
        )

    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    receipt = SegmentDrawReceipt(
        triangles_drawn=len(all_triangles),
        triangles_culled=total_culled,
        segments_rendered=rendered_segments,
        segments_without_loads=segments_without_loads,
        render_time_ms=total_time_ms,
        total_triangles=len(all_triangles) + total_culled,
    )
    return result, receipt
