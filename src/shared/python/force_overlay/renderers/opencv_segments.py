"""OpenCV video segment mesh renderer (FTO-26, #11311).

Draws filled, shaded segment volumes (e.g. from ShapeLibrary or capsule fallback)
onto video frames, coloured by tension/compression loads from AxialLoadFrame.
Follows painter's algorithm depth sorting and Lambert shading.
"""

from __future__ import annotations

import functools
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import numpy.typing as npt

from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.shared.python.body_part_viz.shapes import CapsuleShape
from src.shared.python.force_overlay.conversions import SegmentAxis

if TYPE_CHECKING:
    from src.motion_capture.reconstruct.cameras import PinholeCamera
    from src.shared.python.body_part_viz.asset_library import ShapeLibrary
    from src.shared.python.force_overlay.renderers.opencv_glyphs import ImageProjector
    from src.shared.python.pose_estimation.observations import CameraCalibration

__all__ = [
    "MAX_TRIANGLES_PER_FRAME",
    "SegmentDrawReceipt",
    "SegmentPose",
    "SegmentShading",
    "draw_segment_meshes_on_frame",
    "segment_poses_from_axes",
]

MAX_TRIANGLES_PER_FRAME: int = 20_000


@dataclass(frozen=True)
class SegmentShading:
    """Visual shading parameters for segment mesh rendering."""

    ambient: float = 0.35
    opacity: float = 0.55

    def __post_init__(self) -> None:
        if not np.isfinite(self.ambient) or not (0.0 <= self.ambient <= 1.0):
            raise ValueError(f"ambient must be in [0, 1]; got {self.ambient}")
        if not np.isfinite(self.opacity) or not (0.0 <= self.opacity <= 1.0):
            raise ValueError(f"opacity must be in [0, 1]; got {self.opacity}")


@dataclass(frozen=True)
class SegmentPose:
    """Pose, mesh id, transform, and scale for a single body segment."""

    name: str
    mesh_id: str = "capsule"
    T_world_segment: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    base_color: str = "#808080"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.mesh_id, str) or not self.mesh_id.strip():
            raise ValueError("mesh_id must be a non-empty string")

        T = np.asarray(self.T_world_segment, dtype=np.float64)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ValueError(f"T_world_segment must be finite (4, 4); got {T.shape}")
        R = T[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
            raise ValueError("T_world_segment rotation must be orthonormal within 1e-5")
        if not np.isclose(float(np.linalg.det(R)), 1.0, atol=1e-5):
            raise ValueError("T_world_segment rotation determinant must be +1")
        if not np.allclose(T[3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise ValueError("T_world_segment last row must be [0, 0, 0, 1]")
        object.__setattr__(
            self, "T_world_segment", np.ascontiguousarray(T, dtype=np.float64)
        )

        s = tuple(float(x) for x in self.scale)
        if len(s) != 3 or any(not np.isfinite(x) or x <= 0.0 for x in s):
            raise ValueError(f"scale must have 3 positive finite elements; got {s}")
        object.__setattr__(self, "scale", s)

        if not isinstance(self.base_color, str) or not re.fullmatch(
            r"#[0-9a-fA-F]{6}", self.base_color
        ):
            raise ValueError("base_color must be an opaque #RRGGBB color")
        object.__setattr__(self, "base_color", self.base_color.lower())


@dataclass(frozen=True)
class SegmentDrawReceipt:
    """Execution receipt of segment mesh rendering."""

    triangles_drawn: int
    triangles_culled: int
    segments_without_loads: int
    render_time_ms: float
    frame: np.ndarray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "triangles_drawn": self.triangles_drawn,
            "triangles_culled": self.triangles_culled,
            "segments_without_loads": self.segments_without_loads,
            "render_time_ms": self.render_time_ms,
        }

    def __iter__(self) -> Any:
        yield self.frame
        yield self


@functools.lru_cache(maxsize=128)
def _get_capsule_mesh(
    length_round: float, radius_round: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build and cache capsule vertices and outward-facing faces."""
    cap = CapsuleShape(length=length_round, radius=radius_round, n_facets=16, n_lat=8)
    faces = cap.faces()[:, [0, 2, 1]]
    return cap.vertices_at_rest(), faces


def _load_mesh_for_pose(
    pose: SegmentPose, shape_library: ShapeLibrary | None
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve vertices and faces for a pose, applying scale to vertices."""
    if shape_library is not None and pose.mesh_id != "capsule":
        try:
            shape = shape_library.get(pose.mesh_id)
            rest_v = shape.vertices_at_rest()
            faces = shape.faces()
            scale = np.asarray(pose.scale, dtype=np.float64)
            return rest_v * scale, faces
        except (KeyError, FileNotFoundError, AttributeError):
            pass

    length, radius = pose.scale[0], pose.scale[1]
    return _get_capsule_mesh(round(length, 6), round(radius, 6))


def _camera_extrinsics(projector: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract (R_world_from_camera, pos_world) from projector."""
    camera = getattr(projector, "camera", None)
    if camera is not None:
        r_w_c = getattr(camera, "rotation_world_from_camera", None)
        t_w_c = getattr(camera, "translation_world_from_camera_m", None)
        if r_w_c is not None and t_w_c is not None:
            return np.asarray(r_w_c, dtype=float), np.asarray(t_w_c, dtype=float)
        extrinsics = getattr(camera, "extrinsics", None)
        if extrinsics is not None:
            return (
                np.asarray(extrinsics.rotation_world_from_camera, dtype=float),
                np.asarray(extrinsics.translation_world_from_camera_m, dtype=float),
            )
    projection = getattr(projector, "projection", None)
    if projection is not None:
        rot = getattr(projection, "rotation", None)
        trans = getattr(projection, "translation", None)
        if rot is not None and trans is not None:
            r_w2c = np.asarray(rot, dtype=float)
            t_w2c = np.asarray(trans, dtype=float)
            r_c2w = r_w2c.T
            pos_w = -r_c2w @ t_w2c
            return r_c2w, pos_w
    raise ValueError("Cannot resolve camera extrinsics from projector")


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert hex string '#RRGGBB' to BGR integer tuple."""
    clean = hex_color.lstrip("#")
    r, g, b = (int(clean[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def _resolve_segment_color(
    seg: SegmentPose,
    loads: AxialLoadFrame | None,
    color_scale: ForceColorScale | None,
) -> tuple[tuple[int, int, int], bool]:
    """Return ((B, G, R), had_load)."""
    if loads is not None and seg.name in loads.values_n:
        f_val = loads.values_n[seg.name]
        if f_val is not None and math.isfinite(float(f_val)):
            if color_scale is not None and color_scale.enabled:
                hex_c = color_scale.color(float(f_val), seg.base_color)
            else:
                hex_c = seg.base_color
            return _hex_to_bgr(hex_c), True
    return _hex_to_bgr(seg.base_color), False


def segment_poses_from_axes(
    axes: Sequence[SegmentAxis], radius_m: float
) -> list[SegmentPose]:
    """Build capsule SegmentPose instances from proximal/distal endpoints."""
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError(f"radius_m must be finite and > 0; got {radius_m}")

    poses: list[SegmentPose] = []
    for axis in axes:
        p = np.asarray(axis.proximal_m, dtype=np.float64)
        d = np.asarray(axis.distal_m, dtype=np.float64)
        delta = d - p
        length = float(np.linalg.norm(delta))
        if length <= 1e-12:
            raise ValueError("SegmentAxis has degenerate zero length")

        u_x = delta / length
        v_tmp = (
            np.array([0.0, 1.0, 0.0])
            if abs(u_x[0]) >= 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        u_y = np.cross(u_x, v_tmp)
        u_y /= np.linalg.norm(u_y)
        u_z = np.cross(u_x, u_y)

        r_mat = np.column_stack([u_x, u_y, u_z])
        t_mat = np.eye(4, dtype=np.float64)
        t_mat[:3, :3] = r_mat
        t_mat[:3, 3] = p

        poses.append(
            SegmentPose(
                name=axis.segment,
                mesh_id="capsule",
                T_world_segment=t_mat,
                scale=(length, float(radius_m), float(radius_m)),
            )
        )
    return poses


def _collect_segment_triangles(
    seg: SegmentPose,
    projector: ImageProjector,
    rot_c2w: np.ndarray,
    pos_w: np.ndarray,
    shading: SegmentShading,
    base_bgr: tuple[int, int, int],
    shape_library: ShapeLibrary | None,
) -> tuple[list[tuple[float, np.ndarray, tuple[int, int, int]]], int]:
    """Project and shade visible triangles for one segment."""
    local_verts, faces = _load_mesh_for_pose(seg, shape_library)
    r_seg = seg.T_world_segment[:3, :3]
    t_seg = seg.T_world_segment[:3, 3]
    world_verts = local_verts @ r_seg.T + t_seg
    cam_verts = (world_verts - pos_w) @ rot_c2w

    pixels, valid = projector.project(world_verts)
    triangles: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
    culled = 0

    for face in faces:
        if not (valid[face[0]] and valid[face[1]] and valid[face[2]]):
            culled += 1
            continue
        xyz = cam_verts[face]
        if np.any(xyz[:, 2] <= 1e-4):
            culled += 1
            continue

        p0, p1, p2 = pixels[face[0]], pixels[face[1]], pixels[face[2]]
        cross2d = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if cross2d >= 0:
            culled += 1
            continue

        normal = np.cross(xyz[1] - xyz[0], xyz[2] - xyz[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            culled += 1
            continue

        normal_unit = normal / norm
        cos_theta = max(0.0, -float(normal_unit[2]))
        shade = shading.ambient + (1.0 - shading.ambient) * cos_theta
        tri_bgr = (
            int(round(base_bgr[0] * shade)),
            int(round(base_bgr[1] * shade)),
            int(round(base_bgr[2] * shade)),
        )
        mean_depth = float(xyz[:, 2].mean())
        triangles.append((mean_depth, np.array([p0, p1, p2]), tri_bgr))

    return triangles, culled


def draw_segment_meshes_on_frame(
    frame_bgr: np.ndarray,
    segments: Sequence[SegmentPose],
    projector: ImageProjector,
    *,
    shading: SegmentShading | None = None,
    loads: AxialLoadFrame | None = None,
    color_scale: ForceColorScale | None = None,
    inplace: bool = False,
    shape_library: ShapeLibrary | None = None,
) -> SegmentDrawReceipt:
    """Project and draw shaded segment meshes onto a BGR video frame."""
    t_start = time.perf_counter()
    if not isinstance(frame_bgr, np.ndarray):
        raise TypeError("frame_bgr must be a numpy ndarray")
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be uint8 array with shape (H, W, 3)")

    shd = shading if shading is not None else SegmentShading()
    out = frame_bgr if inplace else frame_bgr.copy()
    if shd.opacity <= 0.0 or len(segments) == 0:
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return SegmentDrawReceipt(
            triangles_drawn=0,
            triangles_culled=0,
            segments_without_loads=len(segments),
            render_time_ms=elapsed_ms,
            frame=out,
        )

    rot_c2w, pos_w = _camera_extrinsics(projector)
    all_triangles: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
    total_culled = 0
    seg_without_loads = 0

    for seg in segments:
        base_bgr, had_load = _resolve_segment_color(seg, loads, color_scale)
        if not had_load:
            seg_without_loads += 1
        tris, culled = _collect_segment_triangles(
            seg, projector, rot_c2w, pos_w, shd, base_bgr, shape_library
        )
        all_triangles.extend(tris)
        total_culled += culled

    # Sort painter's order (far to near -> descending mean depth)
    all_triangles.sort(key=lambda item: item[0], reverse=True)
    if len(all_triangles) > MAX_TRIANGLES_PER_FRAME:
        all_triangles = all_triangles[:MAX_TRIANGLES_PER_FRAME]

    drawn_layer = out.copy()
    for _, tri_px, color in all_triangles:
        cv2.fillConvexPoly(
            drawn_layer, np.rint(tri_px).astype(np.int32), color, cv2.LINE_AA
        )

    cv2.addWeighted(drawn_layer, shd.opacity, out, 1.0 - shd.opacity, 0, dst=out)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return SegmentDrawReceipt(
        triangles_drawn=len(all_triangles),
        triangles_culled=total_culled,
        segments_without_loads=seg_without_loads,
        render_time_ms=elapsed_ms,
        frame=out,
    )
