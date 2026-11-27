"""Camera perspective matching and bidirectional ray/frustum projections (Epic #15660, #15669).

Binds pinhole CameraProjection (from #11246) to 3D virtual viewport parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np

from src.shared.python.motion_matching.historical_fit.contracts import CameraProjection

_TOL = 1e-8


def _vec(v: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Return ``name`` as a length-3 finite vector."""
    arr = np.asarray(v, dtype=float).ravel()
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain 3 finite numbers")
    return arr


def _unit(v: np.ndarray, name: str) -> np.ndarray:
    """Return the unit vector of ``name``, rejecting a near-zero magnitude."""
    norm = float(np.linalg.norm(v))
    if norm < _TOL:
        raise ValueError(f"{name} must have non-zero magnitude")
    return v / norm


@dataclass(frozen=True)
class CameraViewParameters:
    """3D viewport camera perspective matching a 2D source video projection."""

    position_m: tuple[float, float, float]
    target_m: tuple[float, float, float]
    up_vector: tuple[float, float, float]
    fov_horizontal_deg: float
    fov_vertical_deg: float
    aspect_ratio: float
    near_clip_m: float
    far_clip_m: float
    view_matrix: np.ndarray
    projection_matrix: np.ndarray

    def __post_init__(self) -> None:
        """Reject field of view, clip, aspect, or matrix values that cannot form a view."""
        if not (
            0.0 < self.fov_horizontal_deg < 180.0
            and 0.0 < self.fov_vertical_deg < 180.0
        ):
            raise ValueError("FOVs must be in (0, 180)")
        if (
            self.aspect_ratio <= 0.0
            or self.near_clip_m <= 0.0
            or self.far_clip_m <= self.near_clip_m
        ):
            raise ValueError("Invalid aspect ratio or clipping planes")
        if self.view_matrix.shape != (4, 4) or self.projection_matrix.shape != (4, 4):
            raise ValueError("view_matrix and projection_matrix must be 4x4")


def match_perspective_from_projection(
    projection: CameraProjection,
    image_size: tuple[int, int],
    lookat_distance_m: float = 2.0,
    near_clip_m: float = 0.01,
    far_clip_m: float = 100.0,
) -> CameraViewParameters:
    """Convert pinhole CameraProjection to 3D viewport parameters and matrices."""
    w, h = image_size
    if (
        w <= 0
        or h <= 0
        or lookat_distance_m <= 0.0
        or not math.isfinite(lookat_distance_m)
    ):
        raise ValueError(
            "Image dimensions and lookat_distance_m must be positive and finite"
        )
    if near_clip_m <= 0.0 or far_clip_m <= near_clip_m:
        raise ValueError("Clipping planes require 0 < near < far")

    r_cw, t_cw = projection.rotation, projection.translation
    r_wc = r_cw.T
    pos = -r_wc @ t_cw
    target = pos + lookat_distance_m * r_wc[:, 2]

    fx, fy = float(projection.intrinsics[0, 0]), float(projection.intrinsics[1, 1])
    fov_h = math.degrees(2.0 * math.atan((w / 2.0) / fx))
    fov_v = math.degrees(2.0 * math.atan((h / 2.0) / fy))
    aspect = float(w / h)

    view_mat = np.eye(4, dtype=float)
    view_mat[:3, :3], view_mat[:3, 3] = r_cw, t_cw

    proj_mat = np.zeros((4, 4), dtype=float)
    tan_half_v = math.tan(math.radians(fov_v) / 2.0)
    proj_mat[0, 0] = 1.0 / (aspect * tan_half_v)
    proj_mat[1, 1] = 1.0 / tan_half_v
    proj_mat[2, 2] = -(far_clip_m + near_clip_m) / (far_clip_m - near_clip_m)
    proj_mat[2, 3] = -(2.0 * far_clip_m * near_clip_m) / (far_clip_m - near_clip_m)
    proj_mat[3, 2] = -1.0

    return CameraViewParameters(
        position_m=(float(pos[0]), float(pos[1]), float(pos[2])),
        target_m=(float(target[0]), float(target[1]), float(target[2])),
        up_vector=(float(-r_wc[0, 1]), float(-r_wc[1, 1]), float(-r_wc[2, 1])),
        fov_horizontal_deg=fov_h,
        fov_vertical_deg=fov_v,
        aspect_ratio=aspect,
        near_clip_m=near_clip_m,
        far_clip_m=far_clip_m,
        view_matrix=view_mat,
        projection_matrix=proj_mat,
    )


def projection_from_viewer_perspective(
    position_m: Sequence[float] | np.ndarray,
    target_m: Sequence[float] | np.ndarray,
    up_m: Sequence[float] | np.ndarray,
    image_size: tuple[int, int],
    fov_vertical_deg: float,
) -> CameraProjection:
    """Derive CameraProjection from 3D viewer perspective parameters."""
    w, h = image_size
    if w <= 0 or h <= 0 or not (0.0 < fov_vertical_deg < 180.0):
        raise ValueError("Invalid dimensions or fov_vertical_deg")

    pos, tgt, up = _vec(position_m, "pos"), _vec(target_m, "tgt"), _vec(up_m, "up")
    fwd = _unit(tgt - pos, "view direction")
    rgt = _unit(np.cross(fwd, up), "camera right")
    dwn = _unit(np.cross(fwd, rgt), "camera down")
    r_cw = np.column_stack([rgt, dwn, fwd]).T

    fy = (h / 2.0) / math.tan(math.radians(fov_vertical_deg) / 2.0)
    k = np.array([[fy, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]], dtype=float)
    return CameraProjection(intrinsics=k, rotation=r_cw, translation=-r_cw @ pos)


def unproject_pixel_to_ray(
    projection: CameraProjection, pixel_xy: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a 2D pixel to a 3D ray (origin, unit_direction) in world coordinates."""
    u, v = pixel_xy
    if not (math.isfinite(u) and math.isfinite(v)):
        raise ValueError("Pixel coordinates must be finite")
    k, r_wc = projection.intrinsics, projection.rotation.T
    dir_c = np.array(
        [(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], 1.0], dtype=float
    )
    return -r_wc @ projection.translation, _unit(r_wc @ dir_c, "ray")


def unproject_pixel_to_plane(
    projection: CameraProjection,
    pixel_xy: tuple[float, float],
    plane_normal: Sequence[float] | np.ndarray,
    plane_point: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Intersect an unprojected ray with a 3D plane in world coordinates."""
    origin, ray_dir = unproject_pixel_to_ray(projection, pixel_xy)
    n = _unit(_vec(plane_normal, "plane_normal"), "plane_normal")
    denom = float(np.dot(n, ray_dir))
    if abs(denom) < _TOL:
        raise ValueError("Ray is parallel to the target plane")
    s = float(np.dot(_vec(plane_point, "plane_point") - origin, n)) / denom
    if s <= 0.0:
        raise ValueError("Plane intersection is behind or at the camera")
    return origin + s * ray_dir


def frustum_quad_world_m(
    projection: CameraProjection, image_size: tuple[int, int], distance_m: float
) -> np.ndarray:
    """Compute the 4 corners of the video frame in 3D world space at a given depth."""
    w, h = image_size
    if w <= 0 or h <= 0 or distance_m <= 0.0 or not math.isfinite(distance_m):
        raise ValueError("Dimensions and distance_m must be positive and finite")
    k, r_wc = projection.intrinsics, projection.rotation.T
    orig = -r_wc @ projection.translation
    corners = [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]
    pts = [
        r_wc
        @ np.array(
            [
                (u - k[0, 2]) * distance_m / k[0, 0],
                (v - k[1, 2]) * distance_m / k[1, 1],
                distance_m,
            ]
        )
        + orig
        for u, v in corners
    ]
    return np.array(pts, dtype=float)
