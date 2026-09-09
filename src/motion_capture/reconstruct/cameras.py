"""Pinhole cameras in the ADR-0041 world frame.

World frame: right-handed SI metres, x toward the target, y up, z to the
golfer's right. A camera is stored as ``T_world_from_camera`` (rotation and
translation), exactly as :class:`~pose_estimation.observations.CameraExtrinsics`
records it, so a synthetic rig and a calibrated real rig are the same object.
Projection is the textbook ``x = K [R | t]_camera_from_world X`` with no
distortion; distortion belongs to the intrinsic calibration stage (#9622) and
is applied by undistorting observations before they reach this model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require
from src.shared.python.pose_estimation.observations import (
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
)

Array: TypeAlias = npt.NDArray[np.float64]
Mask: TypeAlias = npt.NDArray[np.bool_]
_TOL = 1e-6


def _unit(v: Array, name: str) -> Array:
    norm = float(np.linalg.norm(v))
    require(norm > _TOL, f"{name} must be non-zero")
    return v / norm


@dataclass(frozen=True)
class PinholeCamera:
    """One camera: intrinsics ``K`` and pose ``T_world_from_camera``."""

    camera_id: str
    matrix: Array  # 3x3 intrinsics
    rotation_world_from_camera: Array  # 3x3
    translation_world_from_camera_m: Array  # (3,)
    image_size_px: tuple[int, int]

    def __post_init__(self) -> None:
        k = np.asarray(self.matrix, dtype=float)
        r = np.asarray(self.rotation_world_from_camera, dtype=float)
        t = np.asarray(self.translation_world_from_camera_m, dtype=float).reshape(3)
        require(k.shape == (3, 3) and k[0, 0] > 0 and k[1, 1] > 0, "invalid K")
        require(r.shape == (3, 3), "rotation must be 3x3")
        require(np.allclose(r.T @ r, np.eye(3), atol=1e-6), "rotation not orthonormal")
        require(np.isclose(np.linalg.det(r), 1.0, atol=1e-6), "rotation det must be +1")
        require(len(self.image_size_px) == 2, "image_size_px must be (w, h)")
        object.__setattr__(self, "matrix", k)
        object.__setattr__(self, "rotation_world_from_camera", r)
        object.__setattr__(self, "translation_world_from_camera_m", t)

    @property
    def position_m(self) -> Array:
        """Camera centre in world coordinates."""
        return self.translation_world_from_camera_m

    def camera_from_world(self, points_world: Array) -> Array:
        """``(N, 3)`` world points expressed in the camera frame."""
        p = np.asarray(points_world, dtype=float).reshape(-1, 3)
        r_cw = self.rotation_world_from_camera.T
        return (p - self.translation_world_from_camera_m) @ r_cw.T

    def project(self, points_world: Array) -> tuple[Array, Mask]:
        """Pixel coordinates ``(N, 2)`` and a mask of points in front of the camera.

        Points behind the camera (depth <= 0) get NaN pixels and ``False``.
        Postcondition: shapes are ``(N, 2)`` and ``(N,)``.
        """
        pc = self.camera_from_world(points_world)
        depth = pc[:, 2]
        in_front: Mask = depth > _TOL
        px: Array = np.full((pc.shape[0], 2), np.nan)
        if in_front.any():
            hom = pc[in_front] / depth[in_front, None]
            uv = hom @ self.matrix.T
            px[in_front] = uv[:, :2]
        return px, in_front

    def in_image(self, px: Array) -> Mask:
        """Mask of pixel coordinates inside the image bounds (NaN is outside)."""
        p = np.asarray(px, dtype=float).reshape(-1, 2)
        w, h = self.image_size_px
        with np.errstate(invalid="ignore"):
            inside: Mask = (
                (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
            )
        return inside

    def to_calibration(self) -> CameraCalibration:
        """The ADR-0041 record for this camera (no distortion)."""
        return CameraCalibration(
            camera_id=self.camera_id,
            intrinsics=CameraIntrinsics(matrix=self.matrix),
            extrinsics=CameraExtrinsics(
                rotation_world_from_camera=self.rotation_world_from_camera,
                translation_world_from_camera_m=self.translation_world_from_camera_m,
            ),
            image_size_px=self.image_size_px,
        )

    @classmethod
    def from_calibration(cls, record: CameraCalibration) -> PinholeCamera:
        intrinsics = record.intrinsics
        extrinsics = record.extrinsics
        return cls(
            camera_id=record.camera_id,
            matrix=np.asarray(intrinsics.matrix, dtype=float),
            rotation_world_from_camera=np.asarray(
                extrinsics.rotation_world_from_camera, dtype=float
            ),
            translation_world_from_camera_m=np.asarray(
                extrinsics.translation_world_from_camera_m, dtype=float
            ),
            image_size_px=record.image_size_px,
        )


def intrinsics_from_fov(
    width_px: int, height_px: int, horizontal_fov_deg: float
) -> Array:
    """Square-pixel ``K`` for an image of the given size and horizontal field of view."""
    require(width_px > 0 and height_px > 0, "image size must be positive")
    require(0.0 < horizontal_fov_deg < 180.0, "fov must be in (0, 180) degrees")
    f = (width_px / 2.0) / np.tan(np.radians(horizontal_fov_deg) / 2.0)
    return np.array([[f, 0.0, width_px / 2.0], [0.0, f, height_px / 2.0], [0, 0, 1.0]])


def look_at(position_m: Array, target_m: Array, up: Array | None = None) -> Array:
    """``R_world_from_camera`` for a camera at ``position`` looking at ``target``.

    Camera convention: +z forward (optical axis), +x right, +y down (OpenCV),
    so the world ``up`` maps to camera ``-y``. Precondition: position and
    target differ and the view direction is not parallel to ``up``.
    """
    pos = np.asarray(position_m, dtype=float).reshape(3)
    tgt = np.asarray(target_m, dtype=float).reshape(3)
    up_w = np.array([0.0, 1.0, 0.0]) if up is None else np.asarray(up, dtype=float)
    forward = _unit(tgt - pos, "view direction")
    right = np.cross(forward, up_w)
    require(bool(np.linalg.norm(right) > _TOL), "view direction parallel to up")
    right = _unit(right, "right")
    down = np.cross(forward, right)
    return np.column_stack([right, down, forward])
