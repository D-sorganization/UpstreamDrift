"""Calibrated scene registration, event synchronization and distortion projection (#9865).

Enables registering external reference trajectories (e.g. ReferenceMotion) or 2D expert
reference videos (ReferenceVideo) into a multi-camera scene coordinate system (ADR-0041).
Strictly preserves:
- Explicit rigid/similarity transforms (R, t, scale) and body-size normalization.
- Coordinate frame conversion (canonical z_up_right_handed into ADR-0041 camera world).
- Event-anchor and offset synchronization with bounded monotonic time warping.
- Missing-joint and gap masking across interpolation (None never interpolated).
- Pinhole camera projection with optional lens distortion and frustum clipping.
- 2D video homography/affine alignment without false 3D claims.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Literal, Self
from uuid import UUID

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reference.model import Asset, ReferenceMotion
from src.shared.python.core.contracts import check_finite, require
from src.shared.python.estimation.residuals import project_pinhole
from src.shared.python.pose_estimation.observations import CameraCalibration

from .synchronization import EventAnchors as EventAnchors, TimeMapping as TimeMapping
from .evidence import CameraSnapshot, ViewClock, asset_identity

Matrix3x3 = tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]
Vector3 = tuple[float, float, float]
_ROTATION_TOL = 1.0e-5


def canonical_z_up_to_adr0041_world(
    points: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert points from canonical z_up_right_handed to ADR-0041 world frame.

    Canonical z-up right-handed:
      X = toward target (forward)
      Y = golfer's left
      Z = up
    ADR-0041 world frame:
      X = toward target
      Y = up
      Z = golfer's right

    Transformation:
      x_world = x_can
      y_world = z_can
      z_world = -y_can
    """
    pts = np.asarray(points, dtype=float)
    require(pts.shape[-1] == 3, "points must have 3 coordinates in the last dimension")
    converted = np.empty_like(pts)
    converted[..., 0] = pts[..., 0]
    converted[..., 1] = pts[..., 2]
    converted[..., 2] = -pts[..., 1]
    return converted


class ReferenceTransform(BaseModel):
    """Rigid or similarity 3D transform mapping reference coordinates to scene space."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    rotation: Matrix3x3 = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    translation_m: Vector3 = (0.0, 0.0, 0.0)
    scale: float = Field(default=1.0, gt=0.0)
    body_size_normalized: bool = False
    is_calibrated: bool = False

    @model_validator(mode="after")
    def validate_rotation(self) -> Self:
        r = np.asarray(self.rotation, dtype=float)
        ortho = r.T @ r
        if not np.allclose(ortho, np.eye(3), atol=_ROTATION_TOL):
            raise ValueError("Rotation matrix must be orthonormal")
        det = float(np.linalg.det(r))
        if not np.isclose(det, 1.0, atol=_ROTATION_TOL):
            raise ValueError("Rotation matrix determinant must be +1 (no reflection)")
        return self

    def apply(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Apply s * R @ p + t to an array of points of shape (..., 3)."""
        pts = np.asarray(points, dtype=float)
        require(pts.shape[-1] == 3, "points must have last dimension 3")
        r = np.asarray(self.rotation, dtype=float)
        t = np.asarray(self.translation_m, dtype=float)
        # pts @ R.T is equivalent to (R @ p) for row vectors
        scaled_rotated = self.scale * (pts @ r.T)
        return scaled_rotated + t


class ReferenceRegistration(BaseModel):
    """Documented registration of a reference asset into a camera rig scene."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal["reference-registration/1.0.0"] = (
        "reference-registration/1.0.0"
    )
    reference_id: str
    calibration_id: str
    transform: ReferenceTransform = Field(default_factory=ReferenceTransform)
    time_mapping: TimeMapping = Field(default_factory=TimeMapping)
    coordinate_convention: Literal["z_up_right_handed_to_adr0041"] = (
        "z_up_right_handed_to_adr0041"
    )
    image_transform_2d: Matrix3x3 | None = None
    assumption_labels: tuple[str, ...] = ()
    is_calibrated: bool = False
    max_gap_s: float = Field(default=0.25, gt=0, le=10)
    asset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    camera: CameraSnapshot | None = None
    clock: ViewClock | None = None

    @model_validator(mode="after")
    def validate_registration(self) -> Self:
        if str(UUID(self.reference_id)) != self.reference_id:
            # Validate UUID syntax or raise
            raise ValueError("reference_id must be a valid canonical UUID")
        if not self.calibration_id.strip():
            raise ValueError("calibration_id must be non-empty")
        if self.camera and self.clock and self.camera.camera_id != self.clock.view:
            raise ValueError("Camera and clock must describe the same view")
        return self

    def validate_binding(
        self, asset: Asset, camera: CameraSnapshot | None, clock: ViewClock
    ) -> None:
        """Check once when opening/saving/exporting, outside the per-frame sampler."""
        if asset.id != self.reference_id or self.asset_sha256 != asset_identity(asset):
            raise ValueError("Reference geometry changed; review its alignment")
        if (self.camera.identity if self.camera else None) != (
            camera.identity if camera else None
        ):
            raise ValueError("Reference camera changed; review its alignment")
        if self.clock != clock:
            raise ValueError("Reference clock changed; review its event alignment")

    def bound(
        self, asset: Asset, camera: CameraSnapshot | None, clock: ViewClock
    ) -> Self:
        """Snapshot the manual reference recipe; camera presence is not registration accuracy."""
        if asset.id != self.reference_id:
            raise ValueError("Registration belongs to a different reference")
        if self.asset_sha256 is not None:
            self.validate_binding(asset, camera, clock)
        return type(self).model_validate(
            self.model_dump()
            | {
                "asset_sha256": asset_identity(asset),
                "camera": self.camera or camera,
                "clock": clock,
                "calibration_id": camera.identity if camera else "unavailable",
                "is_calibrated": False,
            }
        )

    def scene_time(self, original_time: float) -> float:
        return self.clock.player_time(original_time) if self.clock else original_time


def transform_reference_motion(
    motion: ReferenceMotion, registration: ReferenceRegistration
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Transform ReferenceMotion points into scene world coordinates.

    Returns:
      (scene_times (T,), points_world (T, K, 3), valid_mask (T, K))
    """
    t_count = len(motion.time_s)
    k_count = len(motion.joint_names)

    scene_times = np.asarray(
        [registration.time_mapping.reference_to_scene(t) for t in motion.time_s],
        dtype=float,
    )
    raw_pts = np.zeros((t_count, k_count, 3), dtype=float)
    valid_mask = np.zeros((t_count, k_count), dtype=bool)

    for i, row in enumerate(motion.points_m):
        for j, pt in enumerate(row):
            if pt is not None and check_finite(np.asarray(pt)):
                raw_pts[i, j] = pt
                valid_mask[i, j] = True

    # 1. Convert coordinate convention: canonical z-up to ADR-0041 world
    conv_pts = canonical_z_up_to_adr0041_world(raw_pts)
    # 2. Apply registration transform: s * R @ p + t
    transformed_pts = registration.transform.apply(conv_pts)
    # Zero out invalid entries
    transformed_pts[~valid_mask] = 0.0

    return scene_times, transformed_pts, valid_mask


def sample_reference_motion(
    motion: ReferenceMotion,
    registration: ReferenceRegistration,
    scene_times: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Sample reference motion at requested scene times using bounded linear interpolation.

    Strictly propagates missing-joint masks: if either bounding frame has a missing point
    for joint k, the interpolated point at time t is masked as invalid (valid_mask=False).
    """
    require(
        registration.reference_id == motion.id,
        "Registration belongs to a different reference",
    )
    t_eval = np.asarray(scene_times, dtype=float)
    require(t_eval.ndim == 1, "scene_times must be one-dimensional")
    reference_times = registration.time_mapping.scene_to_reference(t_eval)
    k_count = len(motion.joint_names)
    out_pts = np.zeros((len(t_eval), k_count, 3), dtype=float)
    out_valid = np.zeros((len(t_eval), k_count), dtype=bool)
    for sample, time in enumerate(reference_times):
        right = bisect_left(motion.time_s, time)
        exact = next(
            (
                i
                for i in (right, right - 1)
                if 0 <= i < len(motion.time_s) and abs(motion.time_s[i] - time) <= 1e-9
            ),
            None,
        )
        if exact is not None:
            left, right, alpha = exact, exact, 0.0
        elif right == 0 or right == len(motion.time_s):
            continue
        else:
            left = right - 1
            gap = motion.time_s[right] - motion.time_s[left]
            if gap > registration.max_gap_s:
                continue
            alpha = (time - motion.time_s[left]) / gap
        for joint, (a, b) in enumerate(
            zip(motion.points_m[left], motion.points_m[right], strict=True)
        ):
            if a is not None and b is not None:
                out_pts[sample, joint] = (1 - alpha) * np.asarray(
                    a
                ) + alpha * np.asarray(b)
                out_valid[sample, joint] = True
    converted = canonical_z_up_to_adr0041_world(out_pts)
    transformed = registration.transform.apply(converted)
    transformed[~out_valid] = 0
    return transformed, out_valid


def project_reference_to_camera(
    points_world: npt.NDArray[np.float64],
    valid_mask: npt.NDArray[np.bool_],
    camera: PinholeCamera | CameraCalibration,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Project 3D world points onto camera image coordinates with clipping and distortion.

    Accepts points of shape (..., 3) and matching boolean valid_mask (...).
    Returns (projected_px (..., 2), visible_mask (...)).
    Points with depth <= 0 or outside image bounds are clipped to visible=False.
    """
    pts = np.asarray(points_world, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    shape_prefix = pts.shape[:-1]
    flat_pts = pts.reshape(-1, 3)
    flat_mask = mask.reshape(-1)

    if isinstance(camera, PinholeCamera):
        matrix = camera.matrix
        r_w2c = camera.rotation_world_from_camera.T
        t_w2c = -r_w2c @ camera.translation_world_from_camera_m
        distortion = None
        w, h = camera.image_size_px
    else:
        matrix = camera.intrinsics.matrix
        distortion = camera.intrinsics.distortion
        r_w2c = camera.extrinsics.rotation_world_from_camera.T
        t_w2c = -r_w2c @ camera.extrinsics.translation_world_from_camera_m
        w, h = camera.image_size_px

    # Calculate camera coordinates to check depth (z_c > 0)
    pts_cam = flat_pts @ r_w2c.T + t_w2c
    in_front = pts_cam[:, 2] > 1.0e-5

    # Safe projection via project_pinhole
    projected = project_pinhole(
        flat_pts,
        matrix,
        rotation_world_to_camera=r_w2c,
        translation_world_to_camera=t_w2c,
        distortion=distortion,
    )

    in_image = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < w)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < h)
    )

    final_visible = flat_mask & in_front & in_image
    projected[~final_visible] = np.nan

    return projected.reshape(*shape_prefix, 2), final_visible.reshape(*shape_prefix)
