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

from typing import Annotated, Any, Literal, Self
from uuid import UUID

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reference.model import ReferenceMotion
from src.shared.python.core.contracts import check_finite, require
from src.shared.python.estimation.residuals import project_pinhole
from src.shared.python.pose_estimation.observations import CameraCalibration

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


class EventAnchors(BaseModel):
    """Key swing events anchored in both reference time and scene time."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    reference: dict[str, float]
    scene: dict[str, float]

    @model_validator(mode="after")
    def validate_anchors(self) -> Self:
        common_keys = set(self.reference.keys()) & set(self.scene.keys())
        if not common_keys:
            raise ValueError("Event anchors must share at least one common event key")
        for key in common_keys:
            if not np.isfinite(self.reference[key]) or not np.isfinite(self.scene[key]):
                raise ValueError("Anchor timestamps must be finite numbers")
        return self


class TimeMapping(BaseModel):
    """Synchronizes reference timestamps into the scene timeline."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    offset_s: float = 0.0
    rate_scale: float = Field(default=1.0, gt=0.01, lt=100.0)
    event_anchors: EventAnchors | None = None

    def reference_to_scene(self, t_ref: float | npt.NDArray[np.float64]) -> Any:
        """Convert a reference timestamp (or array) to scene time."""
        arr = np.asarray(t_ref, dtype=float)
        anchors = self.event_anchors
        if anchors is not None:
            # Piecewise linear warping through common sorted events
            ref_dict = anchors.reference
            scene_dict = anchors.scene
            common = sorted(
                set(ref_dict.keys()) & set(scene_dict.keys()),
                key=lambda k: ref_dict[k],
            )
            if len(common) >= 2:
                r_times = [float(ref_dict[k]) for k in common]
                s_times = [float(scene_dict[k]) for k in common]
                # Bounded interpolation with linear extrapolation at endpoints
                interp_val = np.interp(arr, r_times, s_times)
                # Extrapolate linearly beyond bounds
                left_mask = arr < r_times[0]
                right_mask = arr > r_times[-1]
                warped = np.array(interp_val, dtype=float, copy=True)
                if np.any(left_mask):
                    slope_l = (s_times[1] - s_times[0]) / (r_times[1] - r_times[0])
                    warped[left_mask] = (
                        s_times[0] + (arr[left_mask] - r_times[0]) * slope_l
                    )
                if np.any(right_mask):
                    slope_r = (s_times[-1] - s_times[-2]) / (r_times[-1] - r_times[-2])
                    warped[right_mask] = (
                        s_times[-1] + (arr[right_mask] - r_times[-1]) * slope_r
                    )
                return float(warped) if arr.ndim == 0 else warped
        res = arr * self.rate_scale + self.offset_s
        return float(res) if arr.ndim == 0 else res

    def scene_to_reference(self, t_scene: float | npt.NDArray[np.float64]) -> Any:
        """Convert a scene timestamp (or array) back to reference time."""
        arr = np.asarray(t_scene, dtype=float)
        anchors = self.event_anchors
        if anchors is not None:
            ref_dict = anchors.reference
            scene_dict = anchors.scene
            common = sorted(
                set(ref_dict.keys()) & set(scene_dict.keys()),
                key=lambda k: scene_dict[k],
            )
            if len(common) >= 2:
                s_times = [float(scene_dict[k]) for k in common]
                r_times = [float(ref_dict[k]) for k in common]
                interp_val = np.interp(arr, s_times, r_times)
                left_mask = arr < s_times[0]
                right_mask = arr > s_times[-1]
                warped = np.array(interp_val, dtype=float, copy=True)
                if np.any(left_mask):
                    slope_l = (r_times[1] - r_times[0]) / (s_times[1] - s_times[0])
                    warped[left_mask] = (
                        r_times[0] + (arr[left_mask] - s_times[0]) * slope_l
                    )
                if np.any(right_mask):
                    slope_r = (r_times[-1] - r_times[-2]) / (s_times[-1] - s_times[-2])
                    warped[right_mask] = (
                        r_times[-1] + (arr[right_mask] - s_times[-1]) * slope_r
                    )
                return float(warped) if arr.ndim == 0 else warped
        res = (arr - self.offset_s) / self.rate_scale
        return float(res) if arr.ndim == 0 else res


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

    @model_validator(mode="after")
    def validate_registration(self) -> Self:
        if str(UUID(self.reference_id)) != self.reference_id:
            # Validate UUID syntax or raise
            raise ValueError("reference_id must be a valid canonical UUID")
        if not self.calibration_id.strip():
            raise ValueError("calibration_id must be non-empty")
        return self


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
    source_scene_times, pts_world, source_mask = transform_reference_motion(
        motion, registration
    )

    t_eval = np.asarray(scene_times, dtype=float)
    n_samples = len(t_eval)
    k_count = pts_world.shape[1]

    out_pts = np.zeros((n_samples, k_count, 3), dtype=float)
    out_valid = np.zeros((n_samples, k_count), dtype=bool)

    for sample_idx, t in enumerate(t_eval):
        # Find bracketing indices
        if t < source_scene_times[0] or t > source_scene_times[-1]:
            # Beyond reference bounds
            continue

        idx = np.searchsorted(source_scene_times, t)
        if idx == 0:
            if np.isclose(t, source_scene_times[0]):
                out_pts[sample_idx] = pts_world[0]
                out_valid[sample_idx] = source_mask[0]
            continue
        if idx == len(source_scene_times):
            if np.isclose(t, source_scene_times[-1]):
                out_pts[sample_idx] = pts_world[-1]
                out_valid[sample_idx] = source_mask[-1]
            continue

        # Check exact hit
        t_prev = source_scene_times[idx - 1]
        t_next = source_scene_times[idx]
        if np.isclose(t, t_prev):
            out_pts[sample_idx] = pts_world[idx - 1]
            out_valid[sample_idx] = source_mask[idx - 1]
            continue
        if np.isclose(t, t_next):
            out_pts[sample_idx] = pts_world[idx]
            out_valid[sample_idx] = source_mask[idx]
            continue

        alpha = (t - t_prev) / (t_next - t_prev)
        # Joint is valid ONLY if BOTH endpoints are valid
        both_valid = source_mask[idx - 1] & source_mask[idx]
        out_valid[sample_idx] = both_valid
        out_pts[sample_idx, both_valid] = (1.0 - alpha) * pts_world[
            idx - 1, both_valid
        ] + alpha * pts_world[idx, both_valid]

    return out_pts, out_valid


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
