"""Virtual camera fitting and 2D swing envelope generation (COV-4, #11272).

Fits a virtual camera mapping capture-O 3D landmarks into 2D video views and
computes phase-normalized 2D p5/p50/p95 swing variation envelopes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.shared.python.core.contracts import require
from src.shared.python.motion_matching.historical_fit.contracts import CameraProjection
from src.shared.python.pose_estimation.observations import CameraIntrinsics

logger = logging.getLogger(__name__)


class CameraFitOutcome(str, Enum):
    """Outcome status of virtual camera fitting."""

    FITTED = "fitted"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class VirtualCameraResult:
    """Estimated virtual camera parameters, uncertainty, and fit quality."""

    parameters: dict[str, float]
    covariance: np.ndarray
    fit_rms_px: float
    condition_number: float
    outcome: CameraFitOutcome
    degraded_reason: str | None = None
    camera: PinholeCamera | None = None


@dataclass(frozen=True)
class SwingEnvelope2D:
    """Per-landmark phase-normalized 2D percentile bands."""

    phase_bins: np.ndarray
    p5: np.ndarray
    p50: np.ndarray
    p95: np.ndarray
    sample_counts: np.ndarray
    landmark_names: tuple[str, ...] = ()
    p5_height_normalized: np.ndarray | None = None
    p50_height_normalized: np.ndarray | None = None
    p95_height_normalized: np.ndarray | None = None


def pinhole_camera_to_camera_projection(camera: PinholeCamera) -> CameraProjection:
    """Convert PinholeCamera (world-from-camera) to CameraProjection (world-to-camera)."""
    r_cw = camera.rotation_world_from_camera.T
    t_cw = -r_cw @ camera.translation_world_from_camera_m
    return CameraProjection(
        intrinsics=camera.matrix,
        rotation=r_cw,
        translation=t_cw,
    )


def camera_projection_to_pinhole_camera(
    projection: CameraProjection,
    image_size_px: tuple[int, int],
    *,
    camera_id: str = "virtual",
) -> PinholeCamera:
    """Convert CameraProjection (world-to-camera) to PinholeCamera (world-from-camera)."""
    r_wc = projection.rotation.T
    t_wc = -r_wc @ projection.translation
    return PinholeCamera(
        camera_id=camera_id,
        matrix=projection.intrinsics,
        rotation_world_from_camera=r_wc,
        translation_world_from_camera_m=t_wc,
        image_size_px=image_size_px,
    )


def _check_inputs_finite_and_shapes(
    world_points: np.ndarray,
    image_points: np.ndarray,
) -> None:
    """Validate numeric arrays for finiteness, dimension and correspondence count."""
    require(np.isfinite(world_points).all(), "world_points must be finite")
    require(np.isfinite(image_points).all(), "image_points must be finite")
    require(
        world_points.ndim == 2 and world_points.shape[1] == 3,
        "world_points must have shape (N, 3)",
    )
    require(
        image_points.ndim == 2 and image_points.shape[1] == 2,
        "image_points must have shape (N, 2)",
    )
    require(
        len(world_points) == len(image_points),
        "world_points and image_points must have matching length",
    )
    require(
        len(world_points) >= 4,
        "At least 4 correspondences are required for camera fitting",
    )


def _check_axis_and_chirality(
    world_points: np.ndarray,
    image_points: np.ndarray,
) -> None:
    """Validate that world points follow ADR-0041 convention and are not mirrored."""
    ptp_y = float(np.ptp(world_points[:, 1]))
    ptp_z = float(np.ptp(world_points[:, 2]))
    require(
        ptp_y >= ptp_z * 0.6,
        "Mirrored projection detected: input points do not follow the required axis convention "
        "(vertical span along Y is smaller than lateral span along Z; ensure y_up_to_z_up was applied)",
    )


def _check_geometric_degeneracy(world_points: np.ndarray) -> str | None:
    """Return a reason if 3D correspondences are geometrically degenerate."""
    centered = world_points - np.mean(world_points, axis=0)
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s[0] < 1e-6:
        return "Correspondences are coincident"
    if s[1] / s[0] < 1e-4 or s[1] < 1e-6:
        return "Correspondences are collinear (rank < 2)"
    if s[2] / s[0] < 0.02 or s[2] < 1e-4:
        return "Correspondences are coplanar (rank < 3); depth/pose is ambiguous"
    return None


def _make_degraded_result(
    matrix: np.ndarray,
    image_size_px: tuple[int, int],
    reason: str,
    camera_id: str,
) -> VirtualCameraResult:
    """Construct a degraded result holding parameters at prior."""
    r_wc = np.eye(3, dtype=float)
    t_wc = np.array([0.0, 1.2, 3.0], dtype=float)
    cam = PinholeCamera(
        camera_id=camera_id,
        matrix=matrix,
        rotation_world_from_camera=r_wc,
        translation_world_from_camera_m=t_wc,
        image_size_px=image_size_px,
    )
    params = {
        "r_x": 0.0,
        "r_y": 0.0,
        "r_z": 0.0,
        "t_x": float(t_wc[0]),
        "t_y": float(t_wc[1]),
        "t_z": float(t_wc[2]),
        "fx_px": float(matrix[0, 0]),
        "fy_px": float(matrix[1, 1]),
        "cx_px": float(matrix[0, 2]),
        "cy_px": float(matrix[1, 2]),
    }
    return VirtualCameraResult(
        parameters=params,
        covariance=np.zeros((6, 6), dtype=float),
        fit_rms_px=0.0,
        condition_number=float("inf"),
        outcome=CameraFitOutcome.DEGRADED,
        degraded_reason=reason,
        camera=cam,
    )


def _optimize_camera_pose(
    world_points: np.ndarray,
    image_points: np.ndarray,
    matrix: np.ndarray,
    weights: np.ndarray,
    initial_pose: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate camera rotation and translation via least-squares reprojection."""
    if initial_pose is not None:
        r_cw_init = initial_pose[0].T
        t_cw_init = -r_cw_init @ initial_pose[1]
        x0 = np.concatenate(
            [cv2.Rodrigues(r_cw_init)[0].flatten(), t_cw_init.flatten()]
        )
    else:
        success, rvec, tvec = cv2.solvePnP(
            world_points.astype(np.float64),
            image_points.astype(np.float64),
            matrix.astype(np.float64),
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        x0 = (
            np.concatenate([rvec.flatten(), tvec.flatten()])
            if success
            else np.array([0, 0, 0, 0, 0, 3.0])
        )

    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    sqrt_w = np.sqrt(weights)[:, None]

    def _residuals(params: np.ndarray) -> np.ndarray:
        r_mat = cv2.Rodrigues(params[:3])[0]
        pts_cam = world_points @ r_mat.T + params[3:6]
        depth = np.maximum(pts_cam[:, 2], 1e-5)
        u = fx * (pts_cam[:, 0] / depth) + cx
        v = fy * (pts_cam[:, 1] / depth) + cy
        pred = np.column_stack([u, v])
        return ((pred - image_points) * sqrt_w).reshape(-1)

    opt = least_squares(_residuals, x0, method="lm", xtol=1e-10, ftol=1e-10)
    jac_arr = (
        np.asarray(opt.jac.toarray(), dtype=float)
        if hasattr(opt.jac, "toarray")
        else np.asarray(opt.jac, dtype=float)
    )
    return np.asarray(opt.x, dtype=float), jac_arr, np.asarray(opt.fun, dtype=float)


def fit_virtual_camera(
    world_points: npt.ArrayLike,
    image_points: npt.ArrayLike,
    intrinsics_prior: npt.ArrayLike | CameraIntrinsics,
    image_size_px: tuple[int, int],
    *,
    weights: npt.ArrayLike | None = None,
    initial_pose: tuple[np.ndarray, np.ndarray] | None = None,
    max_condition_number: float = 1e4,
    camera_id: str = "virtual",
) -> VirtualCameraResult:
    """Fit camera extrinsics mapping 3D world points to 2D image coordinates."""
    w_pts = np.asarray(world_points, dtype=float)
    img_pts = np.asarray(image_points, dtype=float)
    _check_inputs_finite_and_shapes(w_pts, img_pts)
    _check_axis_and_chirality(w_pts, img_pts)

    matrix = (
        intrinsics_prior.matrix
        if isinstance(intrinsics_prior, CameraIntrinsics)
        else np.asarray(intrinsics_prior, dtype=float)
    )
    require(
        matrix.shape == (3, 3) and matrix[0, 0] > 0 and matrix[1, 1] > 0,
        "Invalid intrinsics matrix",
    )

    degraded_reason = _check_geometric_degeneracy(w_pts)
    if degraded_reason is not None:
        return _make_degraded_result(matrix, image_size_px, degraded_reason, camera_id)

    w = (
        np.ones(len(w_pts), dtype=float)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    require(
        bool(len(w) == len(w_pts) and np.isfinite(w).all() and (w >= 0).all()),
        "Invalid weights",
    )

    sol, jac, res = _optimize_camera_pose(w_pts, img_pts, matrix, w, initial_pose)
    s_j = np.linalg.svd(jac, compute_uv=False)
    cond = float(s_j[0] / s_j[-1]) if s_j[-1] > 1e-12 else float("inf")

    if cond > max_condition_number:
        return _make_degraded_result(
            matrix,
            image_size_px,
            f"Condition number {cond:.1e} exceeds {max_condition_number:.1e}",
            camera_id,
        )

    r_cw = cv2.Rodrigues(sol[:3])[0]
    r_wc = np.asarray(r_cw.T, dtype=float)
    t_wc = np.asarray(-r_wc @ sol[3:6], dtype=float)
    s2 = float(np.sum(res**2) / max(1, len(res) - 6))
    cov = s2 * np.linalg.pinv(jac.T @ jac)

    # Reprojection RMS in pixels
    pts_c = w_pts @ r_cw.T + sol[3:6]
    pred = np.column_stack(
        [
            matrix[0, 0] * (pts_c[:, 0] / pts_c[:, 2]) + matrix[0, 2],
            matrix[1, 1] * (pts_c[:, 1] / pts_c[:, 2]) + matrix[1, 2],
        ]
    )
    rms_px = float(np.sqrt(np.mean(np.sum((pred - img_pts) ** 2, axis=-1))))

    cam = PinholeCamera(
        camera_id=camera_id,
        matrix=matrix,
        rotation_world_from_camera=r_wc,
        translation_world_from_camera_m=t_wc,
        image_size_px=image_size_px,
    )
    params = {
        "r_x": float(sol[0]),
        "r_y": float(sol[1]),
        "r_z": float(sol[2]),
        "t_x": float(t_wc[0]),
        "t_y": float(t_wc[1]),
        "t_z": float(t_wc[2]),
        "fx_px": float(matrix[0, 0]),
        "fy_px": float(matrix[1, 1]),
        "cx_px": float(matrix[0, 2]),
        "cy_px": float(matrix[1, 2]),
    }
    return VirtualCameraResult(
        parameters=params,
        covariance=cov,
        fit_rms_px=rms_px,
        condition_number=cond,
        outcome=CameraFitOutcome.FITTED,
        camera=cam,
    )


def compute_2d_envelope(
    projected_swings: npt.ArrayLike,
    phase_bins: npt.ArrayLike,
    *,
    body_height_px: float | None = None,
    landmark_names: Sequence[str] | None = None,
    expected_swings: int = 13,
) -> SwingEnvelope2D:
    """Compute per-landmark p5/p50/p95 2D envelope bands across phase-normalized swings."""
    swings = np.asarray(projected_swings, dtype=float)
    require(
        swings.ndim == 4 and swings.shape[-1] == 2,
        "projected_swings must have shape (S, T, K, 2)",
    )
    require(
        len(swings) == expected_swings,
        f"Expected {expected_swings} swings, got {len(swings)}",
    )

    bins = np.asarray(phase_bins, dtype=float)
    require(
        bins.ndim == 1 and len(bins) == swings.shape[1],
        "phase_bins must be 1D matching time dimension",
    )

    n_swings, n_phases, n_landmarks, _ = swings.shape
    p5 = np.full((n_phases, n_landmarks, 2), np.nan, dtype=float)
    p50 = np.full((n_phases, n_landmarks, 2), np.nan, dtype=float)
    p95 = np.full((n_phases, n_landmarks, 2), np.nan, dtype=float)
    counts = np.zeros((n_phases, n_landmarks), dtype=int)

    for t in range(n_phases):
        for k in range(n_landmarks):
            coords = swings[:, t, k, :]  # (S, 2)
            valid = ~np.isnan(coords).any(axis=1)
            n_valid = int(np.sum(valid))
            counts[t, k] = n_valid
            if n_valid > 0:
                valid_pts = coords[valid]
                p5[t, k, :] = np.percentile(valid_pts, 5, axis=0)
                p50[t, k, :] = np.percentile(valid_pts, 50, axis=0)
                p95[t, k, :] = np.percentile(valid_pts, 95, axis=0)

    p5_norm = (
        p5 / body_height_px
        if body_height_px is not None and body_height_px > 0
        else None
    )
    p50_norm = (
        p50 / body_height_px
        if body_height_px is not None and body_height_px > 0
        else None
    )
    p95_norm = (
        p95 / body_height_px
        if body_height_px is not None and body_height_px > 0
        else None
    )

    names = tuple(landmark_names) if landmark_names is not None else ()
    return SwingEnvelope2D(
        phase_bins=bins,
        p5=p5,
        p50=p50,
        p95=p95,
        sample_counts=counts,
        landmark_names=names,
        p5_height_normalized=p5_norm,
        p50_height_normalized=p50_norm,
        p95_height_normalized=p95_norm,
    )
