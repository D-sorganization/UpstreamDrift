"""Pose-conditioned camera initialization, never a measured calibration."""

from __future__ import annotations

import cv2
import numpy as np

from .contracts import CameraProjection


def initialize_camera_hypothesis(
    assumed_world_points: np.ndarray,
    observed_pixels: np.ndarray,
    assumed_intrinsics: np.ndarray,
) -> CameraProjection:
    """Estimate extrinsics conditional on supplied pose, geometry and optics priors.

    Correspondences must be observed; callers remove missing landmarks first.
    A small residual does not establish correct geometry, optics or metric scale.
    """
    points = np.ascontiguousarray(assumed_world_points, dtype=float)
    pixels = np.ascontiguousarray(observed_pixels, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or len(points) < 6
        or pixels.shape != (len(points), 2)
        or not np.isfinite(points).all()
        or not np.isfinite(pixels).all()
        or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2
    ):
        raise ValueError(
            "Camera initialization requires six nondegenerate geometry correspondences"
        )
    validated = CameraProjection(assumed_intrinsics, np.eye(3), np.zeros(3))
    try:
        success, rotation_vector, translation = cv2.solvePnP(
            points, pixels, validated.intrinsics, None, flags=cv2.SOLVEPNP_SQPNP
        )
        if not success:
            raise ValueError("Pose-conditioned camera initialization did not converge")
        rotation_vector, translation = cv2.solvePnPRefineLM(
            points, pixels, validated.intrinsics, None, rotation_vector, translation
        )
        rotation, _ = cv2.Rodrigues(rotation_vector)
    except cv2.error as exc:
        raise ValueError("Pose-conditioned camera initialization failed") from exc
    camera = CameraProjection(validated.intrinsics, rotation, translation.reshape(3))
    camera.project(points)
    return camera
