"""OpenCV lens correction at the raw-observation to pinhole-fit boundary.

Projection/undistortion use OpenCV's calibrated camera API. The adapter retains
source files and explicitly identifies ideal-pixel observations to prevent a
second correction. Camera metadata retains distortion for later video overlays.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

import cv2
import numpy as np

from src.shared.python.pose_estimation.observations import CameraIntrinsics

if TYPE_CHECKING:
    from .cameras import PinholeCamera


def project_raw_camera_points(
    points_camera: np.ndarray, matrix: np.ndarray, distortion: np.ndarray
) -> np.ndarray:
    """Project finite camera-frame points with OpenCV's complete lens model.

    OpenCV ignores K's skew entry, so apply that existing camera-axis shear
    after its calibrated normalized projection. Visibility remains with callers.
    """
    points = np.asarray(points_camera, dtype=float)
    intrinsics = CameraIntrinsics(matrix, distortion)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Camera projection requires finite N by 3 points")
    if len(distortion) not in {4, 5, 8, 12, 14}:
        raise ValueError("Unsupported OpenCV pinhole distortion coefficients")
    if not len(points):
        return np.empty((0, 2))
    projection_matrix = intrinsics.matrix.copy()
    projection_matrix[0, 1] = 0
    projected, _ = cv2.projectPoints(
        points, np.zeros(3), np.zeros(3), projection_matrix, distortion
    )
    pixels = projected.reshape(-1, 2)
    pixels[:, 0] += matrix[0, 1] * (pixels[:, 1] - matrix[1, 2]) / matrix[1, 1]
    return np.asarray(pixels, dtype=float)


def retain_camera_lenses(
    cameras: Sequence[PinholeCamera], corrections: Mapping[str, LensCorrection]
) -> tuple[PinholeCamera, ...]:
    """Carry raw-image lens metadata through a fit that uses ideal pixel coordinates."""
    return tuple(
        replace(camera, distortion=corrections[camera.camera_id].intrinsics.distortion)
        if camera.camera_id in corrections
        else camera
        for camera in cameras
    )


def correct_camera_views(
    views: dict[str, dict[str, Any]],
    cameras: Sequence[PinholeCamera] | None,
    lenses: Mapping[str, LensCorrection] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, LensCorrection]]:
    """Apply camera-bound lens metadata once before any keypoint averaging or fit."""
    corrections = dict(lenses or {})
    for camera in cameras or ():
        calibration = camera.to_calibration()
        bound = LensCorrection(calibration.intrinsics, camera.image_size_px)
        if (
            camera.camera_id in corrections
            and corrections[camera.camera_id].signature != bound.signature
        ):
            raise ValueError("Starting camera and supplied lens calibration disagree")
        corrections[camera.camera_id] = bound
    corrected = {
        view: corrections[view].apply(payload) if view in corrections else payload
        for view, payload in views.items()
    }
    return corrected, corrections


@dataclass(frozen=True)
class LensCorrection:
    intrinsics: CameraIntrinsics
    image_size_px: tuple[int, int]

    def __post_init__(self) -> None:
        coefficients = self.intrinsics.distortion
        if coefficients is not None and len(coefficients) not in {0, 4, 5, 8, 12, 14}:
            raise ValueError(
                "Lens correction needs OpenCV pinhole distortion coefficients"
            )
        if len(self.image_size_px) != 2 or min(self.image_size_px) <= 0:
            raise ValueError("Lens correction requires the original image dimensions")
        if not np.allclose(self.intrinsics.matrix[2], [0, 0, 1]):
            raise ValueError("Lens correction requires a normalized camera matrix")

    @property
    def signature(self) -> str:
        payload = {
            "intrinsics": self.intrinsics.to_dict(),
            "image_size_px": self.image_size_px,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return ideal pixels in the same K, leaving raw rows and confidence intact."""
        previous = payload.get("lens_correction")
        if previous is not None:
            if previous != {
                "space": "ideal-pixels",
                "calibration_sha256": self.signature,
            }:
                raise ValueError(
                    "Observations were corrected with a different lens; use the original detections"
                )
            return payload
        size = (payload.get("width"), payload.get("height"))
        if all(value is not None for value in size) and size != self.image_size_px:
            raise ValueError(
                "Observation image dimensions do not match the lens calibration"
            )
        distortion = self.intrinsics.distortion
        if distortion is None or not np.any(distortion):
            return payload
        frames = []
        for row in payload["frames"]:
            points = np.asarray(row["keypoints_px"], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("Expected original image keypoints with shape N by 2")
            corrected = points.copy()
            finite = np.isfinite(points).all(axis=1)
            if finite.any():
                corrected[finite] = self._ideal(points[finite], distortion)
            frames.append({**row, "keypoints_px": corrected.tolist()})
        return {
            **payload,
            "frames": frames,
            "lens_correction": {
                "space": "ideal-pixels",
                "calibration_sha256": self.signature,
            },
        }

    def _ideal(self, points: np.ndarray, distortion: np.ndarray) -> np.ndarray:
        matrix = self.intrinsics.matrix
        adjusted = points.copy()
        adjusted[:, 0] -= matrix[0, 1] * (points[:, 1] - matrix[1, 2]) / matrix[1, 1]
        camera_matrix = matrix.copy()
        camera_matrix[0, 1] = 0
        # Match the canonical provider's convergence criterion and OpenCV 4/5 API.
        criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
        if hasattr(cv2, "undistortPointsIter"):
            rays = cv2.undistortPointsIter(
                adjusted.reshape(-1, 1, 2),
                camera_matrix,
                distortion,
                np.eye(3),
                np.eye(3),
                criteria,
            )
        else:
            # OpenCV 4 stubs do not describe the OpenCV 5 criteria overload.
            iterative_undistort: Any = cv2.undistortPoints
            rays = iterative_undistort(
                adjusted.reshape(-1, 1, 2),
                camera_matrix,
                distortion,
                R=np.eye(3),
                P=np.eye(3),
                criteria=criteria,
            )
        xyz = np.column_stack((rays.reshape(-1, 2), np.ones(len(points))))
        check, _ = cv2.projectPoints(
            xyz, np.zeros(3), np.zeros(3), camera_matrix, distortion
        )
        if not np.isfinite(xyz).all() or not np.allclose(
            check.reshape(-1, 2), adjusted, atol=1e-5, rtol=0
        ):
            raise ValueError(
                "Lens correction did not converge; review calibration and original image coordinates"
            )
        return np.asarray((xyz @ matrix.T)[:, :2], dtype=float)


def lens_corrections_from(path: Path) -> dict[str, LensCorrection]:
    """Read existing flat intrinsic or nested camera-calibration records."""
    payload = json.loads(path.read_bytes())
    records = payload["cameras"] if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError("No camera calibration records in file")
    result = {}
    for record in records:
        view = str(record["camera_id"])
        if view in result:
            raise ValueError("Camera calibration contains duplicate view IDs")
        intrinsics = CameraIntrinsics.from_dict(record.get("intrinsics", record))
        result[view] = LensCorrection(intrinsics, tuple(record["image_size_px"]))
    return result
