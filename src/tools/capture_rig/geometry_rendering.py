"""Calibrated scene-reference overlay for both preview and export frames."""

from math import isfinite

import cv2
import numpy as np

from src.motion_capture.coaching.geometry import (
    ReferenceGeometry,
    ReferencePlane,
    ReferencePoint,
)
from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reference.registration import project_reference_to_camera
from src.shared.python.pose_estimation.observations import CameraCalibration


def render_geometry(
    frame: np.ndarray,
    geometry: ReferenceGeometry,
    camera: PinholeCamera | CameraCalibration | None,
    time_s: float,
) -> np.ndarray:
    """Overlay world references without mutating the source BGR uint8 image.

    Near-plane crossing polygons are omitted; image borders are clipped by
    OpenCV. References are illustrative overlays without model occlusion.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("Reference overlay needs a BGR uint8 image")
    if not isfinite(time_s):
        raise ValueError("Scene time must be finite")
    if camera is None:
        raise ValueError("Metric references require a calibrated or virtual camera")
    if frame.shape[:2] != tuple(reversed(camera.image_size_px)):
        raise ValueError("Reference image dimensions must match the camera")
    result = frame.copy()
    references: tuple[ReferencePlane | ReferencePoint, ...] = (
        *geometry.planes,
        *geometry.points,
    )
    for reference in references:
        if not reference.at(time_s) or reference.opacity == 0:
            continue
        if isinstance(reference, ReferencePlane):
            vertices = reference.vertices()
        else:
            vertices = np.asarray([reference.position_m], dtype=np.float64)
        pixels, visible = project_reference_to_camera(
            vertices, np.ones(len(vertices), dtype=bool), camera, clip_image=False
        )
        if (
            not visible.all()
            or not np.isfinite(pixels).all()
            or np.abs(pixels).max() > 1e7
        ):
            continue
        colour = tuple(int(reference.colour[i : i + 2], 16) for i in (5, 3, 1))
        polygon = np.rint(pixels).astype(np.int32)
        drawn = result.copy()
        if isinstance(reference, ReferencePlane):
            cv2.fillConvexPoly(drawn, polygon, colour, cv2.LINE_AA)
        else:
            point = tuple(int(value) for value in polygon[0])
            cv2.drawMarker(drawn, point, colour, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        result = cv2.addWeighted(
            drawn, reference.opacity, result, 1 - reference.opacity, 0
        )
    return np.asarray(result, dtype=np.uint8)
