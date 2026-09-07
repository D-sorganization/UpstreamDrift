"""MediaPipe pose estimation on the MediaPipe Tasks API.

mediapipe >= 0.10 removed the legacy ``mp.solutions`` API. This module drives
``mediapipe.tasks.python.vision.PoseLandmarker`` in VIDEO mode, which needs a
monotonically increasing timestamp per frame and a ``.task`` model file on
disk. Model resolution and verification live in :mod:`.mediapipe_models`; the
library never downloads at ``load_model`` time.

Preserved from the legacy implementation: the 33-landmark naming, the golf
joint mapping, Kalman temporal smoothing, and the joint-angle output.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]
import numpy as np

from src.shared.python.core.contracts import StateError, require
from src.shared.python.engine_core.engine_availability import MEDIAPIPE_AVAILABLE
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_estimation.interface import (
    PoseEstimationResult,
    PoseEstimator,
)
from src.shared.python.pose_estimation.joint_angle_utils import compute_joint_angles
from src.shared.python.pose_estimation.mediapipe_models import resolve_pose_model
from src.shared.python.signal_toolkit.signal_processing import KalmanFilter

# Import mediapipe if available
if MEDIAPIPE_AVAILABLE:
    import mediapipe as mp
else:
    mp = None

logger = get_logger(__name__)


def _tasks_api() -> tuple[Any, Any]:
    """Return ``(vision, BaseOptions)`` from the MediaPipe Tasks API (lazy)."""
    from mediapipe.tasks.python import BaseOptions, vision

    return vision, BaseOptions


class MediaPipeEstimator(PoseEstimator):
    """MediaPipe PoseLandmarker behind the :class:`PoseEstimator` interface.

    Postconditions: ``estimate_from_image`` returns landmarks in normalized
    image coordinates (x, y in [0, 1], z relative) keyed by
    :attr:`LANDMARK_MAP`; ``confidence`` is the mean landmark visibility.
    """

    # MediaPipe Pose landmark mapping (33 landmarks)
    LANDMARK_MAP: ClassVar[dict[int, str]] = {
        0: "nose",
        1: "left_eye_inner",
        2: "left_eye",
        3: "left_eye_outer",
        4: "right_eye_inner",
        5: "right_eye",
        6: "right_eye_outer",
        7: "left_ear",
        8: "right_ear",
        9: "mouth_left",
        10: "mouth_right",
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        17: "left_pinky",
        18: "right_pinky",
        19: "left_index",
        20: "right_index",
        21: "left_thumb",
        22: "right_thumb",
        23: "left_hip",
        24: "right_hip",
        25: "left_knee",
        26: "right_knee",
        27: "left_ankle",
        28: "right_ankle",
        29: "left_heel",
        30: "right_heel",
        31: "left_foot_index",
        32: "right_foot_index",
    }

    # Golf-specific joint mapping for biomechanical analysis
    GOLF_JOINT_MAP: ClassVar[dict[str, str]] = {
        "left_shoulder": "left_shoulder_flexion",
        "right_shoulder": "right_shoulder_flexion",
        "left_elbow": "left_elbow_flexion",
        "right_elbow": "right_elbow_flexion",
        "left_wrist": "left_wrist_flexion",
        "right_wrist": "right_wrist_flexion",
        "left_hip": "left_hip_flexion",
        "right_hip": "right_hip_flexion",
        "left_knee": "left_knee_flexion",
        "right_knee": "right_knee_flexion",
        "left_ankle": "left_ankle_flexion",
        "right_ankle": "right_ankle_flexion",
    }

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        enable_temporal_smoothing: bool = True,
        *,
        model_variant: str | None = None,
        nominal_fps: float = 30.0,
    ) -> None:
        """Configure the estimator; no model is loaded until :meth:`load_model`.

        Args:
            min_detection_confidence: Minimum confidence for pose detection.
            min_tracking_confidence: Minimum confidence for pose tracking.
            enable_temporal_smoothing: Whether to apply Kalman filtering.
            model_variant: ``lite`` / ``full`` / ``heavy``; ``None`` defers to
                ``MEDIAPIPE_POSE_MODEL_VARIANT`` (default ``full``).
            nominal_fps: Frame rate assumed when callers pass no timestamp.
        """
        require(
            0.0 <= min_detection_confidence <= 1.0,
            "min_detection_confidence must be in [0, 1]",
            min_detection_confidence,
        )
        require(
            0.0 <= min_tracking_confidence <= 1.0,
            "min_tracking_confidence must be in [0, 1]",
            min_tracking_confidence,
        )
        require(nominal_fps > 0, "nominal_fps must be positive", nominal_fps)
        self.pose_detector: Any | None = None
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.model_variant = model_variant
        self.nominal_fps = nominal_fps
        self.model_path: Path | None = None
        self._is_loaded = False
        self._frame_index = 0
        self._last_timestamp_ms = -1

        # Temporal smoothing state
        self.previous_landmarks: dict[str, np.ndarray] | None = None
        self.kalman_filters: dict[str, KalmanFilter] = {}

        if not MEDIAPIPE_AVAILABLE:
            logger.warning(
                "MediaPipe Tasks API not available. MediaPipeEstimator will not function."
            )

    def load_model(self, model_path: Path | None = None) -> None:
        """Create the PoseLandmarker from a verified ``.task`` model file.

        Args:
            model_path: Explicit model file; ``None`` resolves through
                ``MEDIAPIPE_POSE_MODEL_PATH`` / the per-user cache.

        Raises:
            ImportError: mediapipe (Tasks API) or OpenCV is not installed.
            ModelError: no model file is present (message carries the fix).
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "mediapipe with the Tasks API (>=0.10) is not installed: pip install mediapipe"
            )
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is not installed.")
        vision, base_options = _tasks_api()
        path = resolve_pose_model(model_path, self.model_variant)
        running_modes = vision.RunningMode
        landmarker_cls = vision.PoseLandmarker
        options = vision.PoseLandmarkerOptions(
            base_options=base_options(model_asset_path=str(path)),
            running_mode=running_modes.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        try:
            self.pose_detector = landmarker_cls.create_from_options(options)
        except (RuntimeError, TypeError, ValueError):
            logger.exception("Failed to create MediaPipe PoseLandmarker from %s", path)
            raise
        self.model_path = path
        self._is_loaded = True
        self._frame_index = 0
        self._last_timestamp_ms = -1
        logger.info("MediaPipe PoseLandmarker loaded from %s", path)

    def close(self) -> None:
        """Release the landmarker; safe to call twice."""
        if self.pose_detector is not None:
            self.pose_detector.close()
            self.pose_detector = None
        self._is_loaded = False

    def _next_timestamp_ms(self, provided: int | None) -> int:
        """VIDEO mode needs strictly increasing timestamps; derive or validate one."""
        if provided is None:
            timestamp = int(round(self._frame_index * 1000.0 / self.nominal_fps))
            self._frame_index += 1
        else:
            timestamp = int(provided)
        if timestamp <= self._last_timestamp_ms:
            timestamp = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp
        return timestamp

    def estimate_from_image(
        self, image: np.ndarray, timestamp_ms: int | None = None
    ) -> PoseEstimationResult:
        """Estimate pose from one BGR frame.

        Args:
            image: Input image (H, W, 3) in BGR format.
            timestamp_ms: Frame time for VIDEO-mode tracking; derived from
                ``nominal_fps`` when omitted.
        """
        if not self._is_loaded or self.pose_detector is None:
            raise StateError("Model not loaded. Call load_model() first.")
        require(
            image.ndim == 3 and image.shape[2] == 3, "image must be HxWx3", image.shape
        )

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_formats = mp.ImageFormat
        mp_image = mp.Image(image_format=image_formats.SRGB, data=rgb_image)
        result = self.pose_detector.detect_for_video(
            mp_image, self._next_timestamp_ms(timestamp_ms)
        )
        poses = result.pose_landmarks
        if not poses:
            return PoseEstimationResult(
                joint_angles={}, confidence=0.0, timestamp=time.time(), raw_keypoints={}
            )
        landmarks = poses[0]
        keypoints_3d: dict[str, np.ndarray] = {}
        confidences: dict[str, float] = {}
        for idx, landmark in enumerate(landmarks):
            name = self.LANDMARK_MAP.get(idx, f"landmark_{idx}")
            keypoints_3d[name] = np.array([landmark.x, landmark.y, landmark.z])
            confidences[name] = float(getattr(landmark, "visibility", 0.0) or 0.0)

        if self.enable_temporal_smoothing:
            keypoints_3d = self._apply_temporal_smoothing(keypoints_3d)

        values = list(confidences.values())
        confidence = sum(values) / len(values) if values else 0.0
        return PoseEstimationResult(
            joint_angles=self._keypoints_to_joint_angles(keypoints_3d),
            confidence=confidence,
            timestamp=time.time(),
            raw_keypoints=keypoints_3d,
            raw_confidences=confidences,
        )

    def estimate_from_video(self, video_path: Path) -> list[PoseEstimationResult]:
        """Process an entire video file with temporal consistency."""
        if not self._is_loaded:
            raise StateError("Model not loaded. Call load_model() first.")
        self.reset_temporal_state()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or self.nominal_fps
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Processing video: %d frames at %.2f FPS", frame_count, fps)

        results: list[PoseEstimationResult] = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_s = frame_idx / fps
            result = self.estimate_from_image(frame, int(round(timestamp_s * 1000)))
            result.timestamp = timestamp_s
            results.append(result)
            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info("Processed %d/%d frames", frame_idx, frame_count)
        cap.release()
        logger.info("Video processing complete: %d frames processed", len(results))
        return results

    def _apply_temporal_smoothing(
        self, keypoints_3d: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Apply a constant-velocity Kalman filter per landmark."""
        require(keypoints_3d is not None, "keypoints_3d must be provided")
        smoothed = {}
        for landmark_name, current_pos in keypoints_3d.items():
            if landmark_name not in self.kalman_filters:
                self.kalman_filters[landmark_name] = self._new_filter(current_pos)
            kf = self.kalman_filters[landmark_name]
            kf.predict()
            kf.update(current_pos)
            smoothed[landmark_name] = kf.x[:3]
        return smoothed

    @staticmethod
    def _new_filter(initial_pos: np.ndarray) -> KalmanFilter:
        # State [x, y, z, vx, vy, vz]; constant velocity with unit normalized dt.
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = 1.0
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        x = np.zeros(6)
        x[:3] = initial_pos
        return KalmanFilter(
            dim_x=6, dim_z=3, F=F, H=H, Q=np.eye(6) * 1e-4, R=np.eye(3) * 1e-3, x=x
        )

    def _keypoints_to_joint_angles(
        self, keypoints_3d: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Convert 3D keypoints to joint angles (radians) via the shared utility."""
        try:
            return compute_joint_angles(keypoints_3d)
        except (RuntimeError, ValueError, OSError):
            logger.warning("Error calculating joint angles", exc_info=True)
            return {}

    def reset_temporal_state(self) -> None:
        """Reset smoothing and VIDEO-mode timestamps (call between videos)."""
        self.previous_landmarks = None
        self.kalman_filters.clear()
        self._frame_index = 0
        self._last_timestamp_ms = -1
