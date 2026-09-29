"""Interface for pose estimation modules."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Estimator types the runtime pipeline can actually construct, derived
# from the estimator registry (epic #8390, C2/#8402 — one seam instead of
# the historical 5-place edit tax). The API's VALID_ESTIMATOR_TYPES must
# mirror this set — enforced by tests/unit/test_estimator_type_consistency.py
# (A2/#8392). Import stays dependency-light: the registry only imports
# heavy estimator modules inside factories.
from src.shared.python.pose_estimation.registry import (  # noqa: E402
    implemented_estimator_types as _implemented_estimator_types,
)

IMPLEMENTED_ESTIMATOR_TYPES: frozenset[str] = _implemented_estimator_types()


@dataclass
class PoseEstimationResult:
    """Standardized result from a pose estimator."""

    joint_angles: dict[str, float]  # Joint name -> angle (radians)
    confidence: float  # 0.0 to 1.0
    timestamp: float
    raw_keypoints: dict[str, np.ndarray] | None = None  # Optional raw 2D/3D points
    # Optional per-keypoint confidence in [0, 1], keyed like raw_keypoints.
    raw_confidences: dict[str, float] | None = None


class PoseEstimator(ABC):
    """Abstract base class for pose estimators."""

    @abstractmethod
    def load_model(self, model_path: Path | None = None) -> None:
        """Load the estimation model/weights.

        Args:
            model_path: Path to model weights, or None for default.
        """

    @abstractmethod
    def estimate_from_image(self, image: np.ndarray) -> PoseEstimationResult:
        """Estimate pose from a single image frame.

        Args:
            image: Input image (H, W, C) usually BGR or RGB.

        Returns:
            PoseEstimationResult containing joint angles.
        """

    @abstractmethod
    def estimate_from_video(self, video_path: Path) -> list[PoseEstimationResult]:
        """Process an entire video file.

        Args:
            video_path: Path to video file.

        Returns:
            List of results for each frame.
        """


def estimate_video_frames(
    video_path: Path,
    estimate_frame: Callable[[np.ndarray, int], PoseEstimationResult],
) -> list[PoseEstimationResult]:
    """Decode every frame of a video file and estimate pose on each one.

    Shared frame loop for whole-frame estimators (issue #9648): the
    per-frame work stays in the caller's ``estimate_frame`` callback, which
    receives the BGR frame and a millisecond timestamp derived from the
    frame rate (falling back to 30 fps when the container reports none).

    Args:
        video_path: Path to the video file.
        estimate_frame: Callable invoked per frame with ``(frame, ts_ms)``.

    Returns:
        One :class:`PoseEstimationResult` per decoded frame.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    results: list[PoseEstimationResult] = []
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            results.append(estimate_frame(frame, int(round(index * 1000.0 / fps))))
            index += 1
    finally:
        cap.release()
    return results


def detection_result(
    keypoints: dict[str, np.ndarray],
    confidences: dict[str, float],
    min_confidence: float,
    timestamp_ms: int | None = None,
) -> PoseEstimationResult:
    """Build a detector-style result from raw peaks (issue #9648).

    Shared result assembly for whole-frame detectors: ``confidence`` is the
    mean over joints whose peak reaches ``min_confidence`` (0.0 when none
    do), and the timestamp converts a millisecond frame timestamp to
    seconds, falling back to wall-clock time.

    Args:
        keypoints: Raw joint locations keyed by joint name.
        confidences: Raw peak values keyed by joint name.
        min_confidence: Threshold a joint must reach to count as detected.
        timestamp_ms: Frame timestamp in milliseconds, or None for wall clock.

    Returns:
        A :class:`PoseEstimationResult` with empty ``joint_angles``.
    """
    import time

    detected = [c for c in confidences.values() if c >= min_confidence]
    return PoseEstimationResult(
        joint_angles={},
        confidence=float(np.mean(detected)) if detected else 0.0,
        timestamp=(timestamp_ms / 1000.0) if timestamp_ms is not None else time.time(),
        raw_keypoints=keypoints,
        raw_confidences=confidences,
    )
