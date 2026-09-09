"""OpenPose BODY_25 through OpenCV's DNN module (no ``pyopenpose`` build).

The CMU ``pyopenpose`` bindings need a CUDA build of Caffe that is not
available for current GPUs on Windows, so this estimator runs the published
BODY_25 Caffe network with :mod:`cv2.dnn` on the CPU. It is slower than
MediaPipe (seconds per frame at 368 px input) and exists to *compare* the two
detectors on the same recordings (issue #9628), not to replace either.

Output contract matches :class:`~.mediapipe_estimator.MediaPipeEstimator`:
``raw_keypoints`` are normalized ``(x, y)`` in ``[0, 1]`` keyed by
:attr:`LANDMARK_MAP` names, ``raw_confidences`` are the heat-map peaks, and
``confidence`` is their mean over detected joints. ``joint_angles`` is empty:
this estimator is a detector for the reconstruction pipeline, which computes
angles from fitted skeletons, not from one view.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from src.shared.python.core.contracts import StateError, require
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_estimation.interface import (
    PoseEstimationResult,
    PoseEstimator,
    detection_result,
    estimate_video_frames,
)
from src.shared.python.pose_estimation.openpose_models import resolve_body25

logger = get_logger(__name__)

NetFactory = Callable[[Path, Path], Any]
DEFAULT_INPUT_HEIGHT = 368
MIN_PEAK = 0.05


def _cv2_net(prototxt: Path, weights: Path) -> Any:
    import cv2

    return cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))


class OpenPoseDnnEstimator(PoseEstimator):
    """BODY_25 keypoints from OpenCV DNN heat maps (CPU)."""

    LAYOUT_NAME: ClassVar[str] = "openpose_body25"
    LANDMARK_MAP: ClassVar[dict[int, str]] = {
        0: "nose",
        1: "neck",
        2: "right_shoulder",
        3: "right_elbow",
        4: "right_wrist",
        5: "left_shoulder",
        6: "left_elbow",
        7: "left_wrist",
        8: "mid_hip",
        9: "right_hip",
        10: "right_knee",
        11: "right_ankle",
        12: "left_hip",
        13: "left_knee",
        14: "left_ankle",
        15: "right_eye",
        16: "left_eye",
        17: "right_ear",
        18: "left_ear",
        19: "left_big_toe",
        20: "left_small_toe",
        21: "left_heel",
        22: "right_big_toe",
        23: "right_small_toe",
        24: "right_heel",
    }

    def __init__(
        self,
        *,
        input_height: int = DEFAULT_INPUT_HEIGHT,
        min_peak: float = MIN_PEAK,
        net_factory: NetFactory = _cv2_net,
    ) -> None:
        require(input_height >= 64, "input_height must be at least 64", input_height)
        require(0.0 <= min_peak < 1.0, "min_peak must be in [0, 1)", min_peak)
        self.input_height = input_height
        self.min_peak = min_peak
        self.model_variant = "body25"
        self.model_path: Path | None = None
        self._net_factory = net_factory
        self._net: Any = None

    def load_model(self, model_path: Path | None = None) -> None:
        """Load the network; ``model_path`` is an optional cache directory."""
        prototxt, weights = resolve_body25(model_path)
        self._net = self._net_factory(prototxt, weights)
        self.model_path = weights
        logger.info("OpenPose BODY_25 loaded from %s", weights)

    def is_ready(self) -> bool:
        return self._net is not None

    def close(self) -> None:
        self._net = None

    def estimate_from_video(self, video_path: Path) -> list[PoseEstimationResult]:
        """Run every frame of a video file; timestamps come from the frame rate."""
        if self._net is None:
            raise StateError("load_model() must be called before estimation")
        return estimate_video_frames(video_path, self.estimate_from_image)

    def estimate_from_image(
        self, image: np.ndarray, timestamp_ms: int | None = None
    ) -> PoseEstimationResult:
        """Run the network on one BGR frame and read the 25 heat-map peaks."""
        if self._net is None:
            raise StateError("load_model() must be called before estimation")
        require(
            image.ndim == 3 and image.shape[2] == 3, "image must be HxWx3", image.shape
        )
        import cv2

        height, width = image.shape[:2]
        in_h = self.input_height
        in_w = int(round(width * in_h / height / 8.0)) * 8 or 8
        blob = cv2.dnn.blobFromImage(
            image, 1.0 / 255, (in_w, in_h), (0, 0, 0), swapRB=False, crop=False
        )
        self._net.setInput(blob)
        heatmaps = np.asarray(self._net.forward())[0]
        keypoints, confidences = self.peaks(heatmaps)
        return detection_result(keypoints, confidences, self.min_peak, timestamp_ms)

    def peaks(
        self, heatmaps: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Normalized peak location and value of each BODY_25 heat map.

        Precondition: ``heatmaps`` has at least 25 channels. Joints whose peak is
        below ``min_peak`` are omitted from the keypoints but kept (with their
        low value) in the confidences so the caller sees why they are absent.
        """
        require(heatmaps.ndim == 3 and heatmaps.shape[0] >= 25, "need 25 heat maps")
        _, map_h, map_w = heatmaps.shape
        keypoints: dict[str, np.ndarray] = {}
        confidences: dict[str, float] = {}
        for idx, name in self.LANDMARK_MAP.items():
            heat = heatmaps[idx]
            flat = int(np.argmax(heat))
            row, col = divmod(flat, map_w)
            peak = float(heat[row, col])
            confidences[name] = max(0.0, min(1.0, peak))
            if peak >= self.min_peak:
                keypoints[name] = np.array(
                    [(col + 0.5) / map_w, (row + 0.5) / map_h], dtype=float
                )
        return keypoints, confidences
