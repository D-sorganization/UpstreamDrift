"""Interface for pose estimation modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Estimator types the runtime pipeline can actually construct
# (VideoPosePipeline._load_estimator). The API's VALID_ESTIMATOR_TYPES must
# mirror this set — enforced by tests/unit/test_estimator_type_consistency.py
# (epic #8390, A2/#8392). Lives here rather than in gui_pkg so that
# dependency-light consumers (API config, tests) can import it without
# pulling the GUI stack (matplotlib, cv2).
IMPLEMENTED_ESTIMATOR_TYPES: frozenset[str] = frozenset({"mediapipe", "openpose"})


@dataclass
class PoseEstimationResult:
    """Standardized result from a pose estimator."""

    joint_angles: dict[str, float]  # Joint name -> angle (radians)
    confidence: float  # 0.0 to 1.0
    timestamp: float
    raw_keypoints: dict[str, np.ndarray] | None = None  # Optional raw 2D/3D points


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
