"""Interface for pose estimation modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
        pass

    @abstractmethod
    def estimate_from_image(self, image: np.ndarray) -> PoseEstimationResult:
        """Estimate pose from a single image frame.

        Args:
            image: Input image (H, W, C) usually BGR or RGB.

        Returns:
            PoseEstimationResult containing joint angles.
        """
        pass

    @abstractmethod
    def estimate_from_video(self, video_path: Path) -> list[PoseEstimationResult]:
        """Process an entire video file.

        Args:
            video_path: Path to video file.

        Returns:
            List of results for each frame.
        """
        pass
