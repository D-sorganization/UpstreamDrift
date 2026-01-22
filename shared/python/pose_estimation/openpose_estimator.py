"""OpenPose wrapper implementation for pose estimation.

This module provides a production-ready wrapper around the pyopenpose library,
implementing the standardized PoseEstimator interface.
"""

import logging
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

# Try to import pyopenpose. If not found, we will fall back to mock/error behavior
# explicitly handled in the class methods or via a probe.
try:
    import pyopenpose as op
except ImportError:
    op = None

from shared.python.pose_estimation.interface import (
    PoseEstimationResult,
    PoseEstimator,
)

logger = logging.getLogger(__name__)


class OpenPoseEstimator(PoseEstimator):
    """OpenPose-based implementation of PoseEstimator."""

    # Standard BODY_25 Keypoint Mapping
    KEYPOINT_MAP: ClassVar[dict[int, str]] = {
        0: "Nose",
        1: "Neck",
        2: "RShoulder",
        3: "RElbow",
        4: "RWrist",
        5: "LShoulder",
        6: "LElbow",
        7: "LWrist",
        8: "MidHip",
        9: "RHip",
        10: "RKnee",
        11: "RAnkle",
        12: "LHip",
        13: "LKnee",
        14: "LAnkle",
        15: "REye",
        16: "LEye",
        17: "REar",
        18: "LEar",
        19: "LBigToe",
        20: "LSmallToe",
        21: "LHeel",
        22: "RBigToe",
        23: "RSmallToe",
        24: "RHeel",
    }

    def __init__(self) -> None:
        """Initialize the OpenPose estimator wrapper."""
        self.wrapper: Any | None = None
        self.params: dict[str, Any] = {}
        self._is_loaded = False

        if op is None:
            logger.warning(
                "pyopenpose library not found. OpenPoseEstimator will not function."
            )

    def load_model(self, model_path: Path | None = None) -> None:
        """Configure and start the OpenPose wrapper.

        Args:
            model_path: Path to the 'models' directory of OpenPose.
                        If None, attempts to find it in standard locations or env vars.
        """
        if op is None:
            raise ImportError("pyopenpose module is not installed.")

        self.params = {}

        # Set model folder
        if model_path:
            self.params["model_folder"] = str(model_path)
        else:
            # Try to guess or rely on default
            # Often assumes models are at relative path or configured in ENV
            # Placeholder default for Windows standard install
            default_path = Path("C:/openpose/models")
            if default_path.exists():
                self.params["model_folder"] = str(default_path)
            else:
                logger.warning("No model_path provided and default not found.")

        # Default configuration
        self.params["model_pose"] = "BODY_25"
        self.params["net_resolution"] = "-1x368"
        self.params["number_people_max"] = 1  # Assume single golfer for analysis

        try:
            self.wrapper = op.WrapperPython()
            self.wrapper.configure(self.params)
            self.wrapper.start()
            self._is_loaded = True
            logger.info("OpenPose wrapper started successfully.")
        except Exception as e:
            logger.error(f"Failed to start OpenPose wrapper: {e}")
            raise

    def estimate_from_image(self, image: np.ndarray) -> PoseEstimationResult:
        """Process a single image frame."""
        if not self._is_loaded or self.wrapper is None:
            raise RuntimeError("OpenPose model not loaded. Call load_model() first.")

        try:
            datum = op.Datum()
            datum.cvInputData = image
            self.wrapper.emplaceAndPop([datum])

            # consistency check
            if datum.poseKeypoints is None or datum.poseKeypoints.size == 0:
                logger.warning("No pose detected in frame.")
                return PoseEstimationResult(
                    joint_angles={}, confidence=0.0, timestamp=0.0, raw_keypoints={}
                )

            # datum.poseKeypoints shape: (num_people, num_parts, 3) -> (x, y, score)
            # We take the first person (index 0)
            person_keypoints = datum.poseKeypoints[0]

            keypoints_dict = {}
            total_score = 0.0
            valid_points = 0

            for idx, (x, y, score) in enumerate(person_keypoints):
                if score > 0.0:
                    name = self.KEYPOINT_MAP.get(idx, f"Pt{idx}")
                    keypoints_dict[name] = np.array([x, y, score])
                    total_score += score
                    valid_points += 1

            avg_confidence = total_score / valid_points if valid_points > 0 else 0.0

            # returning empty joint angles until a biomechanical model is integrated.
            return PoseEstimationResult(
                raw_keypoints=keypoints_dict,
                confidence=avg_confidence,
                timestamp=time.time(),
                joint_angles={},
            )

        except Exception as e:
            logger.error(f"Error during OpenPose inference: {e}")
            raise

    def estimate_from_video(self, video_path: Path) -> list[PoseEstimationResult]:
        """Process a video file."""
        try:
            import cv2
        except ImportError:
            raise RuntimeError("OpenCV (cv2) is not installed. Cannot process video.")

        results = []
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Timestamp in seconds
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                result = self.estimate_from_image(frame)
                result.timestamp = ts
                results.append(result)
        finally:
            cap.release()

        return results
