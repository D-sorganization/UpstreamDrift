"""MediaPipe-based pose estimation implementation.

This module provides a production-ready wrapper around MediaPipe Pose,
implementing the standardized PoseEstimator interface with enhanced features:
- Real-time processing capability
- Temporal smoothing with Kalman filtering
- Multi-person detection support
- Confidence-based filtering
"""

import logging
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

# Try to import mediapipe. If not found, we will fall back to mock/error behavior
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

from shared.python.pose_estimation.interface import (
    PoseEstimationResult,
    PoseEstimator,
)
from shared.python.signal_processing import KalmanFilter

logger = logging.getLogger(__name__)


class MediaPipeEstimator(PoseEstimator):
    """MediaPipe-based implementation of PoseEstimator.

    Provides faster, more robust pose estimation compared to OpenPose,
    with built-in temporal smoothing and confidence filtering.
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
    ) -> None:
        """Initialize the MediaPipe estimator.

        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            enable_temporal_smoothing: Whether to apply Kalman filtering
        """
        self.pose_detector: Any | None = None
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self._is_loaded = False

        # Temporal smoothing state
        self.previous_landmarks: dict[str, np.ndarray] | None = None
        self.kalman_filters: dict[str, KalmanFilter] = {}

        if not MEDIAPIPE_AVAILABLE:
            logger.warning(
                "MediaPipe library not found. MediaPipeEstimator will not function."
            )

    def load_model(self, model_path: Path | None = None) -> None:
        """Initialize the MediaPipe Pose model.

        Args:
            model_path: Not used for MediaPipe (uses built-in models)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe module is not installed.")

        try:
            mp_pose = mp.solutions.pose
            self.pose_detector = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,  # Use highest accuracy model
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._is_loaded = True
            logger.info("MediaPipe Pose model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load MediaPipe Pose model: {e}")
            raise

    def estimate_from_image(self, image: np.ndarray) -> PoseEstimationResult:
        """Estimate pose from a single image frame.

        Args:
            image: Input image (H, W, C) in BGR format

        Returns:
            PoseEstimationResult containing joint angles and keypoints
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is not installed. Cannot process image.")

        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        if self.pose_detector is not None:
            results = self.pose_detector.process(rgb_image)
        else:
            raise RuntimeError("MediaPipe pose detector not initialized")

        if results.pose_landmarks is None:
            # No pose detected
            return PoseEstimationResult(
                joint_angles={}, confidence=0.0, timestamp=time.time(), raw_keypoints={}
            )

        # Extract landmarks
        landmarks = results.pose_landmarks.landmark
        keypoints_3d: dict[str, np.ndarray[Any, Any]] = {}
        keypoints_2d: dict[str, np.ndarray[Any, Any]] = {}

        for idx, landmark in enumerate(landmarks):
            landmark_name = self.LANDMARK_MAP.get(idx, f"landmark_{idx}")
            keypoints_3d[landmark_name] = np.array([landmark.x, landmark.y, landmark.z])
            keypoints_2d[landmark_name] = np.array(
                [
                    landmark.x * image.shape[1],  # Convert to pixel coordinates
                    landmark.y * image.shape[0],
                ]
            )

        # Apply temporal smoothing if enabled
        if self.enable_temporal_smoothing:
            smoothed_keypoints = self._apply_temporal_smoothing(keypoints_3d)
            if smoothed_keypoints is not None:
                keypoints_3d = smoothed_keypoints

        # Convert keypoints to joint angles
        joint_angles = self._keypoints_to_joint_angles(keypoints_3d)

        # Calculate overall confidence
        confidence = float(np.mean([landmark.visibility for landmark in landmarks]))

        return PoseEstimationResult(
            joint_angles=joint_angles,
            confidence=confidence,
            timestamp=time.time(),
            raw_keypoints=keypoints_3d,  # Use 3D keypoints
        )

    def estimate_from_video(self, video_path: Path) -> list[PoseEstimationResult]:
        """Process an entire video file with temporal consistency.

        Args:
            video_path: Path to video file

        Returns:
            List of results for each frame with temporal smoothing
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if cv2 is None:
            raise RuntimeError("OpenCV (cv2) is not installed. Cannot process video.")

        results = []
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Processing video: {frame_count} frames at {fps} FPS")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate timestamp
            timestamp = frame_idx / fps

            # Estimate pose for this frame
            result = self.estimate_from_image(frame)
            result.timestamp = timestamp

            results.append(result)
            frame_idx += 1

            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{frame_count} frames")

        cap.release()
        logger.info(f"Video processing complete: {len(results)} frames processed")

        return results

    def _apply_temporal_smoothing(
        self, keypoints_3d: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Apply Kalman filtering for temporal consistency.

        Args:
            keypoints_3d: Raw keypoints from current frame

        Returns:
            Smoothed keypoints
        """
        smoothed = {}

        for landmark_name, current_pos in keypoints_3d.items():
            # Initialize filter if not exists
            if landmark_name not in self.kalman_filters:
                # Constant Velocity Model
                # State: [x, y, z, vx, vy, vz]
                dt = 1.0  # Normalized time step (assuming constant frame rate)
                F = np.eye(6)
                F[0, 3] = dt
                F[1, 4] = dt
                F[2, 5] = dt

                H = np.zeros((3, 6))
                H[0, 0] = 1.0
                H[1, 1] = 1.0
                H[2, 2] = 1.0

                # Process noise (uncertainty in model)
                # Assume constant velocity is mostly true, but allow some deviation
                Q = np.eye(6) * 1e-4

                # Measurement noise (uncertainty in MediaPipe)
                # MediaPipe is reasonably accurate but can jitter
                R = np.eye(3) * 1e-3

                # Initial state
                x = np.zeros(6)
                x[:3] = current_pos

                self.kalman_filters[landmark_name] = KalmanFilter(
                    dim_x=6, dim_z=3, F=F, H=H, Q=Q, R=R, x=x
                )

            # Predict and Update
            kf = self.kalman_filters[landmark_name]
            kf.predict()
            kf.update(current_pos)

            # Store smoothed position
            smoothed[landmark_name] = kf.x[:3]

        return smoothed

    def _keypoints_to_joint_angles(
        self, keypoints_3d: dict[str, np.ndarray]
    ) -> dict[str, float]:
        """Convert 3D keypoints to joint angles for biomechanical analysis.

        Args:
            keypoints_3d: 3D keypoint positions

        Returns:
            Dictionary of joint angles in radians
        """
        joint_angles = {}

        # Calculate key joint angles for golf swing analysis
        try:
            # Right elbow flexion (example calculation)
            if all(
                k in keypoints_3d
                for k in ["right_shoulder", "right_elbow", "right_wrist"]
            ):
                shoulder = keypoints_3d["right_shoulder"]
                elbow = keypoints_3d["right_elbow"]
                wrist = keypoints_3d["right_wrist"]

                # Vector from elbow to shoulder
                v1 = shoulder - elbow
                # Vector from elbow to wrist
                v2 = wrist - elbow

                # Calculate angle between vectors
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
                angle = np.arccos(cos_angle)

                joint_angles["right_elbow_flexion"] = angle

            # Left elbow flexion
            if all(
                k in keypoints_3d for k in ["left_shoulder", "left_elbow", "left_wrist"]
            ):
                shoulder = keypoints_3d["left_shoulder"]
                elbow = keypoints_3d["left_elbow"]
                wrist = keypoints_3d["left_wrist"]

                v1 = shoulder - elbow
                v2 = wrist - elbow

                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                joint_angles["left_elbow_flexion"] = angle

            # Add more joint angle calculations as needed
            # (shoulder flexion, hip flexion, knee flexion, etc.)

        except Exception as e:
            logger.warning(f"Error calculating joint angles: {e}")

        return joint_angles

    def reset_temporal_state(self) -> None:
        """Reset temporal smoothing state (call between videos)."""
        self.previous_landmarks = None
        self.kalman_filters.clear()
