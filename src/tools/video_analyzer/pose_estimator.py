"""
Pose Estimator using MediaPipe.

Provides real-time pose estimation for golf swing analysis,
extracting 33 body landmarks from video frames.
"""

import logging
from typing import Any

import cv2
import numpy as np

from .types import Landmark, PoseFrame

logger = logging.getLogger(__name__)

# Try to import mediapipe, but allow graceful fallback
try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not installed. Install with: pip install mediapipe")


class PoseEstimator:
    """
    MediaPipe-based pose estimation for golf swing analysis.

    Provides 33 body landmarks per frame with 3D coordinates
    and visibility confidence scores.

    Usage:
        estimator = PoseEstimator()
        landmarks = estimator.process_frame(frame)
    """

    # MediaPipe landmark indices
    LANDMARK_NAMES = [
        "nose",
        "left_eye_inner",
        "left_eye",
        "left_eye_outer",
        "right_eye_inner",
        "right_eye",
        "right_eye_outer",
        "left_ear",
        "right_ear",
        "mouth_left",
        "mouth_right",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_pinky",
        "right_pinky",
        "left_index",
        "right_index",
        "left_thumb",
        "right_thumb",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    ]

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smooth_landmarks: bool = True,
    ) -> None:
        """
        Initialize the pose estimator.

        Args:
            model_complexity: Model complexity (0, 1, or 2). Higher is more accurate
                             but slower.
            min_detection_confidence: Minimum confidence for pose detection (0-1).
            min_tracking_confidence: Minimum confidence for landmark tracking (0-1).
            smooth_landmarks: Whether to apply landmark smoothing.
        """
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smooth_landmarks = smooth_landmarks
        self._pose = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize MediaPipe pose estimation.

        Returns:
            True if initialization successful, False otherwise.
        """
        if not MEDIAPIPE_AVAILABLE:
            logger.error("MediaPipe is not available")
            return False

        try:
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                smooth_landmarks=self.smooth_landmarks,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._initialized = True
            logger.info("PoseEstimator initialized successfully")
            return True
        except (RuntimeError, ValueError, OSError) as e:
            logger.error(f"Failed to initialize PoseEstimator: {e}")
            return False

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> PoseFrame | None:
        """
        Process a single video frame and extract pose landmarks.

        Args:
            frame: BGR image frame from OpenCV.
            frame_number: Frame index in the video.
            timestamp: Timestamp in milliseconds.

        Returns:
            PoseFrame with landmarks, or None if no pose detected.
        """
        if not self._initialized:
            if not self.initialize():
                return None

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = self._pose.process(rgb_frame)  # type: ignore[attr-defined]

        if not results.pose_landmarks:
            return None

        # Extract landmarks
        landmarks = []
        total_visibility = 0.0

        for lm in results.pose_landmarks.landmark:
            landmarks.append(
                Landmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=lm.visibility,
                )
            )
            total_visibility += lm.visibility

        # Calculate average confidence
        confidence = total_visibility / len(landmarks) if landmarks else 0.0

        return PoseFrame(
            frame_number=frame_number,
            timestamp=timestamp,
            landmarks=landmarks,
            confidence=confidence,
        )

    def process_frame_world(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> PoseFrame | None:
        """
        Process frame and return world coordinates (real-world scale).

        World landmarks are in meters, centered at the hip midpoint.

        Args:
            frame: BGR image frame from OpenCV.
            frame_number: Frame index in the video.
            timestamp: Timestamp in milliseconds.

        Returns:
            PoseFrame with world-space landmarks, or None if no pose detected.
        """
        if not self._initialized:
            if not self.initialize():
                return None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb_frame)  # type: ignore[attr-defined]

        if not results.pose_world_landmarks:
            return None

        landmarks = []
        total_visibility = 0.0

        # Use world landmarks for real-world scale
        for i, lm in enumerate(results.pose_world_landmarks.landmark):
            # Get visibility from regular landmarks
            vis = (
                results.pose_landmarks.landmark[i].visibility
                if results.pose_landmarks
                else 0.5
            )
            landmarks.append(
                Landmark(
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    visibility=vis,
                )
            )
            total_visibility += vis

        confidence = total_visibility / len(landmarks) if landmarks else 0.0

        return PoseFrame(
            frame_number=frame_number,
            timestamp=timestamp,
            landmarks=landmarks,
            confidence=confidence,
        )

    def draw_landmarks(
        self,
        frame: np.ndarray,
        pose_frame: PoseFrame,
        draw_connections: bool = True,
        landmark_color: tuple = (0, 255, 0),
        connection_color: tuple = (0, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw pose landmarks on a frame.

        Args:
            frame: BGR image frame.
            pose_frame: PoseFrame with landmarks to draw.
            draw_connections: Whether to draw skeleton connections.
            landmark_color: BGR color for landmarks.
            connection_color: BGR color for connections.
            thickness: Line thickness.

        Returns:
            Frame with landmarks drawn.
        """
        if not MEDIAPIPE_AVAILABLE:
            return frame

        h, w = frame.shape[:2]
        output = frame.copy()

        # Draw connections
        if draw_connections:
            connections = self._mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start = pose_frame.landmarks[start_idx]
                end = pose_frame.landmarks[end_idx]

                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(
                        output, start_point, end_point, connection_color, thickness
                    )

        # Draw landmarks
        for landmark in pose_frame.landmarks:
            if landmark.visibility > 0.5:
                center = (int(landmark.x * w), int(landmark.y * h))
                cv2.circle(output, center, 4, landmark_color, -1)
                cv2.circle(output, center, 6, (255, 255, 255), 1)

        return output

    def close(self) -> None:
        """Release resources."""
        if self._pose:
            self._pose.close()
            self._pose = None
            self._initialized = False

    def __enter__(self) -> Any:
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        self.close()
        return False
