import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure repo root is in path
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)

# Skip entire module if cv2 is not available (optional dependency)
cv2 = pytest.importorskip("cv2", reason="OpenCV (cv2) not installed")

from src.shared.python.pose_estimation import mediapipe_estimator  # noqa: E402


class TestMediaPipeEstimator:
    @pytest.fixture
    def mock_mediapipe(self) -> MagicMock:
        """Mock mediapipe module."""
        mock_mp = MagicMock()
        mock_mp.solutions.pose.Pose.return_value = MagicMock()

        # Setup landmark structure
        mock_results = MagicMock()

        # Create a list of distinct landmarks (33 usually)
        landmarks = []
        for _ in range(33):
            lm = MagicMock()
            lm.x = 0.5
            lm.y = 0.5
            lm.z = 0.0
            lm.visibility = 0.9
            landmarks.append(lm)

        mock_results.pose_landmarks.landmark = landmarks

        mock_mp.solutions.pose.Pose.return_value.process.return_value = mock_results

        return mock_mp

    @pytest.fixture
    def estimator_instance(
        self, mock_mediapipe: MagicMock
    ) -> Generator[mediapipe_estimator.MediaPipeEstimator, None, None]:
        """Return a MediaPipeEstimator instance with mocked dependencies."""
        with patch.object(mediapipe_estimator, "MEDIAPIPE_AVAILABLE", True):
            with patch.object(mediapipe_estimator, "mp", mock_mediapipe):
                # We also need to patch cv2 since it's used in methods
                with patch(
                    "src.shared.python.pose_estimation.mediapipe_estimator.cv2",
                    MagicMock(),
                ):
                    estimator = mediapipe_estimator.MediaPipeEstimator(
                        min_detection_confidence=0.7
                    )
                    yield estimator

    def test_initialization(
        self, estimator_instance: mediapipe_estimator.MediaPipeEstimator
    ) -> None:
        """Test initialization of the estimator."""
        assert estimator_instance.min_detection_confidence == 0.7
        assert estimator_instance.pose_detector is None
        assert estimator_instance._is_loaded is False

    def test_load_model(
        self,
        estimator_instance: mediapipe_estimator.MediaPipeEstimator,
        mock_mediapipe: MagicMock,
    ) -> None:
        """Test loading the model."""
        estimator_instance.load_model()

        assert estimator_instance._is_loaded is True
        assert estimator_instance.pose_detector is not None
        mock_mediapipe.solutions.pose.Pose.assert_called_once()

    def test_estimate_from_image(
        self, estimator_instance: mediapipe_estimator.MediaPipeEstimator
    ) -> None:
        """Test estimation from a single image."""
        estimator_instance.enable_temporal_smoothing = False
        estimator_instance.load_model()

        # Create dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Ensure cv2 is mocked correctly in the module
        with patch(
            "src.shared.python.pose_estimation.mediapipe_estimator.cv2"
        ) as mock_cv2:
            mock_cv2.cvtColor.return_value = image  # Return same image

            result = estimator_instance.estimate_from_image(image)

            assert result.confidence == 0.9
            assert result.raw_keypoints is not None
            assert "nose" in result.raw_keypoints
            assert result.raw_keypoints["nose"][0] == 0.5

    def test_estimate_from_video(
        self, estimator_instance: mediapipe_estimator.MediaPipeEstimator
    ) -> None:
        """Test estimation from video."""
        estimator_instance.load_model()

        # Mock cv2.VideoCapture via the module import
        with patch(
            "src.shared.python.pose_estimation.mediapipe_estimator.cv2"
        ) as mock_cv2:
            mock_cap = MagicMock()
            mock_cv2.VideoCapture.return_value = mock_cap
            mock_cap.isOpened.return_value = True
            mock_cap.get.side_effect = [30.0, 10.0]  # fps, frame_count

            # Mock reading 2 frames then stop
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_cap.read.side_effect = [(True, frame), (True, frame), (False, None)]

            # We also need to mock cvtColor
            mock_cv2.cvtColor.return_value = frame

            results = estimator_instance.estimate_from_video(Path("test.mp4"))

            assert len(results) == 2
            assert results[0].timestamp == 0.0
            assert results[1].timestamp == 1 / 30.0

    def test_temporal_smoothing(
        self, estimator_instance: mediapipe_estimator.MediaPipeEstimator
    ) -> None:
        """Test that temporal smoothing affects results."""
        estimator_instance.enable_temporal_smoothing = True
        estimator_instance.load_model()

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch(
            "src.shared.python.pose_estimation.mediapipe_estimator.cv2"
        ) as mock_cv2:
            mock_cv2.cvtColor.return_value = image

            # Process first frame
            result1 = estimator_instance.estimate_from_image(image)

            # Process second frame
            result2 = estimator_instance.estimate_from_image(image)

            assert result1 is not None
            assert result2 is not None

    def test_joint_angles_calculation(
        self,
        estimator_instance: mediapipe_estimator.MediaPipeEstimator,
        mock_mediapipe: MagicMock,
    ) -> None:
        """Test calculation of joint angles."""
        estimator_instance.load_model()

        mock_results = (
            mock_mediapipe.solutions.pose.Pose.return_value.process.return_value
        )
        landmarks = mock_results.pose_landmarks.landmark

        # Reset all to 0
        for lm in landmarks:
            lm.x = 0.0
            lm.y = 0.0
            lm.z = 0.0
            lm.visibility = 1.0

        # Set specific landmarks
        # right shoulder (12)
        landmarks[12].x = 0.0
        landmarks[12].y = 1.0
        landmarks[12].z = 0.0
        # right elbow (14)
        landmarks[14].x = 0.0
        landmarks[14].y = 0.0
        landmarks[14].z = 0.0
        # right wrist (16)
        landmarks[16].x = 1.0
        landmarks[16].y = 0.0
        landmarks[16].z = 0.0

        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch(
            "src.shared.python.pose_estimation.mediapipe_estimator.cv2"
        ) as mock_cv2:
            mock_cv2.cvtColor.return_value = image

            result = estimator_instance.estimate_from_image(image)

            assert "right_elbow_flexion" in result.joint_angles
            angle = result.joint_angles["right_elbow_flexion"]
            assert np.isclose(angle, np.pi / 2, atol=1e-5)

    def test_missing_mediapipe(self) -> None:
        """Test behavior when mediapipe is missing."""
        # We patch module attributes to simulate missing mediapipe
        with patch.object(mediapipe_estimator, "MEDIAPIPE_AVAILABLE", False):
            with patch.object(mediapipe_estimator, "mp", None):
                # Initialize should check MEDIAPIPE_AVAILABLE but only warn
                # The __init__ in the code warns but doesn't raise
                with patch(
                    "src.shared.python.pose_estimation.mediapipe_estimator.logger"
                ) as mock_logger:
                    estimator = mediapipe_estimator.MediaPipeEstimator()
                    mock_logger.warning.assert_called()

                # load_model should raise ImportError
                with pytest.raises(
                    ImportError, match="MediaPipe module is not installed"
                ):
                    estimator.load_model()
