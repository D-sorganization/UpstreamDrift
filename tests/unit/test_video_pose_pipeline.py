# Ensure shared/python is in sys.path
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../shared/python"))
)

# Mock cv2 before importing video_pose_pipeline
sys.modules["cv2"] = MagicMock()

# Mock shared.python.pose_estimation.mediapipe_estimator before import
mock_mp_module = MagicMock()
sys.modules["src.shared.python.pose_estimation.mediapipe_estimator"] = mock_mp_module

mock_op_module = MagicMock()
sys.modules["src.shared.python.pose_estimation.openpose_estimator"] = mock_op_module


from src.shared.python.pose_estimation.interface import (
    PoseEstimationResult,  # noqa: E402
)
from src.shared.python.video_pose_pipeline import (  # noqa: E402
    VideoPosePipeline,
    VideoProcessingConfig,
    VideoProcessingResult,
)


@pytest.fixture
def mock_cv2():
    return sys.modules["cv2"]


@pytest.fixture
def mock_output_manager():
    with patch("src.shared.python.video_pose_pipeline.OutputManager") as mock:
        yield mock


@pytest.fixture
def pipeline(mock_cv2, mock_output_manager):
    config = VideoProcessingConfig(estimator_type="mediapipe", min_confidence=0.5)

    # Ensure MediaPipeEstimator class is a mock
    mock_mp_module.MediaPipeEstimator.return_value = MagicMock()

    pipeline = VideoPosePipeline(config)
    return pipeline


def test_initialization(pipeline):
    """Test pipeline initialization."""
    assert pipeline.config.estimator_type == "mediapipe"
    assert pipeline.estimator is not None
    # Verify load_model was called on the instance
    pipeline.estimator.load_model.assert_called_once()


def test_process_video_file_not_found(pipeline):
    """Test processing a non-existent video."""
    with pytest.raises(FileNotFoundError):
        pipeline.process_video(Path("non_existent.mp4"))


def test_process_video_success(pipeline, mock_cv2):
    """Test successful video processing."""
    video_path = Path("test_video.mp4")

    # Setup CV2 constants
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_FPS = 5

    # Mock file existence
    with patch.object(Path, "exists", return_value=True):
        # Mock cv2.VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        # Simplified get_prop
        mock_cap.get.return_value = 10.0

        # Mock reading frames: 1 success, then fail
        mock_cap.read.side_effect = [(True, np.zeros((100, 100, 3))), (False, None)]
        mock_cv2.VideoCapture.return_value = mock_cap

        # Mock estimator result
        mock_pose_result = PoseEstimationResult(
            raw_keypoints={"nose": np.array([0.5, 0.5, 0.0])},
            confidence=0.9,
            timestamp=0.0,
            joint_angles={},
        )

        # Force estimator to be a MagicMock that returns our result
        pipeline.estimator = MagicMock()
        # Delete estimate_from_video to force frame-by-frame processing path
        del pipeline.estimator.estimate_from_video
        pipeline.estimator.estimate_from_image.return_value = mock_pose_result

        # Run processing
        result = pipeline.process_video(video_path)

        assert isinstance(result, VideoProcessingResult)
        assert result.total_frames == 10
        assert result.processed_frames == 1
        assert result.average_confidence == 0.9


def test_filter_by_quality(pipeline):
    """Test quality filtering logic."""
    results = [
        PoseEstimationResult(
            joint_angles={}, confidence=0.9, timestamp=0.0, raw_keypoints={}
        ),
        PoseEstimationResult(
            joint_angles={}, confidence=0.1, timestamp=0.1, raw_keypoints={}
        ),  # Low confidence
        PoseEstimationResult(
            joint_angles={}, confidence=0.8, timestamp=0.2, raw_keypoints={}
        ),
    ]

    filtered = pipeline._filter_by_quality(results)
    assert len(filtered) == 2
    assert filtered[0].confidence == 0.9
    assert filtered[1].confidence == 0.8


def test_is_outlier(pipeline):
    """Test outlier detection."""
    # Create a set of consistent results with slight variance to ensure std > 0
    consistent_results = []
    for i in range(10):
        consistent_results.append(
            PoseEstimationResult(
                raw_keypoints={},
                confidence=0.9,
                timestamp=0.0,
                joint_angles={"elbow": 90.0 + (i % 2) * 0.1},  # 90.0 and 90.1
            )
        )

    # Outlier result (elbow angle far from ~90)
    outlier = PoseEstimationResult(
        raw_keypoints={}, confidence=0.9, timestamp=0.0, joint_angles={"elbow": 180.0}
    )

    # Normal result
    normal = PoseEstimationResult(
        raw_keypoints={}, confidence=0.9, timestamp=0.0, joint_angles={"elbow": 90.05}
    )

    assert pipeline._is_outlier(outlier, consistent_results) is True
    assert pipeline._is_outlier(normal, consistent_results) is False


def test_process_batch(pipeline):
    """Test batch processing."""
    video_paths = [Path("vid1.mp4"), Path("vid2.mp4")]
    output_dir = Path("output")

    with patch.object(pipeline, "process_video") as mock_process:
        mock_process.return_value = MagicMock(
            valid_frames=10, total_frames=20, average_confidence=0.8
        )

        results = pipeline.process_batch(video_paths, output_dir)

        assert len(results) == 2
        assert mock_process.call_count == 2
        pipeline.output_manager.save_simulation_results.assert_called()  # Batch summary
