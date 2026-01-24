import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We import the module under test normally
# dependencies will be handled by patching attributes or local imports
from src.shared.python.pose_estimation.openpose_estimator import (
    OpenPoseEstimator,
)


@pytest.fixture
def op_mock():
    mock_op = MagicMock()
    with patch("src.shared.python.pose_estimation.openpose_estimator.op", mock_op):
        yield mock_op


@pytest.fixture
def mock_op_wrapper(op_mock):
    wrapper = MagicMock()
    op_mock.WrapperPython.return_value = wrapper
    return wrapper


@pytest.fixture
def estimator(mock_op_wrapper):
    est = OpenPoseEstimator()
    # Mock loaded state for processing tests
    est.wrapper = mock_op_wrapper
    est._is_loaded = True
    return est


def test_initialization():
    est = OpenPoseEstimator()
    assert est.wrapper is None
    assert est._is_loaded is False


def test_load_model_success(estimator, op_mock):
    # Reset to unloaded
    estimator._is_loaded = False
    estimator.wrapper = None

    path = Path("/tmp/models")
    estimator.load_model(path)

    assert estimator._is_loaded is True
    assert estimator.params["model_folder"] == str(path)
    op_mock.WrapperPython.assert_called()
    if estimator.wrapper is not None:
        estimator.wrapper.configure.assert_called()
        estimator.wrapper.start.assert_called()


def test_load_model_failure(estimator, op_mock):
    estimator._is_loaded = False
    op_mock.WrapperPython.side_effect = RuntimeError("OpenPose Error")

    with pytest.raises(RuntimeError):
        estimator.load_model(Path("/tmp"))


def test_estimate_from_image_not_loaded():
    est = OpenPoseEstimator()
    with pytest.raises(RuntimeError):
        est.estimate_from_image(np.zeros((100, 100, 3)))


def test_estimate_from_image_success(estimator, op_mock):
    # Setup mock datum
    datum = MagicMock()
    # Mock shape: (1 person, 25 parts, 3 values)
    datum.poseKeypoints = np.ones((1, 25, 3))
    # Make scores variable to test confidence calc
    datum.poseKeypoints[0, :, 2] = 0.8

    # Mock op.Datum()
    op_mock.Datum.return_value = datum

    # Mock wrapper behavior
    estimator.wrapper.emplaceAndPop.side_effect = (
        lambda x: None
    )  # Modify datum in place effectively

    img = np.zeros((100, 100, 3))
    result = estimator.estimate_from_image(img)

    assert result.confidence == pytest.approx(0.8)
    assert len(result.raw_keypoints) == 25
    assert "Nose" in result.raw_keypoints


def test_estimate_from_image_no_pose(estimator, op_mock):
    datum = MagicMock()
    datum.poseKeypoints = None  # Or empty array
    op_mock.Datum.return_value = datum

    result = estimator.estimate_from_image(np.zeros((100, 100, 3)))
    assert result.confidence == 0.0
    assert len(result.raw_keypoints) == 0


def test_estimate_from_video_success(estimator, op_mock):
    # Mock cv2 module since it's imported locally
    mock_cv2 = MagicMock()
    with patch.dict(sys.modules, {"cv2": mock_cv2}):
        # Setup VideoCapture mock
        cap = mock_cv2.VideoCapture.return_value
        cap.isOpened.return_value = True

        # Return 2 frames then stop
        cap.read.side_effect = [
            (True, np.zeros((100, 100, 3))),
            (True, np.zeros((100, 100, 3))),
            (False, None),
        ]
        cap.get.return_value = 100.0  # Timestamp

        # Mock image processing
        datum = MagicMock()
        datum.poseKeypoints = np.ones((1, 25, 3))
        op_mock.Datum.return_value = datum

        results = estimator.estimate_from_video(Path("test.mp4"))
        assert len(results) == 2
        assert results[0].timestamp == 0.1  # 100ms -> 0.1s


def test_estimate_from_video_not_found(estimator):
    mock_cv2 = MagicMock()
    with patch.dict(sys.modules, {"cv2": mock_cv2}):
        cap = mock_cv2.VideoCapture.return_value
        cap.isOpened.return_value = False

        with pytest.raises(FileNotFoundError):
            estimator.estimate_from_video(Path("test.mp4"))
