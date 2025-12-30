import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shared.python.pose_estimation.openpose_estimator import OpenPoseEstimator


@pytest.fixture(autouse=True)
def mock_openpose_modules():
    """Mock pyopenpose and cv2 modules for all tests."""
    with patch.dict(sys.modules, {"pyopenpose": MagicMock(), "cv2": MagicMock()}):
        # We need to reload the module or ensure it picks up the mock
        # But since we patch sys.modules before import in the original file,
        # we can't easily undo the import at module level.
        # However, for the purpose of the test, we can patch sys.modules
        # and re-import or just rely on the fact that we patched it.

        # NOTE: Since the module 'shared.python.pose_estimation.openpose_estimator'
        # imports pyopenpose at top level (maybe?), we must ensure mocks are in place.
        # But 'OpenPoseEstimator' is already imported at top of this file.
        # The original test file had sys.modules set at TOP LEVEL.
        # Refactoring to a fixture means we must move the import INSIDE the test or fixture
        # OR ensure the mock is established before the test runs.

        # Current 'OpenPoseEstimator' import:
        # from shared.python.pose_estimation.openpose_estimator import OpenPoseEstimator

        # If we move sys.modules mock to a fixture, the top-level import might fail
        # if it really requires pyopenpose to exist.
        # The user wants to fix the module level mocking.

        # Re-importing inside fixture to ensure mocks are used
        import importlib

        import shared.python.pose_estimation.openpose_estimator

        importlib.reload(shared.python.pose_estimation.openpose_estimator)
        yield


# Re-import for type hints if needed, but we rely on the fixture


# Need to mock op globally for the tests to see it?
# In the original file:
# sys.modules["pyopenpose"] = MagicMock()
# import pyopenpose as op
# We need 'op' to be available for assertions.

# We will provide 'op' via fixture or just import it after mocking.


@pytest.fixture
def op_mock(mock_openpose_modules):
    import pyopenpose as op

    return op


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


def test_initialization(mock_openpose_modules):
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
    estimator.wrapper.configure.assert_called()
    estimator.wrapper.start.assert_called()


def test_load_model_failure(estimator, op_mock):
    estimator._is_loaded = False
    op_mock.WrapperPython.side_effect = RuntimeError("OpenPose Error")

    with pytest.raises(RuntimeError):
        estimator.load_model(Path("/tmp"))


def test_estimate_from_image_not_loaded(mock_openpose_modules):
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
    # We mock cv2 inside the test as well to control return values
    with patch("cv2.VideoCapture") as MockCap:
        cap = MockCap.return_value
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
    with patch("cv2.VideoCapture") as MockCap:
        cap = MockCap.return_value
        cap.isOpened.return_value = False

        with pytest.raises(FileNotFoundError):
            estimator.estimate_from_video(Path("test.mp4"))
