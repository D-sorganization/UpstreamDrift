# ruff: noqa: E402
"""Unit tests for OpenPoseEstimator."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock pyopenpose before validation
mock_op = MagicMock()
sys.modules["pyopenpose"] = mock_op

from shared.python.pose_estimation.openpose_estimator import (
    OpenPoseEstimator,  # noqa: E402
)


@pytest.fixture
def estimator():
    return OpenPoseEstimator()


def test_initialization():
    # Test with op mocked (success case)
    with patch("shared.python.pose_estimation.openpose_estimator.op", mock_op):
        est = OpenPoseEstimator()
        assert est.wrapper is None
        assert est._is_loaded is False


def test_load_model_success(estimator):
    with patch("shared.python.pose_estimation.openpose_estimator.op", mock_op):
        estimator.load_model(model_path="dummy/path")

        estimator.wrapper.configure.assert_called()
        estimator.wrapper.start.assert_called()
        assert estimator._is_loaded is True
        assert estimator.params["model_folder"] == "dummy/path"


def test_estimate_from_image_success(estimator):
    with patch("shared.python.pose_estimation.openpose_estimator.op", mock_op):
        # Setup loaded state
        estimator.wrapper = MagicMock()
        estimator._is_loaded = True

        # Mock Datum behavior
        # op.Datum() returns a datum instance
        mock_datum = MagicMock()
        mock_op.Datum.return_value = mock_datum

        # Setup specific keypoint output
        # Shape: (1 person, 25 points, 3 val)
        # Point 0: Nose at 100,100, 0.9 conf
        points = np.zeros((1, 25, 3))
        points[0, 0] = [100, 100, 0.9]

        # emulate wrapper.emplaceAndPop filling the datum
        def side_effect(datum_list):
            datum_list[0].poseKeypoints = points

        estimator.wrapper.emplaceAndPop.side_effect = side_effect

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = estimator.estimate_from_image(img)

        assert result.confidence > 0.0
        assert "Nose" in result.raw_keypoints
        assert np.allclose(result.raw_keypoints["Nose"], [100, 100, 0.9])

        estimator.wrapper.emplaceAndPop.assert_called()


def test_estimate_without_load_raises(estimator):
    with pytest.raises(RuntimeError):
        estimator.estimate_from_image(np.zeros((10, 10, 3)))


def test_no_poses_detected(estimator):
    with patch("shared.python.pose_estimation.openpose_estimator.op", mock_op):
        estimator.wrapper = MagicMock()
        estimator._is_loaded = True

        mock_datum = MagicMock()
        mock_op.Datum.return_value = mock_datum

        # Empty array or None
        def side_effect(datum_list):
            datum_list[0].poseKeypoints = None

        estimator.wrapper.emplaceAndPop.side_effect = side_effect

        result = estimator.estimate_from_image(np.zeros((10, 10, 3)))
        assert result.confidence == 0.0
        assert result.raw_keypoints == {}
