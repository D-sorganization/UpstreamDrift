"""Tests for pose_estimation.interface (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.pose_estimation.interface import (
    PoseEstimationResult,
    PoseEstimator,
)


class TestPoseEstimationResult:
    def test_pose_estimation_construction(self) -> None:
        result = PoseEstimationResult(
            joint_angles={"hip": 0.1, "knee": 0.2},
            confidence=0.95,
            timestamp=1.0,
        )
        assert result is not None

    def test_confidence_field(self) -> None:
        result = PoseEstimationResult(
            joint_angles={"hip": 0.0},
            confidence=0.8,
            timestamp=0.0,
        )
        assert result.confidence == 0.8

    def test_joint_angles_field(self) -> None:
        result = PoseEstimationResult(
            joint_angles={"hip": 0.5, "knee": 0.3},
            confidence=1.0,
            timestamp=0.0,
        )
        assert result.joint_angles["hip"] == 0.5

    def test_optional_raw_keypoints(self) -> None:
        result = PoseEstimationResult(
            joint_angles={},
            confidence=1.0,
            timestamp=0.0,
            raw_keypoints={"nose": np.array([0.5, 0.5, 0.0])},
        )
        assert result.raw_keypoints is not None


class TestPoseEstimatorProtocol:
    def test_is_abstract(self) -> None:
        import inspect

        assert inspect.isabstract(PoseEstimator)

    def test_pose_estimation_importable(self) -> None:
        assert PoseEstimator is not None
