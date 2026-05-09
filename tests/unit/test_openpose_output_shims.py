"""Importability tests for openpose_estimator and output_manager shims (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.output_manager import OutputFormat, OutputManager
from src.shared.python.pose_estimation.openpose_estimator import OpenPoseEstimator


class TestOpenPoseEstimatorImportable:
    def test_openpose_output_shims_importable(self) -> None:
        assert OpenPoseEstimator is not None

    def test_has_keypoint_map(self) -> None:
        assert hasattr(OpenPoseEstimator, "KEYPOINT_MAP")


class TestOutputManagerShimImportable:
    def test_output_manager_importable(self) -> None:
        assert OutputManager is not None

    def test_output_format_importable(self) -> None:
        assert OutputFormat is not None

    def test_output_format_has_json(self) -> None:
        assert OutputFormat.JSON.value == "json"
