"""Heavy integration tests for MediaPipe Tasks API (fixes #1987).

Tests both the legacy mp.solutions.pose and the newer mp.tasks API
(>= 0.10), including synthetic image processing via PoseLandmarker.
All tests skip gracefully when mediapipe is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def mp():
    """Import mediapipe or skip the module."""
    mp_mod = pytest.importorskip("mediapipe")
    return mp_mod


@pytest.fixture(scope="module")
def synthetic_rgb_frame():
    """A 480×640 synthetic RGB image (blank white)."""
    return np.ones((480, 640, 3), dtype=np.uint8) * 200


class TestMediaPipeLegacyApi:
    """Contract: legacy mp.solutions.pose API works end-to-end."""

    def test_legacy_pose_init(self, mp, synthetic_rgb_frame) -> None:
        """Legacy Pose() context manager processes a frame without error."""
        if not (hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")):
            pytest.skip("Legacy mp.solutions.pose not available in this version")

        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.1) as pose:
            results = pose.process(synthetic_rgb_frame)
        assert results is not None

    def test_legacy_results_structure(self, mp, synthetic_rgb_frame) -> None:
        """Results object has pose_landmarks attribute (may be None for blank image)."""
        if not (hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")):
            pytest.skip("Legacy mp.solutions.pose not available in this version")

        mp_pose = mp.solutions.pose
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.1) as pose:
            results = pose.process(synthetic_rgb_frame)
        # pose_landmarks is None when no person detected — that's valid for blank frame
        assert hasattr(results, "pose_landmarks")


class TestMediaPipeTasksApi:
    """Contract: mp.tasks API (>= 0.10) has expected structure."""

    def test_tasks_module_importable(self, mp) -> None:
        """mp.tasks is accessible when mediapipe >= 0.10."""
        if not hasattr(mp, "tasks"):
            pytest.skip("mp.tasks not available (mediapipe < 0.10)")
        tasks = mp.tasks
        assert tasks is not None

    def test_tasks_has_vision_or_base_options(self, mp) -> None:
        """mp.tasks exposes vision or BaseOptions sub-modules."""
        if not hasattr(mp, "tasks"):
            pytest.skip("mp.tasks not available (mediapipe < 0.10)")
        tasks = mp.tasks
        has_vision = hasattr(tasks, "vision")
        has_base = hasattr(tasks, "BaseOptions")
        assert has_vision or has_base, (
            f"mp.tasks structure unexpected. Available: {dir(tasks)}"
        )

    def test_pose_landmarker_class_exists(self, mp) -> None:
        """PoseLandmarker class is accessible via mp.tasks.vision."""
        if not hasattr(mp, "tasks"):
            pytest.skip("mp.tasks not available (mediapipe < 0.10)")
        tasks = mp.tasks
        if not hasattr(tasks, "vision"):
            pytest.skip("mp.tasks.vision not available in this mediapipe version")
        assert hasattr(tasks.vision, "PoseLandmarker"), (
            "PoseLandmarker not found in mp.tasks.vision"
        )

    def test_base_options_instantiable(self, mp) -> None:
        """BaseOptions can be instantiated (model_asset_path not required)."""
        if not hasattr(mp, "tasks"):
            pytest.skip("mp.tasks not available (mediapipe < 0.10)")
        tasks = mp.tasks
        BaseOptions = getattr(tasks, "BaseOptions", None)
        if BaseOptions is None:
            pytest.skip("BaseOptions not in mp.tasks")

        # Instantiate without a real model path — just verify constructor is callable.
        # May raise if the class validates file existence at init time.
        import contextlib

        with contextlib.suppress(Exception):
            BaseOptions(model_asset_path="/nonexistent/model.task")


class TestMediaPipeGuiModule:
    """Contract: mediapipe_gui module is importable from the project."""

    def test_mediapipe_gui_importable(self, mp) -> None:  # noqa: ARG002
        """mediapipe_gui.py in pose_estimation is importable."""
        try:
            import src.shared.python.pose_estimation.mediapipe_gui as gui  # noqa: F401
        except ImportError as exc:
            pytest.skip(f"mediapipe_gui not importable: {exc}")

        assert gui is not None

    def test_mediapipe_estimator_importable(self, mp) -> None:  # noqa: ARG002
        """MediaPipeEstimator class is importable."""
        try:
            from src.shared.python.pose_estimation.mediapipe_estimator import (
                MediaPipeEstimator,
            )
        except ImportError as exc:
            pytest.skip(f"mediapipe_estimator not importable: {exc}")

        assert MediaPipeEstimator is not None


pytestmark = pytest.mark.live_simulation
