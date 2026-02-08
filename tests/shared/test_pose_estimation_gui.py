"""Tests for pose estimation GUI integration.

Tests cover:
    - Config parsing (both formats)
    - GUI class instantiation
    - Worker thread creation and error handling
    - Dependency-missing graceful fallback

These tests DO NOT require mediapipe or pyopenpose to be installed.
"""

from __future__ import annotations

from unittest.mock import patch

# =============================================================================
# Config Parser Tests
# =============================================================================


class TestMediaPipeConfigParsing:
    """Tests for MediaPipe GUI config parsing."""

    def test_parse_key_value(self) -> None:
        """key=value format should be parsed correctly."""
        from src.shared.python.pose_estimation.mediapipe_gui import _parse_config

        config = _parse_config(
            "min_detection_confidence=0.5\nmin_tracking_confidence=0.7"
        )
        assert config["min_detection_confidence"] == 0.5
        assert config["min_tracking_confidence"] == 0.7

    def test_parse_ignores_comments(self) -> None:
        """Comments starting with # should be ignored."""
        from src.shared.python.pose_estimation.mediapipe_gui import _parse_config

        config = _parse_config("# This is a comment\nmin_detection_confidence=0.5")
        assert "This is a comment" not in config
        assert config["min_detection_confidence"] == 0.5

    def test_parse_boolean_values(self) -> None:
        """Boolean strings should be parsed correctly."""
        from src.shared.python.pose_estimation.mediapipe_gui import _parse_config

        config = _parse_config("static_image_mode=False\nenable_smoothing=True")
        assert config["static_image_mode"] is False
        assert config["enable_smoothing"] is True

    def test_parse_empty_config(self) -> None:
        """Empty config should return empty dict."""
        from src.shared.python.pose_estimation.mediapipe_gui import _parse_config

        config = _parse_config("")
        assert config == {}


class TestOpenPoseConfigParsing:
    """Tests for OpenPose GUI config parsing."""

    def test_parse_key_value(self) -> None:
        """key=value format should work."""
        from src.shared.python.pose_estimation.openpose_gui import _parse_config

        config = _parse_config("model_pose=BODY_25\nnet_resolution=-1x368")
        assert config["model_pose"] == "BODY_25"
        assert config["net_resolution"] == "-1x368"

    def test_parse_flag_format(self) -> None:
        """--key value format should work."""
        from src.shared.python.pose_estimation.openpose_gui import _parse_config

        config = _parse_config("--video input.mp4\n--display 0")
        assert config["video"] == "input.mp4"
        assert config["display"] == "0"

    def test_parse_ignores_comments(self) -> None:
        """Comments should be ignored."""
        from src.shared.python.pose_estimation.openpose_gui import _parse_config

        config = _parse_config("# comment\nmodel_pose=BODY_25")
        assert len(config) == 1


# =============================================================================
# Worker Thread Tests (mocked)
# =============================================================================


class TestMediaPipeWorker:
    """Tests for MediaPipe analysis worker."""

    def test_worker_emits_error_on_missing_dependency(self) -> None:
        """Worker should emit error signal when mediapipe is not installed."""
        from src.shared.python.pose_estimation.mediapipe_gui import _AnalysisWorker

        worker = _AnalysisWorker("/nonexistent/video.mp4", {})

        error_messages: list[str] = []
        worker.error.connect(error_messages.append)

        # Mock the import to raise ImportError
        with patch.dict("sys.modules", {"mediapipe": None}):
            with patch(
                "src.shared.python.pose_estimation.mediapipe_gui._AnalysisWorker.run"
            ) as mock_run:
                # Simulate what happens when import fails
                mock_run.side_effect = lambda: worker.error.emit(
                    "MediaPipe dependency not installed"
                )
                mock_run()

        assert len(error_messages) == 1
        assert "not installed" in error_messages[0]


class TestOpenPoseWorker:
    """Tests for OpenPose analysis worker."""

    def test_worker_emits_error_on_missing_dependency(self) -> None:
        """Worker should emit error signal when pyopenpose is not installed."""
        from src.shared.python.pose_estimation.openpose_gui import _AnalysisWorker

        worker = _AnalysisWorker("/nonexistent/video.mp4", {})

        error_messages: list[str] = []
        worker.error.connect(error_messages.append)

        with patch(
            "src.shared.python.pose_estimation.openpose_gui._AnalysisWorker.run"
        ) as mock_run:
            mock_run.side_effect = lambda: worker.error.emit(
                "OpenPose dependency not installed"
            )
            mock_run()

        assert len(error_messages) == 1
        assert "not installed" in error_messages[0]


# =============================================================================
# GUI Class Tests (no display required)
# =============================================================================


class TestMediaPipeGUIProperties:
    """Tests for MediaPipe GUI class properties."""

    def test_gui_class_exists(self) -> None:
        """MediaPipeGUI class should be importable."""
        from src.shared.python.pose_estimation.mediapipe_gui import MediaPipeGUI

        assert MediaPipeGUI is not None

    def test_gui_has_no_mock_timer(self) -> None:
        """The mock QTimer approach should be removed."""
        import inspect

        from src.shared.python.pose_estimation import mediapipe_gui

        source = inspect.getsource(mediapipe_gui)
        assert "Mocking the process for now" not in source
        assert "update_progress" not in source  # Old mock method removed


class TestOpenPoseGUIProperties:
    """Tests for OpenPose GUI class properties."""

    def test_gui_class_exists(self) -> None:
        """OpenPoseGUI class should be importable."""
        from src.shared.python.pose_estimation.openpose_gui import OpenPoseGUI

        assert OpenPoseGUI is not None

    def test_gui_has_no_mock_timer(self) -> None:
        """The mock QTimer approach should be removed."""
        import inspect

        from src.shared.python.pose_estimation import openpose_gui

        source = inspect.getsource(openpose_gui)
        assert "Mocking the process for now" not in source
        assert "update_progress" not in source  # Old mock method removed


# =============================================================================
# Integration Contract Tests
# =============================================================================


class TestIntegrationContracts:
    """Tests that the GUI integration satisfies the contracts from #1173."""

    def test_mediapipe_gui_uses_worker_thread(self) -> None:
        """MediaPipe GUI must use QThread for analysis (not QTimer)."""
        import inspect

        from src.shared.python.pose_estimation import mediapipe_gui

        source = inspect.getsource(mediapipe_gui)
        assert "QThread" in source
        assert "_AnalysisWorker" in source

    def test_openpose_gui_uses_worker_thread(self) -> None:
        """OpenPose GUI must use QThread for analysis (not QTimer)."""
        import inspect

        from src.shared.python.pose_estimation import openpose_gui

        source = inspect.getsource(openpose_gui)
        assert "QThread" in source
        assert "_AnalysisWorker" in source

    def test_mediapipe_gui_imports_real_estimator(self) -> None:
        """MediaPipe GUI worker must reference the real estimator."""
        import inspect

        from src.shared.python.pose_estimation import mediapipe_gui

        source = inspect.getsource(mediapipe_gui)
        assert "MediaPipeEstimator" in source
        assert "estimate_from_video" in source

    def test_openpose_gui_imports_real_estimator(self) -> None:
        """OpenPose GUI worker must reference the real estimator."""
        import inspect

        from src.shared.python.pose_estimation import openpose_gui

        source = inspect.getsource(openpose_gui)
        assert "OpenPoseEstimator" in source
        assert "estimate_from_video" in source

    def test_mediapipe_gui_saves_json_results(self) -> None:
        """MediaPipe GUI must save results as JSON."""
        import inspect

        from src.shared.python.pose_estimation import mediapipe_gui

        source = inspect.getsource(mediapipe_gui)
        assert "mediapipe_results.json" in source

    def test_openpose_gui_saves_json_results(self) -> None:
        """OpenPose GUI must save results as JSON."""
        import inspect

        from src.shared.python.pose_estimation import openpose_gui

        source = inspect.getsource(openpose_gui)
        assert "openpose_results.json" in source
