"""Tests for ``SwingAnalyzer.analyze_video`` (issue #8883).

Covers the video-ingestion path that wires the already-tested head-
stability math in ``analyzer.py`` to a real video file via the
MediaPipe pose-estimation registry, instead of stopping at a GUI
placeholder. The MediaPipe estimator itself is mocked so these tests do
not require the optional dependency to be installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.pose_estimation.interface import PoseEstimationResult
from src.tools.video_analyzer.analyzer import SwingAnalyzer

pytestmark = [pytest.mark.unit]


@pytest.fixture()
def analyzer() -> SwingAnalyzer:
    return SwingAnalyzer()


def test_analyze_video_missing_file_raises(analyzer, tmp_path) -> None:  # noqa: ANN001
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError, match="not found"):
        analyzer.analyze_video(missing)


def test_analyze_video_unavailable_estimator_raises_runtime_error(
    analyzer, tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    """When MediaPipe is not installed, the failure is explicit, not silent."""
    video = tmp_path / "swing.mp4"
    video.write_bytes(b"stand-in for a real container")

    import src.shared.python.pose_estimation.registry as registry

    monkeypatch.setattr(
        registry,
        "estimator_availability",
        lambda name: (False, "install mediapipe>=0.10"),
    )

    with pytest.raises(RuntimeError, match="install mediapipe"):
        analyzer.analyze_video(video)


def test_analyze_video_computes_head_stability_from_estimator_output(
    analyzer, tmp_path, monkeypatch
) -> None:  # noqa: ANN001
    """A stationary nose across frames scores near-perfect stability."""
    video = tmp_path / "swing.mp4"
    video.write_bytes(b"stand-in for a real container")

    import src.shared.python.pose_estimation.registry as registry

    monkeypatch.setattr(registry, "estimator_availability", lambda name: (True, None))

    fake_result = PoseEstimationResult(
        joint_angles={},
        confidence=0.9,
        timestamp=0.0,
        raw_keypoints={"nose": np.array([0.5, 0.5, 0.0])},
        raw_confidences={"nose": 0.9},
    )
    fake_estimator = MagicMock()
    fake_estimator.estimate_from_video.return_value = [fake_result, fake_result]
    monkeypatch.setattr(
        registry, "create_estimator", lambda name, **opts: fake_estimator
    )

    metrics = analyzer.analyze_video(video)

    fake_estimator.load_model.assert_called_once()
    fake_estimator.estimate_from_video.assert_called_once_with(video)
    assert metrics.head_stability == pytest.approx(100.0, abs=1.0)


def test_analyze_video_detects_head_drift(analyzer, tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """A moving nose lowers the head-stability score below perfect."""
    video = tmp_path / "swing.mp4"
    video.write_bytes(b"stand-in for a real container")

    import src.shared.python.pose_estimation.registry as registry

    monkeypatch.setattr(registry, "estimator_availability", lambda name: (True, None))

    address = PoseEstimationResult(
        joint_angles={},
        confidence=0.9,
        timestamp=0.0,
        raw_keypoints={"nose": np.array([0.5, 0.5, 0.0])},
        raw_confidences={"nose": 0.9},
    )
    drifted = PoseEstimationResult(
        joint_angles={},
        confidence=0.9,
        timestamp=0.1,
        raw_keypoints={"nose": np.array([0.6, 0.5, 0.0])},
        raw_confidences={"nose": 0.9},
    )
    fake_estimator = MagicMock()
    fake_estimator.estimate_from_video.return_value = [address, drifted]
    monkeypatch.setattr(
        registry, "create_estimator", lambda name, **opts: fake_estimator
    )

    metrics = analyzer.analyze_video(video)

    assert metrics.head_stability < 100.0
