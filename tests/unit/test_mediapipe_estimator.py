"""MediaPipeEstimator on the Tasks API, with the API fully faked.

No model file, no network, no real mediapipe call: ``_tasks_api`` is patched to
return a fake ``vision`` namespace and ``BaseOptions``, and model resolution is
patched to a temp path. The behavioural contract (landmark naming, confidence,
joint angles, smoothing, monotonic VIDEO timestamps) is what is pinned.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.core.contracts import StateError
from src.shared.python.core.error_utils import ModelError
from src.shared.python.engine_core.engine_availability import skip_if_unavailable

pytestmark = [pytest.mark.unit, skip_if_unavailable("cv2")]

from src.shared.python.pose_estimation import mediapipe_estimator  # noqa: E402


def _landmarks(n: int = 33) -> list[MagicMock]:
    out = []
    for _ in range(n):
        lm = MagicMock()
        lm.x, lm.y, lm.z, lm.visibility = 0.5, 0.5, 0.0, 0.9
        out.append(lm)
    return out


@pytest.fixture
def fake_api() -> tuple[MagicMock, MagicMock, MagicMock]:
    """``(vision, BaseOptions, detector)`` fakes shaped like the Tasks API."""
    vision = MagicMock()
    vision.RunningMode.VIDEO = "VIDEO"
    detector = MagicMock()
    result = MagicMock()
    result.pose_landmarks = [_landmarks()]
    detector.detect_for_video.return_value = result
    vision.PoseLandmarker.create_from_options.return_value = detector
    return vision, MagicMock(name="BaseOptions"), detector


@pytest.fixture
def estimator(
    fake_api: tuple[MagicMock, MagicMock, MagicMock], tmp_path: Path
) -> Generator[mediapipe_estimator.MediaPipeEstimator, None, None]:
    vision, base_options, _ = fake_api
    model = tmp_path / "pose_landmarker_full.task"
    model.write_bytes(b"model")
    mp_fake = MagicMock()
    mp_fake.ImageFormat.SRGB = "SRGB"
    with (
        patch.object(mediapipe_estimator, "MEDIAPIPE_AVAILABLE", True),
        patch.object(mediapipe_estimator, "mp", mp_fake),
        patch.object(
            mediapipe_estimator, "_tasks_api", return_value=(vision, base_options)
        ),
        patch.object(mediapipe_estimator, "resolve_pose_model", return_value=model),
        patch.object(mediapipe_estimator, "cv2") as cv2_fake,
    ):
        cv2_fake.cvtColor.side_effect = lambda img, _code: img
        yield mediapipe_estimator.MediaPipeEstimator(min_detection_confidence=0.7)


def test_init_validates_confidences_and_fps() -> None:
    with pytest.raises(Exception, match="min_detection_confidence"):
        mediapipe_estimator.MediaPipeEstimator(min_detection_confidence=1.5)
    with pytest.raises(Exception, match="nominal_fps"):
        mediapipe_estimator.MediaPipeEstimator(nominal_fps=0)


def test_load_model_builds_video_mode_landmarker(
    estimator: mediapipe_estimator.MediaPipeEstimator,
    fake_api: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    vision, base_options, _ = fake_api
    assert estimator.pose_detector is None and estimator._is_loaded is False
    estimator.load_model()
    assert estimator._is_loaded is True
    assert estimator.model_path is not None and estimator.model_path.name.endswith(
        ".task"
    )
    base_options.assert_called_once_with(model_asset_path=str(estimator.model_path))
    kwargs = vision.PoseLandmarkerOptions.call_args.kwargs
    assert kwargs["running_mode"] == "VIDEO"
    assert kwargs["num_poses"] == 1
    assert kwargs["min_pose_detection_confidence"] == 0.7
    vision.PoseLandmarker.create_from_options.assert_called_once()


def test_estimate_from_image_maps_landmarks_and_confidence(
    estimator: mediapipe_estimator.MediaPipeEstimator,
) -> None:
    estimator.enable_temporal_smoothing = False
    estimator.load_model()
    result = estimator.estimate_from_image(np.zeros((100, 100, 3), dtype=np.uint8))
    assert result.confidence == pytest.approx(0.9)
    assert result.raw_keypoints is not None and "nose" in result.raw_keypoints
    assert result.raw_keypoints["nose"][0] == 0.5
    assert len(result.raw_keypoints) == 33


def test_estimate_requires_loaded_model_and_bgr_shape(
    estimator: mediapipe_estimator.MediaPipeEstimator,
) -> None:
    with pytest.raises(StateError):
        estimator.estimate_from_image(np.zeros((4, 4, 3), dtype=np.uint8))
    estimator.load_model()
    with pytest.raises(Exception, match="HxWx3"):
        estimator.estimate_from_image(np.zeros((4, 4), dtype=np.uint8))


def test_video_mode_timestamps_are_strictly_increasing(
    estimator: mediapipe_estimator.MediaPipeEstimator,
    fake_api: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    _, _, detector = fake_api
    estimator.nominal_fps = 50.0
    estimator.load_model()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for _ in range(3):
        estimator.estimate_from_image(image)
    estimator.estimate_from_image(image, timestamp_ms=10)  # would go backwards
    stamps = [call.args[1] for call in detector.detect_for_video.call_args_list]
    assert stamps == [0, 20, 40, 41]
    estimator.reset_temporal_state()
    estimator.estimate_from_image(image)
    assert detector.detect_for_video.call_args_list[-1].args[1] == 0


def test_no_pose_yields_empty_result(
    estimator: mediapipe_estimator.MediaPipeEstimator,
    fake_api: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    _, _, detector = fake_api
    detector.detect_for_video.return_value.pose_landmarks = []
    estimator.load_model()
    result = estimator.estimate_from_image(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result.confidence == 0.0 and result.joint_angles == {}


def test_joint_angles_calculation(
    estimator: mediapipe_estimator.MediaPipeEstimator,
    fake_api: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    _, _, detector = fake_api
    estimator.enable_temporal_smoothing = False
    estimator.load_model()
    landmarks = detector.detect_for_video.return_value.pose_landmarks[0]
    for lm in landmarks:
        lm.x = lm.y = lm.z = 0.0
        lm.visibility = 1.0
    landmarks[12].y = 1.0  # right shoulder above
    landmarks[16].x = 1.0  # right wrist to the side; elbow (14) at origin
    result = estimator.estimate_from_image(np.zeros((100, 100, 3), dtype=np.uint8))
    assert np.isclose(result.joint_angles["right_elbow_flexion"], np.pi / 2, atol=1e-5)


def test_temporal_smoothing_runs_across_frames(
    estimator: mediapipe_estimator.MediaPipeEstimator,
) -> None:
    estimator.enable_temporal_smoothing = True
    estimator.load_model()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    first = estimator.estimate_from_image(image)
    second = estimator.estimate_from_image(image)
    assert first.raw_keypoints is not None and second.raw_keypoints is not None
    assert len(estimator.kalman_filters) == 33
    estimator.reset_temporal_state()
    assert estimator.kalman_filters == {}


def test_close_is_idempotent(
    estimator: mediapipe_estimator.MediaPipeEstimator,
    fake_api: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    _, _, detector = fake_api
    estimator.load_model()
    estimator.close()
    estimator.close()
    detector.close.assert_called_once()
    assert estimator._is_loaded is False


def test_missing_mediapipe_raises_import_error() -> None:
    with (
        patch.object(mediapipe_estimator, "MEDIAPIPE_AVAILABLE", False),
        patch.object(mediapipe_estimator, "mp", None),
    ):
        with pytest.raises(ImportError, match="Tasks API"):
            mediapipe_estimator.MediaPipeEstimator().load_model()


def test_missing_model_propagates_model_error(
    fake_api: tuple[MagicMock, MagicMock, MagicMock],
) -> None:
    vision, base_options, _ = fake_api
    with (
        patch.object(mediapipe_estimator, "MEDIAPIPE_AVAILABLE", True),
        patch.object(mediapipe_estimator, "mp", MagicMock()),
        patch.object(mediapipe_estimator, "cv2", MagicMock()),
        patch.object(
            mediapipe_estimator, "_tasks_api", return_value=(vision, base_options)
        ),
        patch.object(
            mediapipe_estimator,
            "resolve_pose_model",
            side_effect=ModelError("pose_landmarker_full.task", "resolve", "no model"),
        ),
    ):
        with pytest.raises(ModelError, match="no model"):
            mediapipe_estimator.MediaPipeEstimator().load_model()
