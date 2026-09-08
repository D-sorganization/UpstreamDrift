"""Fake-session tests for the RTMPose ONNX estimator (#9648).

No ``onnxruntime`` install is needed: the session is injected, and
``onnxruntime`` absence is asserted dynamically, so the suite runs both with
and without the optional dependency. The whole-frame letterbox is exercised
with hand-computed SimCC peak locations.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.motion_capture.rig.ingest import RegisteredFrameEstimator
from src.shared.python.core.error_utils import ModelError
from src.shared.python.engine_core.engine_availability import skip_if_unavailable
from src.shared.python.pose_estimation.registry import (
    EstimatorInfo,
    create_estimator,
    estimator_availability,
    get_estimator_info,
    implemented_estimator_types,
    register_estimator,
    unregister_estimator,
)

pytestmark = [pytest.mark.unit, skip_if_unavailable("cv2")]

_SPLIT = 2.0
FRAME = 64  # square test frame; scale = min(192, 256) / 64 = 3, no padding


def _input_coord(normalized: float, frame_axis: int) -> float:
    """Letterboxed model-input coordinate of a normalized frame coordinate."""
    return normalized * frame_axis * (256 / FRAME)


class FakeSimccSession:
    """A stand-in ONNX session whose SimCC peaks encode chosen keypoints.

    Args:
        peaks: ``{joint_index: ((x_loc, y_loc), value)}`` in split space,
            for a COCO-17 channel layout.
        input_size: ``(width, height)`` the estimator letterboxes to.
    """

    def __init__(
        self,
        peaks: dict[int, tuple[tuple[int, int], float]],
        input_size: tuple[int, int],
    ) -> None:
        self._peaks = peaks
        self._input_w, self._input_h = input_size
        self.calls: list[np.ndarray] = []

    def get_inputs(self) -> list[Any]:
        class _Input:
            name = "images"

        return [_Input()]

    def run(self, _names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append(feeds["images"])
        count = 17  # COCO-17 channel count the estimator requires
        simcc_x = np.zeros((1, count, round(self._input_w * _SPLIT)), dtype=np.float32)
        simcc_y = np.zeros((1, count, round(self._input_h * _SPLIT)), dtype=np.float32)
        for joint, ((x_loc, y_loc), value) in self._peaks.items():
            simcc_x[0, joint, x_loc] = value
            simcc_y[0, joint, y_loc] = value
        return [simcc_x, simcc_y]


def _fake_session(peaks: dict[int, tuple[tuple[int, int], float]]) -> FakeSimccSession:
    return FakeSimccSession(peaks, (192, 256))


# ---------------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------------


def test_rtmpose_onnx_is_registered_without_capture_source() -> None:
    info = get_estimator_info("rtmpose_onnx")
    assert info.capture_source is False
    assert info.probe_module == "onnxruntime"
    assert "rtmpose_onnx" in implemented_estimator_types()
    assert [j["name"] for j in info.skeleton][:3] == [
        "nose",
        "left_eye",
        "right_eye",
    ]  # COCO-17 detector order, like the other skeletons
    assert len(info.skeleton) == 17


def test_availability_tracks_onnxruntime_without_crashing() -> None:
    available, reason = estimator_availability("rtmpose_onnx")
    if find_spec("onnxruntime") is None:
        assert available is False
        assert reason is not None and "onnxruntime" in reason
    else:
        assert available is True and reason is None


def test_load_model_without_onnxruntime_raises_clean_import_error(
    tmp_path: Path,
) -> None:
    estimator = create_estimator("rtmpose_onnx")
    dummy = tmp_path / "fake.onnx"
    dummy.write_bytes(b"")
    if find_spec("onnxruntime") is None:
        with pytest.raises(ImportError, match="onnxruntime"):
            estimator.load_model(dummy)
    else:
        estimator.load_model(dummy)
        estimator.close()


def test_unknown_keypoint_set_rejected_at_construction() -> None:
    with pytest.raises(Exception, match="keypoint_set"):
        create_estimator("rtmpose_onnx", keypoint_set="coco18")


# ---------------------------------------------------------------------------
# Landmark naming
# ---------------------------------------------------------------------------


def test_coco17_landmark_names_in_official_order() -> None:
    estimator = create_estimator("rtmpose_onnx")
    names = [estimator.LANDMARK_MAP[i] for i in sorted(estimator.LANDMARK_MAP)]
    assert names == [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    assert estimator.LAYOUT_NAME == "rtmpose_coco17"


def test_halpe26_landmark_names_in_official_order() -> None:
    estimator = create_estimator("rtmpose_onnx", keypoint_set="halpe26")
    names = [estimator.LANDMARK_MAP[i] for i in sorted(estimator.LANDMARK_MAP)]
    assert len(names) == 26
    assert names[:17] == [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    assert names[17:] == [
        "head",
        "neck",
        "hip",
        "left_big_toe",
        "right_big_toe",
        "left_small_toe",
        "right_small_toe",
        "left_heel",
        "right_heel",
    ]
    assert estimator.LAYOUT_NAME == "rtmpose_halpe26"


# ---------------------------------------------------------------------------
# SimCC decode with a fake session (hand-computed peaks)
# ---------------------------------------------------------------------------


def _dummy_onnx(tmp_path: Path) -> Path:
    """A path that exists so load_model skips resolving the pinned weights."""
    dummy = tmp_path / "fake.onnx"
    dummy.write_bytes(b"")
    return dummy


def test_simcc_decode_maps_peaks_to_normalized_frame_coords(tmp_path: Path) -> None:
    # 64x64 frame -> scale 3 -> 192x192 content, centered with pad_y = 32.
    # nose at normalized (0.25, 0.5): frame (16, 32) -> input (48, 128)
    # -> split locs (96, 256). left_hip (0.5, 0.75): input (96, 176)
    # -> split locs (192, 352). right_hip peaks at 0.1, below the default
    # 0.3 score: unobserved.
    peaks = {
        0: ((96, 256), 0.9),
        11: ((192, 352), 0.8),
        12: ((10, 10), 0.1),
    }
    session = _fake_session(peaks)
    estimator = create_estimator("rtmpose_onnx", session_factory=lambda path: session)
    estimator.load_model(_dummy_onnx(tmp_path))
    assert estimator.is_ready()

    frame = np.zeros((FRAME, FRAME, 3), dtype=np.uint8)
    result = estimator.estimate_from_image(frame, timestamp_ms=33)
    assert result.raw_keypoints["nose"] == pytest.approx([0.25, 0.5])
    assert result.raw_keypoints["left_hip"] == pytest.approx([0.5, 0.75])
    assert "right_hip" not in result.raw_keypoints  # below min score: unobserved
    assert result.raw_confidences["right_hip"] == pytest.approx(0.1)
    assert result.confidence == pytest.approx(0.85)  # mean of the detected peaks
    assert result.timestamp == pytest.approx(0.033)
    # The session received one (1, 3, 256, 192) float32 blob.
    [blob] = session.calls
    assert blob.shape == (1, 3, 256, 192) and blob.dtype == np.float32


def test_load_model_without_weights_names_the_download_command(tmp_path: Path) -> None:
    estimator = create_estimator("rtmpose_onnx")
    with pytest.raises(ModelError, match="rtmpose_models"):
        estimator.load_model(tmp_path)


# ---------------------------------------------------------------------------
# RegisteredFrameEstimator must honour instance-level layouts (#9648)
# ---------------------------------------------------------------------------


class _InstanceLayoutEstimator:
    """Declares LANDMARK_MAP on the instance so a variant can pick its set."""

    def __init__(self, **_options: Any) -> None:
        self.LANDMARK_MAP = {0: "hip", 1: "neck"}
        self.LAYOUT_NAME = "fake_instance_layout"
        self.loaded = False

    def load_model(self, model_path: Path | None = None) -> None:
        self.loaded = True

    def estimate_from_image(self, image: np.ndarray, _ts: int | None = None) -> Any: ...

    def estimate_from_video(self, _video: Path) -> list[Any]:
        return []

    def close(self) -> None: ...


def test_registered_frame_estimator_reads_the_instance_layout() -> None:
    register_estimator(
        EstimatorInfo(
            name="fake_instance_layout",
            display_name="Fake",
            description="instance-layout probe",
            probe_module="json",
            install_hint="",
            factory=_InstanceLayoutEstimator,
        )
    )
    try:
        adapter = RegisteredFrameEstimator("fake_instance_layout")
        assert adapter.layout.name == "fake_instance_layout"
        assert list(adapter.layout.keypoint_names) == ["hip", "neck"]
    finally:
        unregister_estimator("fake_instance_layout")


# ---------------------------------------------------------------------------
# Pinned model metadata (PENDING OWNER APPROVAL)
# ---------------------------------------------------------------------------


def test_pinned_model_specs_carry_official_url_and_size() -> None:
    from src.shared.python.pose_estimation import rtmpose_models

    for keypoint_set in ("coco17", "halpe26"):
        spec = rtmpose_models.RTMPOSE_MODELS[keypoint_set]
        assert spec.url.startswith("https://download.openmmlab.com/mmpose/")
        assert spec.size_bytes > 0
        assert spec.sha256 is None  # PENDING OWNER APPROVAL: unverified download
        assert spec.filename.endswith(".onnx")
