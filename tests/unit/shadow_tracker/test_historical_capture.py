"""Historical capture contracts; synthetic observations test software only."""

from fractions import Fraction

import numpy as np
import pytest

from src.shared.python.pose_estimation.interface import PoseEstimationResult
from src.shared.python.shadow_tracker.historical_capture import (
    CaptureWindow,
    observation_record,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "start,end", [(-1, 2), (2, 2), (3, 2), (0, float("inf")), (float("nan"), 1)]
)
def test_window_rejects_invalid_interval(start: float, end: float) -> None:
    with pytest.raises(ValueError):
        CaptureWindow(start_s=start, end_s=end)


def test_window_is_half_open() -> None:
    window = CaptureWindow(start_s=1, end_s=2)
    assert not window.contains(Fraction(9, 10))
    assert window.contains(Fraction(1))
    assert not window.contains(Fraction(2))
    decimal_window = CaptureWindow(start_s=0.2, end_s=0.5)
    assert decimal_window.contains(Fraction(1, 5))
    assert not decimal_window.contains(Fraction(1, 2))


def test_missing_detection_has_no_invented_landmarks() -> None:
    result = PoseEstimationResult({}, 0.0, 123.0)
    record = observation_record(result)
    assert record["status"] == "missing"
    assert record["landmarks"] == {}
    assert record["physical_time_s"] is None


def test_observation_preserves_image_coordinates_and_missingness() -> None:
    result = PoseEstimationResult(
        {}, 0.8, 123.0, {"wrist": np.array([0.2, 0.4, -0.1])}, {"wrist": 0.6}
    )
    record = observation_record(result)
    assert record["landmarks"]["wrist"] == {"x": 0.2, "y": 0.4, "visibility": 0.6}
    assert record["coordinate_system"] == "normalized_image_xy"
    assert "joint_angles" not in record


@pytest.mark.parametrize("point", [np.array([float("nan"), 0, 0]), np.array([0])])
def test_invalid_detector_output_fails_closed(point: np.ndarray) -> None:
    with pytest.raises(ValueError):
        observation_record(
            PoseEstimationResult({}, 0.5, 0, {"wrist": point}, {"wrist": 0.5})
        )


def test_unknown_visibility_is_preserved() -> None:
    record = observation_record(
        PoseEstimationResult({}, 0.5, 0, {"wrist": np.array([0, 0, 0])})
    )
    assert record["landmarks"]["wrist"]["visibility"] is None


def test_streamed_capture_binds_frames_and_abstains_on_physics(tmp_path) -> None:
    av = pytest.importorskip("av")
    from src.shared.python.shadow_tracker.historical_capture import export_capture

    source = tmp_path / "source.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for _ in range(10):
            frame = av.VideoFrame.from_ndarray(
                np.zeros((32, 32, 3), dtype=np.uint8), format="bgr24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    class MissingEstimator:
        def estimate_from_image(self, image, timestamp_ms):
            return PoseEstimationResult({}, 0, 999)

    target = tmp_path / "capture"
    receipt = export_capture(
        source,
        target,
        CaptureWindow(start_s=0.2, end_s=0.5),
        subject_id="test-player",
        estimator=MissingEstimator(),
        detector_identity={"name": "synthetic-test-only"},
    )
    assert receipt["frame_count"] == 3
    assert receipt["detected_count"] == 0
    assert receipt["qualification"] == "image_observations_only"
    assert receipt["source"]["asset_id"].endswith(receipt["source"]["content_sha256"])
    import json

    rows = [
        json.loads(line)
        for line in (target / "observations.jsonl").read_text().splitlines()
    ]
    assert rows[0]["frame"]["physical_time_s"] is None
    assert rows[0]["frame"]["pts_ticks"] > 0
    assert rows[0]["frame"]["is_timing_exact"]
    assert len(rows[0]["frame"]["frame_sha256"]) == 64
    assert rows[0]["observation"]["landmarks"] == {}
    assert (target / rows[0]["image"]).is_file()
    from src.shared.python.shadow_tracker.ingestion import compute_frame_hash
    import cv2

    saved = cv2.imread(str(target / rows[0]["image"]))
    assert (
        compute_frame_hash(saved.tobytes(), decoder_name="pyav")
        == rows[0]["frame"]["frame_sha256"]
    )
    with pytest.raises(FileExistsError):
        export_capture(
            source,
            target,
            CaptureWindow(start_s=0, end_s=1),
            subject_id="test-player",
            estimator=MissingEstimator(),
            detector_identity={},
        )
