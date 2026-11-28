"""TDD tests for Track A historical-capture pose pipeline (Refs #15837)."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import numpy as np
import pytest

from src.shared.python.pose_estimation.interface import PoseEstimationResult
from src.shared.python.shadow_tracker import track_a_pose_pipeline as tp

pytestmark = pytest.mark.unit


class _Est:
    def __init__(self, results: list[PoseEstimationResult]) -> None:
        self._r, self._i = list(results), 0

    def estimate_from_image(self, image: np.ndarray) -> PoseEstimationResult:
        del image
        res = self._r[self._i]
        self._i += 1
        return res


def _fr(n: int = 2) -> list[np.ndarray]:
    return [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(n)]


def _res(
    kp: dict[str, list[float]] | None = None, cf: dict[str, float] | None = None
) -> PoseEstimationResult:
    arrs = {k: np.array(v, dtype=float) for k, v in (kp or {}).items()}
    return PoseEstimationResult({}, 0.8 if kp else 0.0, 0.5, arrs, cf)


def _fail(exc: Exception) -> Any:
    return lambda: (_ for _ in ()).throw(exc)


def test_sanitize_missing_low_vis_and_detected() -> None:
    exp = ("nose", "left_wrist", "right_wrist")
    m = tp.sanitize_track_a_frame(_res(), frame_index=4, expected_keypoints=exp)
    assert (m.frame_index, m.status, m.keypoint_coverage) == (4, "missing", 0.0)
    assert m.missing_keypoints == exp and m.physical_time_s is None
    assert (m.timebase, m.qualification) == ("unverified", "image_observations_only")
    assert all(m.landmarks[k]["x"] is None for k in exp)

    kp = {"nose": [0.5, 0.25], "left_wrist": [0.3, 0.6]}
    cf = {"nose": 0.9, "left_wrist": 0.02, "right_wrist": 0.01}
    p = tp.sanitize_track_a_frame(_res(kp, cf), expected_keypoints=(*exp, "mid_hip"))
    assert p.status == "partial" and p.keypoint_coverage == pytest.approx(0.25)
    assert p.low_visibility_keypoints == ("left_wrist",)
    assert p.missing_keypoints == ("right_wrist", "mid_hip")

    d = tp.sanitize_track_a_frame(_res({"nose": [0.5, 0.2]}, {"nose": 0.9}))
    assert d.status == "detected" and d.keypoint_coverage == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("kp", "vis"),
    [
        ({"nose": [float("nan"), 0.5]}, 0.9),
        ({"nose": [0.5]}, 0.9),
        ({"nose": [0.5, 0.5]}, -0.1),
        ({"nose": [0.5, 0.5]}, 1.1),
        ({}, float("nan")),
    ],
)
def test_sanitize_rejects_invalid_inputs(
    kp: dict[str, list[float]], vis: float
) -> None:
    with pytest.raises(ValueError):
        tp.sanitize_track_a_frame(_res(kp, {"nose": vis}))


def test_setup_cap_fallback_and_separate_statuses() -> None:
    assert tp.OPENPOSE_SETUP_CAP_S == 7200.0
    ticks = iter([0.0, 7200.5, 7201.0, 7201.2, 7201.2, 7201.5])
    b25 = [_res({"nose": [0.5, 0.2]}, {"nose": 0.8, "neck": 0.01}), _res()]
    rtm = [_res({"nose": [0.5, 0.2]}, {"nose": 0.9})] * 2
    rec = tp.run_track_a_pipeline(
        subject_id="golfer-a",
        frames=_fr(2),
        monotonic_fn=lambda: next(ticks),
        openpose_native_probe=lambda: _Est(b25),
        openpose_body25_onnx_factory=lambda: _Est(b25),
        rtmpose_factory=lambda: _Est(rtm),
    )
    nat, b25_s, rtm_s = rec.openpose_native, rec.openpose_body25_onnx, rec.rtmpose
    assert rec.fallback_triggered and nat.status == "blocked"
    assert "7200" in nat.reason
    assert b25_s.status == "ok" and b25_s.detected_frames == 1
    assert b25_s.mean_coverage == pytest.approx(0.25)
    assert rtm_s.status == "ok" and rtm_s.detected_frames == 2
    assert rec.active_backends == ("openpose_body25_onnx", "rtmpose")

    b25_1 = _Est([_res({"nose": [0.5, 0.3]})])
    part = tp.run_track_a_pipeline(
        subject_id="golfer-b",
        frames=_fr(1),
        openpose_body25_onnx_factory=lambda: b25_1,
        rtmpose_factory=_fail(ImportError("ort missing")),
    )
    assert part.openpose_native.status == "blocked"
    assert part.openpose_body25_onnx.status == "ok"
    assert part.rtmpose.status == "blocked" and "ort" in part.rtmpose.reason


def test_native_ok_and_default_env_blocked(tmp_path: Any) -> None:
    ticks = iter([10.0, 12.5])
    ok = tp.run_track_a_pipeline(
        subject_id="golfer-c",
        frames=_fr(1),
        monotonic_fn=lambda: next(ticks),
        openpose_native_probe=lambda: _Est([_res({"nose": [0.5, 0.2]})]),
    )
    assert not ok.fallback_triggered and ok.openpose_native.status == "ok"
    assert ok.openpose_body25_onnx.status == "blocked"
    assert ok.rtmpose.status == "blocked"

    stub = tmp_path / "clip.mp4"
    stub.write_bytes(b"stub")
    env = tp.run_track_a_pipeline(
        subject_id="env", source_path=stub, frame_decoder=lambda _p: _fr(1)
    )
    assert env.fallback_triggered and env.openpose_native.status == "blocked"
    assert env.openpose_body25_onnx.status == "blocked"
    assert env.rtmpose.status == "blocked"
    with pytest.raises(ImportError, match="av"):
        tp.run_track_a_pipeline(subject_id="no-av", source_path=stub)


def test_receipt_contract_json_and_invariants() -> None:
    rec = tp.run_track_a_pipeline(
        subject_id="archive-01",
        frames=_fr(1),
        openpose_body25_onnx_factory=_fail(FileNotFoundError("w")),
        rtmpose_factory=_fail(ImportError("ort")),
    )
    assert rec.schema_version == tp.HISTORICAL_CAPTURE_SCHEMA_VERSION
    assert not rec.physical_time_verified and rec.timebase == "unverified"
    assert rec.qualification == "image_observations_only"
    dec = json.loads(json.dumps(rec.to_dict(), allow_nan=False))
    assert dec["schema_version"] == "historical-capture/1.0.0"
    assert not dec["physical_time_verified"]
    with pytest.raises(ValueError):
        replace(rec, physical_time_verified=True)
    with pytest.raises(ValueError):
        replace(rec, schema_version="wrong/1.0.0")
