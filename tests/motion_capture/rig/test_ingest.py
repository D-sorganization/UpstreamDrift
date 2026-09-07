"""Ingest: recordings to per-view observations, with a fake estimator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.motion_capture.rig import __main__ as cli
from src.motion_capture.rig.bundle import build_index, write_bundle
from src.motion_capture.rig.ingest import (
    INGEST_INDEX_FILE,
    VIEW_OBSERVATIONS_SCHEMA_VERSION,
    FramePose,
    IngestIndex,
    ViewObservations,
    ingest_bundle,
    ingest_view,
    registry_estimator_factory,
)
from src.motion_capture.rig.plan import CameraBinding, CaptureMode, RigPlan
from src.motion_capture.rig.probe import RecordingProbe
from src.motion_capture.rig.recorder import RecordingResult
from src.shared.python.engine_core.engine_availability import skip_if_unavailable
from src.shared.python.pose_estimation.observations import DetectorLayout

pytestmark = [pytest.mark.unit, skip_if_unavailable("cv2")]

W, H, N = 64, 48, 6


class FakeEstimator:
    """Two keypoints at fixed normalized positions; frame 2 has no pose."""

    layout = DetectorLayout(name="fake2", keypoint_names=("nose", "hip"))
    provenance: dict[str, Any] = {"estimator": "fake", "model_variant": None}
    closed = False

    def __init__(self) -> None:
        self.timestamps: list[int] = []

    def estimate(self, image: np.ndarray, timestamp_ms: int) -> FramePose | None:
        self.timestamps.append(timestamp_ms)
        if len(self.timestamps) == 3:
            return None
        return FramePose(
            keypoints_norm=np.array([[0.25, 0.5], [0.75, 1.0]]),
            confidence=np.array([0.9, 0.4]),
        )

    def close(self) -> None:
        self.closed = True


def _write_video(path: Path, frames: int = N) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), 30.0, (W, H))
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((H, W, 3), 20 + i, dtype=np.uint8))
    writer.release()


def _bundle(tmp_path: Path, *, break_second: bool = False) -> Path:
    plan = RigPlan(
        name="ingest-test",
        cameras=(
            CameraBinding(
                view="a", serial="1", mode=CaptureMode(width=W, height=H, fps=30)
            ),
            CameraBinding(
                view="b", serial="2", mode=CaptureMode(width=W, height=H, fps=30)
            ),
        ),
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    a, b = bundle / "a_1.avi", bundle / "b_2.avi"
    _write_video(a)
    _write_video(b)
    results = [
        RecordingResult("1", a, 0, a.stat().st_size),
        RecordingResult("2", b, 1 if break_second else 0, b.stat().st_size),
    ]

    def probe(path: Path) -> RecordingProbe:
        return RecordingProbe(
            frames=N, duration_s=N / 30, width=W, height=H, nominal_fps=30.0
        )

    index = build_index(plan, results, N / 30, bundle, prober=probe)
    write_bundle(bundle, plan, index, started_utc="2026-09-06T21:00:00+00:00")
    return bundle


def test_ingest_view_scales_to_pixels_and_times_by_frame_index(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    from src.motion_capture.rig.bundle import load_bundle

    _, index, _ = load_bundle(bundle)
    entry = index.recordings[0]
    est = FakeEstimator()
    obs = ingest_view(entry, bundle / entry.file, est)
    assert isinstance(obs, ViewObservations)
    assert obs.schema_version == VIEW_OBSERVATIONS_SCHEMA_VERSION
    assert (obs.frames_total, obs.frames_with_pose) == (N, N - 1)
    assert obs.fps == pytest.approx(30.0) and (obs.width, obs.height) == (W, H)
    first = obs.frames[0]
    assert first["camera_id"] == "1" and first["time_s"] == 0.0
    assert first["keypoints_px"] == [[0.25 * W, 0.5 * H], [0.75 * W, 1.0 * H]]
    assert first["confidence"] == [0.9, 0.4]
    assert obs.frames[3]["time_s"] == pytest.approx(4 / 30)  # frame 2 had no pose
    assert est.timestamps[:3] == [0, 33, 67]
    assert obs.detector_layout == {"name": "fake2", "keypoint_names": ["nose", "hip"]}
    assert obs.provenance["estimator"] == "fake"
    assert obs.provenance["requested_mode"]["fps"] == 30


def test_ingest_view_honours_max_frames_and_preconditions(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    from src.motion_capture.rig.bundle import load_bundle

    _, index, _ = load_bundle(bundle)
    entry = index.recordings[0]
    obs = ingest_view(entry, bundle / entry.file, FakeEstimator(), max_frames=2)
    assert obs.frames_total == 2
    with pytest.raises(Exception, match="max_frames"):
        ingest_view(entry, bundle / entry.file, FakeEstimator(), max_frames=0)
    with pytest.raises(Exception, match="recording must exist"):
        ingest_view(entry, bundle / "missing.avi", FakeEstimator())


def test_ingest_bundle_writes_index_and_marks_broken_views_unavailable(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path, break_second=True)
    out = tmp_path / "obs"
    est = FakeEstimator()
    result = ingest_bundle(bundle, out, lambda: est)
    assert isinstance(result, IngestIndex) and result.plan_name == "ingest-test"
    by_view = {v.view: v for v in result.views}
    assert by_view["a"].status == "available" and by_view["a"].file == "a.json"
    assert by_view["a"].frames_with_pose == N - 1
    assert by_view["b"].status == "unavailable" and "returncode=1" in (
        by_view["b"].reason or ""
    )
    assert est.closed
    on_disk = json.loads((out / INGEST_INDEX_FILE).read_text(encoding="utf-8"))
    assert on_disk["views"][1]["status"] == "unavailable"
    assert "timing" in on_disk and on_disk["provenance"]["estimator"] == "fake"
    view_file = json.loads((out / "a.json").read_text(encoding="utf-8"))
    assert view_file["frames_with_pose"] == N - 1


def test_registry_factory_only_adapts_mediapipe() -> None:
    with pytest.raises(Exception, match="only the mediapipe estimator"):
        registry_estimator_factory("openpose")


def test_cli_ingest_uses_patched_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(cli, "registry_estimator_factory", lambda name: FakeEstimator)
    code = cli.main(["ingest", "--session", str(bundle), "--max-frames", "3"])
    assert code == 0
    index = json.loads(
        (bundle / "observations" / INGEST_INDEX_FILE).read_text(encoding="utf-8")
    )
    assert [v["status"] for v in index["views"]] == ["available", "available"]
    assert index["views"][0]["frames_total"] == 3
