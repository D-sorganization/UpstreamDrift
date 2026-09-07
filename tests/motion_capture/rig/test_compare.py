"""Detector comparison metrics: coverage, jitter, agreement, report."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.rig.compare import (
    DetectorSeries,
    agreement,
    compare_view,
    joint_metrics,
)

pytestmark = pytest.mark.unit

NAMES = ["nose", "left_hip", "left_ankle"]


def _obs(points, confs, fps=10.0, names=NAMES):
    frames = []
    for i, (p, c) in enumerate(zip(points, confs, strict=True)):
        frames.append(
            {"time_s": i / fps, "keypoints_px": p, "confidence": c, "camera_id": "x"}
        )
    return {
        "detector_layout": {"name": "t", "keypoint_names": names},
        "fps": fps,
        "frames_total": len(points),
        "frames": frames,
    }


def _still(n: int, jitter: float = 0.0, conf: float = 0.9):
    rng = np.random.default_rng(0)
    pts, cf = [], []
    for _ in range(n):
        base = np.array([[100.0, 100.0], [100.0, 300.0], [100.0, 500.0]])
        pts.append((base + rng.normal(0, jitter, base.shape)).tolist())
        cf.append([conf, conf, conf])
    return pts, cf


def test_coverage_counts_only_confident_frames() -> None:
    pts, cf = _still(4)
    cf[1][0] = 0.1  # nose low confidence in frame 1
    series = DetectorSeries(_obs(pts, cf))
    m = joint_metrics(series, "nose")
    assert m.frames == 4 and m.coverage == pytest.approx(0.75)
    assert m.mean_confidence == pytest.approx(0.9)


def test_jitter_is_normalised_by_box_height() -> None:
    pts, cf = _still(3)
    pts[1][0] = [140.0, 100.0]  # nose moved 40 px; box height is 400 px
    pts[2][0] = [140.0, 100.0]
    series = DetectorSeries(_obs(pts, cf))
    m = joint_metrics(series, "nose")
    assert m.jitter == pytest.approx(np.median([0.1, 0.0]))


def test_missing_frames_and_unknown_joint_are_handled() -> None:
    pts, cf = _still(3)
    obs = _obs(pts, cf)
    obs["frames"].pop(1)  # frame 1 has no pose at all
    series = DetectorSeries(obs)
    assert joint_metrics(series, "nose").coverage == pytest.approx(2 / 3)
    m = joint_metrics(series, "right_wrist")
    assert m.coverage == 0.0 and m.mean_confidence is None and m.jitter is None


def test_agreement_between_two_detectors() -> None:
    pts_a, cf = _still(3)
    pts_b = [[[x + 20.0, y] for x, y in frame] for frame in pts_a]
    a, b = DetectorSeries(_obs(pts_a, cf)), DetectorSeries(_obs(pts_b, cf))
    n, med = agreement(a, b, "nose")
    assert n == 3 and med == pytest.approx(20 / 400)


def test_compare_view_report_and_markdown() -> None:
    pts, cf = _still(3)
    a, b = DetectorSeries(_obs(pts, cf)), DetectorSeries(_obs(pts, cf))
    report = compare_view("cam_b", {"mediapipe": a, "openpose_dnn": b}, joints=NAMES)
    assert report.detectors == ("mediapipe", "openpose_dnn")
    assert report.agreement["nose"]["median"] == pytest.approx(0.0)
    md = report.markdown()
    assert md.startswith("| Joint |") and "| nose |" in md and "agree" in md
    single = compare_view("cam_b", {"mediapipe": a}, joints=NAMES)
    assert single.agreement == {} and "agree" not in single.markdown()
    with pytest.raises(Exception, match="min_confidence"):
        joint_metrics(a, "nose", min_confidence=2.0)
