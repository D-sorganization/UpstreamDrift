"""Single-view analysis (#9663) and joint reliability (#9662) on a synthetic track."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct.analytics2d import (
    analyze_session_2d,
    series_2d,
    summarize_view_2d,
)
from src.motion_capture.rig.reliability import (
    reliability_report,
    score,
    write_reliability,
)

pytestmark = pytest.mark.unit

NAMES = [
    "nose",
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


def _swing_payload(view: str = "face_on", fps: float = 60.0, frames: int = 120) -> dict:
    """A golfer-shaped 2-D track: still, backswing, fast downswing, still."""
    rng = np.random.default_rng(4)
    rows = []
    for t in range(frames):
        s = t / frames
        # hands: rest for 20 %, slow rise for 40 %, fast drop for 15 %, rest
        if s < 0.2:
            hx, hy = 500.0, 700.0
        elif s < 0.6:
            u = (s - 0.2) / 0.4
            hx, hy = 500.0 + 150.0 * u, 700.0 - 350.0 * u
        elif s < 0.75:
            u = (s - 0.6) / 0.15
            hx, hy = 650.0 - 200.0 * u, 350.0 + 350.0 * u
        else:
            hx, hy = 450.0, 700.0
        base = {
            "nose": (500, 150),
            "left_shoulder": (440, 250),
            "right_shoulder": (560, 250),
            "left_elbow": (420, 400),
            "right_elbow": (580, 400),
            "left_wrist": (hx - 10, hy),
            "right_wrist": (hx + 10, hy),
            "left_hip": (455, 480),
            "right_hip": (545, 480),
            "left_knee": (450, 650),
            "right_knee": (550, 650),
            "left_ankle": (445, 820),
            "right_ankle": (555, 820),
        }
        px = [list(base[n]) for n in NAMES]
        conf = [0.9] * len(NAMES)
        conf[NAMES.index("nose")] = 0.2  # the face is always weak here
        if t % 7 == 0:
            conf[NAMES.index("left_ankle")] = 0.1  # ankle drops out often
        px = (
            np.asarray(px, dtype=float) + rng.normal(0, 0.5, (len(NAMES), 2))
        ).tolist()
        rows.append(
            {
                "camera_id": view,
                "time_s": t / fps,
                "keypoints_px": px,
                "confidence": conf,
            }
        )
    return {
        "view": view,
        "identity": view,
        "camera_id": view,
        "fps": fps,
        "width": 1000,
        "height": 900,
        "frames_total": frames,
        "frames_with_pose": frames,
        "detector_layout": {"name": "test13", "keypoint_names": NAMES},
        "frames": rows,
        "provenance": {"estimator": "mediapipe"},
    }


def test_series_and_events_from_one_view() -> None:
    payload = _swing_payload()
    series, scale = series_2d(payload)
    assert scale == pytest.approx(570.0, abs=5.0)  # nose excluded: shoulders to ankles
    summary = summarize_view_2d(payload)
    ev = summary.events
    assert ev.address_frame < ev.top_frame < ev.peak_speed_frame < ev.finish_frame
    assert 0.55 * 120 <= ev.top_frame <= 0.66 * 120
    assert summary.peak_hand_speed_bh_per_s > 2.0
    assert ev.tempo_ratio is not None and ev.tempo_ratio > 1.5
    assert "box heights" in summary.units


def _session(tmp_path: Path, sets: tuple[str, ...] = ("observations",)) -> Path:
    for name in sets:
        d = tmp_path / name
        d.mkdir()
        payload = _swing_payload()
        (d / "face_on.json").write_text(json.dumps(payload), encoding="utf-8")
        (d / "observations.json").write_text(
            json.dumps(
                {
                    "plan_name": "p",
                    "views": [
                        {
                            "view": "face_on",
                            "identity": "face_on",
                            "status": "available",
                            "file": "face_on.json",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    return tmp_path


def test_analyze_session_writes_per_view_json(tmp_path: Path) -> None:
    root = _session(tmp_path)
    written = analyze_session_2d(root)
    assert set(written) == {"face_on"}
    payload = json.loads(written["face_on"].read_text(encoding="utf-8"))
    assert payload["view"] == "face_on" and payload["events"]["top_frame"] > 0
    with pytest.raises(Exception, match="no observations"):
        analyze_session_2d(tmp_path / "nope")


def test_reliability_grades_and_recommends_exclusions(tmp_path: Path) -> None:
    root = _session(tmp_path, ("observations", "observations_openpose_dnn"))
    (root / "reconstruct").mkdir()
    (root / "reconstruct" / "clean_report.json").write_text(
        json.dumps(
            {
                "face_on": {
                    "frames": 120,
                    "rejected": [
                        {"joint": "right_wrist", "frame": i} for i in range(30)
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    report = reliability_report(root)
    assert report.observation_sets == ("observations", "observations_openpose_dnn")
    assert report.clean_report
    by_name = {j.joint: j for j in report.joints}
    assert by_name["nose"].grade == "weak" and "nose" in report.recommended_exclusions
    assert by_name["left_hip"].grade == "reliable"
    assert by_name["left_hip"].sets == 2
    assert by_name["right_wrist"].rejection_rate == pytest.approx(0.25)
    assert by_name["right_wrist"].score < by_name["left_wrist"].score
    assert report.joints[0].score >= report.joints[-1].score
    path = write_reliability(report, root)
    assert path.is_file() and path.with_suffix(".md").read_text(
        encoding="utf-8"
    ).startswith("| joint")
    assert score(None, None, None, None) is None
    assert score(1.0, 1.0, 0.0, 0.0) == pytest.approx(1.0)
