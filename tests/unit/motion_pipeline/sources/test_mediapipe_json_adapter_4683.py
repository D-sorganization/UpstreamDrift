"""Regression tests for MediaPipe JSON adapter (#4683).

The adapter previously rejected the canonical export form where each
landmark is encoded as a list ``[x, y, z, visibility, presence]`` instead
of a dict. These tests pin both the shipped golden fixture and a hand-built
canonical sample.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources import load_any
from src.shared.python.motion_pipeline.sources.mediapipe_json_adapter import (
    MediaPipeJSONAdapter,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
GOLDEN = REPO_ROOT / "tests" / "data" / "motion_pipeline" / "golden" / "mediapipe.json"


def test_load_shipped_golden_fixture() -> None:
    seq = load_any(GOLDEN)
    assert len(seq.frames) == 30
    assert all(f.schema_name == "MediaPipe_33" for f in seq.frames)
    assert all(len(f.keypoints) == 33 for f in seq.frames)
    # Monotonic timestamps and finite values.
    ts = [f.timestamp for f in seq.frames]
    assert ts == sorted(ts)
    for f in seq.frames:
        for kp in f.keypoints:
            assert math.isfinite(kp.x) and math.isfinite(kp.y)
            assert 0.0 <= kp.confidence <= 1.0


def test_canonical_array_landmarks(tmp_path: Path) -> None:
    """Hand-built canonical MediaPipe with [x,y,z,vis,presence] arrays."""
    payload = {
        "schema": "MediaPipe_33",
        "fps": 30.0,
        "frames": [
            {
                "frame_index": i,
                "timestamp": i / 30.0,
                "landmarks": [[0.5, 0.5, 0.0, 0.9, 0.95] for _ in range(33)],
            }
            for i in range(3)
        ],
    }
    p = tmp_path / "mediapipe.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    seq = MediaPipeJSONAdapter().load(p)
    assert len(seq.frames) == 3
    assert all(len(f.keypoints) == 33 for f in seq.frames)
    assert seq.frames[0].schema_name == "MediaPipe_33"
    assert seq.frames[0].keypoints[0].confidence == pytest.approx(0.9)


def test_canonical_dict_landmarks(tmp_path: Path) -> None:
    """Dict form ``{x,y,z,visibility}`` still parses (back-compat)."""
    payload = {
        "schema": "MediaPipe_33",
        "fps": 60.0,
        "frames": [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "landmarks": [
                    {"x": 0.1, "y": 0.2, "z": 0.0, "visibility": 0.8} for _ in range(33)
                ],
            }
        ],
    }
    p = tmp_path / "mediapipe.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    seq = MediaPipeJSONAdapter().load(p)
    assert seq.frames[0].keypoints[0].confidence == pytest.approx(0.8)


def test_pose_landmarks_alias(tmp_path: Path) -> None:
    """Some MediaPipe exports nest under ``pose_landmarks``."""
    payload = {
        "schema": "MediaPipe_33",
        "fps": 30.0,
        "frames": [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "pose_landmarks": [[0.0, 0.0, 0.0, 1.0] for _ in range(33)],
            }
        ],
    }
    p = tmp_path / "mp.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    seq = MediaPipeJSONAdapter().load(p)
    assert len(seq.frames) == 1
    assert len(seq.frames[0].keypoints) == 33


def test_rejects_no_frames_key(tmp_path: Path) -> None:
    p = tmp_path / "mediapipe.json"
    p.write_text(json.dumps({"schema": "MediaPipe_33"}), encoding="utf-8")
    with pytest.raises(ValueError, match="frames"):
        MediaPipeJSONAdapter().load(p)


def test_rejects_empty_frames(tmp_path: Path) -> None:
    p = tmp_path / "mediapipe.json"
    p.write_text(
        json.dumps({"schema": "MediaPipe_33", "frames": [{"landmarks": []}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no usable frames"):
        MediaPipeJSONAdapter().load(p)
