"""Tests for the MediaPipe JSON adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources.mediapipe_json_adapter import (
    MediaPipeJSONAdapter,
)


def _frame(idx: int) -> dict:
    return {
        "frame_index": idx,
        "timestamp": idx / 30.0,
        "landmarks": [
            {"x": 0.5 + 0.01 * i, "y": 0.5, "z": 0.0, "visibility": 0.99}
            for i in range(33)
        ],
    }


def _write(tmp_path: Path) -> Path:
    payload = {
        "schema": "MediaPipe_33",
        "fps": 30.0,
        "frames": [_frame(0), _frame(1), _frame(2)],
    }
    p = tmp_path / "mediapipe_pose.json"
    p.write_text(json.dumps(payload))
    return p


def test_mediapipe_supports(tmp_path: Path) -> None:
    assert MediaPipeJSONAdapter.supports(_write(tmp_path)) is True


def test_mediapipe_metadata(tmp_path: Path) -> None:
    md = MediaPipeJSONAdapter().metadata(_write(tmp_path))
    assert md.frame_count == 3
    assert md.keypoint_schema == "MediaPipe_33"
    assert md.unit_system == "normalized"


def test_mediapipe_load(tmp_path: Path) -> None:
    seq = MediaPipeJSONAdapter().load_checked(_write(tmp_path))
    assert seq.num_frames == 3
    assert len(seq.frames[0].keypoints) == 33


def test_mediapipe_missing_frames_raises(tmp_path: Path) -> None:
    p = tmp_path / "mediapipe_pose.json"
    p.write_text(json.dumps({"schema": "MediaPipe_33", "fps": 30.0}))
    with pytest.raises(ValueError):
        MediaPipeJSONAdapter().load(p)
