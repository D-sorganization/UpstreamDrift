"""Tests for the OpenPose JSON adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources.openpose_json_adapter import (
    OpenPoseJSONAdapter,
)


def _person(num: int = 25) -> dict:
    flat: list[float] = []
    for i in range(num):
        flat.extend([float(i), float(i + 1), 0.9])
    return {"pose_keypoints_2d": flat}


def _write_singleframe(tmp_path: Path) -> Path:
    payload = {"version": 1.3, "people": [_person()]}
    p = tmp_path / "vid_000000_keypoints.json"
    p.write_text(json.dumps(payload))
    return p


def _write_multi(tmp_path: Path) -> Path:
    payload = [
        {"version": 1.3, "people": [_person()]},
        {"version": 1.3, "people": [_person()]},
    ]
    p = tmp_path / "vid_keypoints.json"
    p.write_text(json.dumps(payload))
    return p


def test_openpose_supports_singleframe(tmp_path: Path) -> None:
    assert OpenPoseJSONAdapter.supports(_write_singleframe(tmp_path)) is True


def test_openpose_metadata(tmp_path: Path) -> None:
    md = OpenPoseJSONAdapter().metadata(_write_multi(tmp_path))
    assert md.frame_count == 2
    assert md.keypoint_schema == "BODY_25"


def test_openpose_load(tmp_path: Path) -> None:
    seq = OpenPoseJSONAdapter().load_checked(_write_multi(tmp_path))
    assert seq.num_frames == 2
    assert seq.frames[0].schema_name == "BODY_25"
    assert len(seq.frames[0].keypoints) == 25


def test_openpose_invalid_payload_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad_keypoints.json"
    p.write_text("not json")
    with pytest.raises(ValueError):
        OpenPoseJSONAdapter().load(p)
