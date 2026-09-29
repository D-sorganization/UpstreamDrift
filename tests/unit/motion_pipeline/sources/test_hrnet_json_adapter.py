"""Tests for the HRNet JSON adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources.hrnet_json_adapter import (
    HRNetJSONAdapter,
)


def _frame(idx: int, n: int = 17) -> dict:
    flat: list[float] = []
    for i in range(n):
        flat.extend([float(i), float(i + 1), 0.9])
    return {"frame_index": idx, "keypoints": flat, "timestamp": idx / 30.0}


def _write(tmp_path: Path) -> Path:
    payload = {"schema": "HRNet", "frames": [_frame(0), _frame(1), _frame(2)]}
    p = tmp_path / "hrnet_pose.json"
    p.write_text(json.dumps(payload))
    return p


def test_hrnet_supports(tmp_path: Path) -> None:
    assert HRNetJSONAdapter.supports(_write(tmp_path)) is True


def test_hrnet_metadata(tmp_path: Path) -> None:
    md = HRNetJSONAdapter().metadata(_write(tmp_path))
    assert md.frame_count == 3
    assert md.keypoint_schema == "COCO_17"


def test_hrnet_load(tmp_path: Path) -> None:
    seq = HRNetJSONAdapter().load_checked(_write(tmp_path))
    assert seq.num_frames == 3
    assert len(seq.frames[0].keypoints) == 17


def test_hrnet_invalid_raises(tmp_path: Path) -> None:
    p = tmp_path / "hrnet_pose.json"
    p.write_text(json.dumps({"schema": "HRNet", "frames": []}))
    with pytest.raises(ValueError):
        HRNetJSONAdapter().load(p)
