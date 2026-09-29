"""Tests for the AlphaPose JSON adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources.alphapose_json_adapter import (
    AlphaPoseJSONAdapter,
)


def _det(image_id: str, n: int = 17, score: float = 0.9) -> dict:
    flat: list[float] = []
    for i in range(n):
        flat.extend([float(i), float(i + 1), 0.85])
    return {
        "image_id": image_id,
        "category_id": 1,
        "keypoints": flat,
        "score": score,
        "idx": [0],
    }


def _write(tmp_path: Path) -> Path:
    payload = [_det("0001.jpg"), _det("0002.jpg"), _det("0003.jpg")]
    p = tmp_path / "alphapose-results.json"
    p.write_text(json.dumps(payload))
    return p


def test_alphapose_supports(tmp_path: Path) -> None:
    assert AlphaPoseJSONAdapter.supports(_write(tmp_path)) is True


def test_alphapose_metadata(tmp_path: Path) -> None:
    md = AlphaPoseJSONAdapter().metadata(_write(tmp_path))
    assert md.frame_count == 3
    assert md.keypoint_schema == "COCO_17"


def test_alphapose_load(tmp_path: Path) -> None:
    seq = AlphaPoseJSONAdapter().load_checked(_write(tmp_path))
    assert seq.num_frames == 3
    assert len(seq.frames[0].keypoints) == 17


def test_alphapose_picks_highest_score_per_frame(tmp_path: Path) -> None:
    payload = [
        _det("0001.jpg", score=0.5),
        _det("0001.jpg", score=0.95),
    ]
    p = tmp_path / "alphapose-results.json"
    p.write_text(json.dumps(payload))
    seq = AlphaPoseJSONAdapter().load_checked(p)
    assert seq.num_frames == 1


def test_alphapose_invalid_raises(tmp_path: Path) -> None:
    p = tmp_path / "alphapose-results.json"
    p.write_text("[]")
    with pytest.raises(ValueError):
        AlphaPoseJSONAdapter().load(p)
