"""Regression tests for HRNet JSON adapter (#4683).

The adapter previously crashed on canonical HRNet/mmpose exports where
``keypoints`` is a nested list ``[[x, y, score], ...]`` rather than the
flat AlphaPose-style triple. These tests pin both the shipped golden
fixture and hand-built canonical samples covering both shapes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources import load_any
from src.shared.python.motion_pipeline.sources.hrnet_json_adapter import (
    HRNetJSONAdapter,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
GOLDEN = REPO_ROOT / "tests" / "data" / "motion_pipeline" / "golden" / "hrnet.json"


def test_load_shipped_golden_fixture() -> None:
    seq = load_any(GOLDEN)
    assert len(seq.frames) == 30
    assert all(f.schema_name == "COCO_17" for f in seq.frames)
    assert all(len(f.keypoints) == 17 for f in seq.frames)
    ts = [f.timestamp for f in seq.frames]
    assert ts == sorted(ts)
    for f in seq.frames:
        for kp in f.keypoints:
            assert math.isfinite(kp.x) and math.isfinite(kp.y)
            assert 0.0 <= kp.confidence <= 1.0


def test_canonical_nested_triplets(tmp_path: Path) -> None:
    """COCO-17 nested ``[[x, y, score], ...]`` parses correctly."""
    payload = {
        "model": "hrnet_w32",
        "schema": "COCO_17",
        "fps": 60.0,
        "frames": [
            {"frame": i, "keypoints": [[1.0 * k, 2.0 * k, 0.9] for k in range(17)]}
            for i in range(3)
        ],
    }
    p = tmp_path / "hrnet.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    seq = HRNetJSONAdapter().load(p)
    assert len(seq.frames) == 3
    assert all(len(f.keypoints) == 17 for f in seq.frames)
    assert seq.frames[0].schema_name == "COCO_17"
    assert seq.frames[0].keypoints[1].x == pytest.approx(1.0)
    assert seq.frames[0].keypoints[1].confidence == pytest.approx(0.9)


def test_flat_triplets_form(tmp_path: Path) -> None:
    """Flat ``[x, y, c, x, y, c, ...]`` form (AlphaPose-style) still parses."""
    flat: list[float] = []
    for k in range(17):
        flat.extend([float(k), float(2 * k), 0.8])
    payload = {
        "fps": 30.0,
        "frames": [{"frame_index": 0, "keypoints": flat}],
    }
    p = tmp_path / "hrnet_flat.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    seq = HRNetJSONAdapter().load(p)
    assert len(seq.frames) == 1
    assert len(seq.frames[0].keypoints) == 17
    assert seq.frames[0].keypoints[0].confidence == pytest.approx(0.8)


def test_frame_index_alias(tmp_path: Path) -> None:
    """``frame`` is accepted as an alias for ``frame_index``."""
    payload = {
        "fps": 30.0,
        "frames": [
            {"frame": 2, "keypoints": [[0.0, 0.0, 1.0] for _ in range(17)]},
            {"frame": 0, "keypoints": [[0.0, 0.0, 1.0] for _ in range(17)]},
            {"frame": 1, "keypoints": [[0.0, 0.0, 1.0] for _ in range(17)]},
        ],
    }
    p = tmp_path / "hrnet.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    seq = HRNetJSONAdapter().load(p)
    indices = [f.frame_index for f in seq.frames]
    assert indices == [0, 1, 2]


def test_rejects_no_frames(tmp_path: Path) -> None:
    p = tmp_path / "hrnet.json"
    p.write_text(json.dumps({"frames": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no frames"):
        HRNetJSONAdapter().load(p)


def test_rejects_malformed_root(tmp_path: Path) -> None:
    p = tmp_path / "hrnet.json"
    p.write_text(json.dumps(42), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list"):
        HRNetJSONAdapter().load(p)
