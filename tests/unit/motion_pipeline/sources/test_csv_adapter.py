"""Tests for the CSV adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources.csv_adapter import CSVAdapter

_CSV = """frame,timestamp,x_hip,y_hip,z_hip,x_knee,y_knee,z_knee
0,0.000,0.0,1.0,0.5,0.0,0.5,0.5
1,0.033,0.1,1.0,0.5,0.05,0.5,0.5
2,0.066,0.2,1.0,0.5,0.10,0.5,0.5
"""


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.csv"
    p.write_text(_CSV)
    return p


def test_csv_supports(tmp_path: Path) -> None:
    assert CSVAdapter.supports(_write(tmp_path)) is True


def test_csv_metadata(tmp_path: Path) -> None:
    md = CSVAdapter().metadata(_write(tmp_path))
    assert md.frame_count == 3
    assert md.keypoint_schema == "custom"
    assert md.fps == pytest.approx(1.0 / 0.033, rel=0.05)


def test_csv_load(tmp_path: Path) -> None:
    seq = CSVAdapter().load_checked(_write(tmp_path))
    assert seq.num_frames == 3
    names = [kp.name for kp in seq.frames[0].keypoints]
    assert "hip" in names and "knee" in names


def test_csv_missing_columns_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises((ValueError, KeyError)):
        CSVAdapter().load(p)
