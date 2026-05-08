"""Tests for the TRC adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources.trc_adapter import TRCAdapter


_TRC = "\n".join(
    [
        "PathFileType\t4\t(X/Y/Z)\ttiny.trc",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        "100.0\t100.0\t3\t2\tmm\t100.0\t1\t3",
        "Frame#\tTime\tHEAD\t\t\tHIP\t\t\t",
        "\t\tX1\tY1\tZ1\tX2\tY2\tZ2",
        "1\t0.0\t1000\t2000\t3000\t100\t200\t300",
        "2\t0.01\t1010\t2010\t3010\t110\t210\t310",
        "3\t0.02\t1020\t2020\t3020\t120\t220\t320",
        "",
    ]
)


def _write_trc(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.trc"
    p.write_text(_TRC)
    return p


def test_trc_supports(tmp_path: Path) -> None:
    p = _write_trc(tmp_path)
    assert TRCAdapter.supports(p) is True


def test_trc_metadata(tmp_path: Path) -> None:
    md = TRCAdapter().metadata(_write_trc(tmp_path))
    assert md.format_name == "trc"
    assert md.frame_count == 3
    assert md.unit_system == "millimeters"
    assert md.fps == pytest.approx(100.0)


def test_trc_load_converts_mm_to_meters(tmp_path: Path) -> None:
    traj = TRCAdapter().load_checked(_write_trc(tmp_path))
    assert traj.num_frames == 3
    # 1000 mm -> 1.0 m
    head_first = traj.frames[0].markers["HEAD"]
    assert head_first.x == pytest.approx(1.0)
    assert head_first.y == pytest.approx(2.0)
    assert head_first.z == pytest.approx(3.0)


def test_trc_truncated_raises(tmp_path: Path) -> None:
    p = tmp_path / "short.trc"
    p.write_text("PathFileType\t4\nfoo\n")
    with pytest.raises(ValueError):
        TRCAdapter().load(p)
