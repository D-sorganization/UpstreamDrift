"""Tests for the OpenSim STO/MOT adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources.sto_mot_adapter import (
    OpenSimSTOMOTAdapter,
)


_STO = """name=tiny
version=1
nRows=3
nColumns=4
inDegrees=yes
endheader
time\thip_flexion\tknee_flexion\tankle_flexion
0.00\t10.0\t5.0\t1.0
0.01\t12.0\t6.0\t1.5
0.02\t14.0\t7.0\t2.0
"""


def _write_sto(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.sto"
    p.write_text(_STO)
    return p


def test_sto_supports(tmp_path: Path) -> None:
    assert OpenSimSTOMOTAdapter.supports(_write_sto(tmp_path)) is True


def test_sto_mot_metadata(tmp_path: Path) -> None:
    md = OpenSimSTOMOTAdapter().metadata(_write_sto(tmp_path))
    assert md.format_name == "opensim_sto_mot"
    assert md.frame_count == 3
    assert md.unit_system == "degrees"
    assert md.fps == pytest.approx(100.0)


def test_sto_load_converts_degrees_to_radians(tmp_path: Path) -> None:
    motion = OpenSimSTOMOTAdapter().load_checked(_write_sto(tmp_path))
    traj = motion.trajectory
    assert traj.num_frames == 3
    # 10 degrees ~ 0.1745 rad
    assert traj.frames[0].q[0] == pytest.approx(0.1745, abs=1e-3)


def test_sto_missing_endheader_raises(tmp_path: Path) -> None:
    p = tmp_path / "broken.sto"
    p.write_text("name=foo\nversion=1\ntime col1\n0 0\n")
    with pytest.raises(ValueError, match="endheader"):
        OpenSimSTOMOTAdapter().load(p)
