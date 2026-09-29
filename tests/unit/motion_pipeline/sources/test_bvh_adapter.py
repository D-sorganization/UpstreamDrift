"""Tests for the BVH adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources.bvh_adapter import BVHAdapter

_BVH_FIXTURE = """HIERARCHY
ROOT Hips
{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET 0.0 10.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
            OFFSET 0.0 5.0 0.0
        }
    }
}
MOTION
Frames: 3
Frame Time: 0.0333333
0 0 0 10 0 0 5 0 0
0 0 0 12 0 0 6 0 0
0 0 0 14 0 0 7 0 0
"""


def _write_bvh(tmp_path: Path, content: str = _BVH_FIXTURE) -> Path:
    p = tmp_path / "tiny.bvh"
    p.write_text(content)
    return p


def test_bvh_supports(tmp_path: Path) -> None:
    p = _write_bvh(tmp_path)
    assert BVHAdapter.supports(p) is True


def test_bvh_metadata(tmp_path: Path) -> None:
    p = _write_bvh(tmp_path)
    md = BVHAdapter().metadata(p)
    assert md.format_name == "bvh"
    assert md.frame_count == 3
    assert md.unit_system == "degrees"
    assert md.fps == pytest.approx(30.0, rel=1e-3)


def test_bvh_load_returns_joint_trajectory(tmp_path: Path) -> None:
    p = _write_bvh(tmp_path)
    traj = BVHAdapter().load_checked(p)
    assert traj.num_frames == 3
    # Monotonic timestamps
    timestamps = [f.timestamp for f in traj.frames]
    assert timestamps == sorted(timestamps)


def test_bvh_missing_motion_section_raises(tmp_path: Path) -> None:
    p = tmp_path / "broken.bvh"
    p.write_text("HIERARCHY\nROOT Hips\n{\n}\n")
    with pytest.raises(ValueError, match="MOTION"):
        BVHAdapter().load(p)
