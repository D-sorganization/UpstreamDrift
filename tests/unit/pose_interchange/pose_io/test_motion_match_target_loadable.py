"""``save_motion_match_target`` produces a JSON loadable by ``load_body_target``."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.motion_matching.load_body_target import load_body_target
from src.shared.python.pose_interchange.canonical import (
    canonical_from_reference_setup,
)
from src.shared.python.pose_interchange.pose_io import (
    MOTION_MATCH_LANDMARKS,
    save_motion_match_target,
)

pytestmark = pytest.mark.unit


def test_motion_match_target_loads_via_load_body_target(tmp_path: Path) -> None:
    pose = canonical_from_reference_setup()
    out = tmp_path / "single_frame_target.json"
    save_motion_match_target(pose, out)
    target = load_body_target(out)

    # All canonical landmarks present in expected order.
    assert target.marker_names == MOTION_MATCH_LANDMARKS
    assert target.marker_xyz.shape == (
        2,
        len(MOTION_MATCH_LANDMARKS),
        3,
    )
    # Frame 0 == frame 1 by construction.
    assert (target.marker_xyz[0] == target.marker_xyz[1]).all()
    assert target.impact_idx == 0
    assert target.coordinate_frame == "z_up_right_handed"
    # Address event annotation propagated.
    assert any(ev.label == "address" for ev in target.events)


def test_motion_match_target_marker_set_filter(tmp_path: Path) -> None:
    pose = canonical_from_reference_setup()
    out = tmp_path / "subset.json"
    save_motion_match_target(pose, out)
    subset = ["pelvis", "l_wrist", "clubhead"]
    target = load_body_target(out, marker_set=subset)
    assert set(target.marker_names) == set(subset)
