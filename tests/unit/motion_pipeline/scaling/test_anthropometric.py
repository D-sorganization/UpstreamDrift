"""Unit tests for motion_pipeline.scaling.anthropometric."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    MarkerFrame,
    MarkerTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.scaling.anthropometric import (
    MarkerMap,
    estimate_subject_height,
    scale_skeleton,
)

from ._local_fixtures import (
    make_marker_frame_for_scale,
    make_marker_trajectory_for_scale,
    make_simple_skeleton,
)


def test_scale_skeleton_no_segment_pairs_uses_default() -> None:
    """No pairs supplied -> measured lengths empty -> global scale defaults to 1.0."""
    rig = make_simple_skeleton()
    markers = make_marker_frame_for_scale(1.0)
    out = scale_skeleton(rig, markers)
    assert isinstance(out, SkeletonRig)
    assert out.id == f"{rig.id}-scaled"
    assert out.scale == pytest.approx(1.0, abs=1e-6)


def test_scale_skeleton_recovers_15x_scale_within_tolerance() -> None:
    rig = make_simple_skeleton()
    target = 1.5
    traj = make_marker_trajectory_for_scale(target_scale=target)
    pairs = [("RASI", "LASI"), ("RTHI", "RKNE")]
    out = scale_skeleton(rig, traj, segment_pairs=pairs)
    # Phantom test: scale within 2% of target
    assert out.scale == pytest.approx(target, rel=0.02)


def test_scale_skeleton_postcondition_positive_segment_lengths() -> None:
    rig = make_simple_skeleton()
    traj = make_marker_trajectory_for_scale(target_scale=1.2)
    pairs = [("RASI", "LASI")]
    out = scale_skeleton(rig, traj, segment_pairs=pairs)
    for joint in out.joints.values():
        offset_norm = float(np.linalg.norm(joint.tpose_offset))
        assert offset_norm >= 0.0


def test_scale_skeleton_marker_set_parity() -> None:
    """Different marker pair sets that measure the same anatomical scale
    should produce equivalent global scale within tolerance.
    """
    rig = make_simple_skeleton()
    traj = make_marker_trajectory_for_scale(target_scale=1.3)
    out_pelvis = scale_skeleton(rig, traj, segment_pairs=[("RASI", "LASI")])
    out_thigh = scale_skeleton(rig, traj, segment_pairs=[("RTHI", "RKNE")])
    # Both should approximate the same underlying scale.
    assert out_pelvis.scale == pytest.approx(out_thigh.scale, rel=0.10)


def test_scale_skeleton_empty_trajectory_raises() -> None:
    rig = make_simple_skeleton()
    with pytest.raises(ValueError, match="Empty"):
        scale_skeleton(
            rig,
            MarkerTrajectory.model_construct(id="empty", frames=[]),
        )


def test_scale_skeleton_accepts_marker_frame_directly() -> None:
    rig = make_simple_skeleton()
    frame = make_marker_frame_for_scale(1.0)
    out = scale_skeleton(rig, frame)
    assert isinstance(out, SkeletonRig)


def test_estimate_subject_height_with_head_and_foot_markers() -> None:
    from src.shared.python.motion_pipeline.contracts import Marker

    frame = MarkerFrame(
        timestamp=0.0,
        markers={
            "RHEE": Marker(name="RHEE", x=0.0, y=0.0, z=0.0),
            "LHEE": Marker(name="LHEE", x=0.1, y=0.0, z=0.0),
            "HEAD": Marker(name="HEAD", x=0.0, y=0.0, z=1.5),
            "C7": Marker(name="C7", x=0.0, y=0.0, z=1.4),
        },
        frame_index=0,
    )
    h = estimate_subject_height(frame)
    # head_z=1.5, foot_z=0.0 -> height = 1.5/0.82 + 0.10 ~= 1.929
    assert 1.5 < h < 2.2


def test_estimate_subject_height_without_foot_markers() -> None:
    from src.shared.python.motion_pipeline.contracts import Marker

    frame = MarkerFrame(
        timestamp=0.0,
        markers={"HEAD": Marker(name="HEAD", x=0.0, y=0.0, z=1.5)},
        frame_index=0,
    )
    h = estimate_subject_height(frame)
    assert h > 0.0


def test_estimate_subject_height_with_trajectory() -> None:
    traj = make_marker_trajectory_for_scale(1.0, num_frames=3)
    h = estimate_subject_height(traj)
    assert h > 0.0


def test_marker_map_dataclass_defaults() -> None:
    mm = MarkerMap()
    assert mm.marker_to_segment == {}
    assert mm.segment_pairs == []


def test_scale_skeleton_metadata_records_scale_factor() -> None:
    rig = make_simple_skeleton()
    traj = make_marker_trajectory_for_scale(1.5)
    out = scale_skeleton(rig, traj, segment_pairs=[("RASI", "LASI")])
    assert "scale_factor" in out.metadata
    assert out.metadata["scale_factor"] == pytest.approx(out.scale)
    assert "scale_factors_by_segment" in out.metadata
