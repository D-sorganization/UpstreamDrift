"""Unit tests for motion_pipeline.preprocessing.resample."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    KeypointSequence,
    MarkerTrajectory,
)
from src.shared.python.motion_pipeline.preprocessing.resample import resample

from ._local_fixtures import make_keypoint_sequence, make_marker_trajectory


def test_resample_unsupported_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        resample("not-a-sequence", target_fps=120.0)  # type: ignore[arg-type]


def test_resample_keypoints_short_sequence_unchanged() -> None:
    seq = make_keypoint_sequence(num_frames=1, num_kp=2, fps=30.0)
    out = resample(seq, target_fps=120.0)
    assert out.num_frames == 1


def test_resample_markers_short_trajectory_unchanged() -> None:
    traj = make_marker_trajectory(num_frames=1, fps=30.0)
    out = resample(traj, target_fps=120.0)
    assert out.num_frames == 1


def test_resample_keypoints_upsamples_frame_count() -> None:
    seq = make_keypoint_sequence(num_frames=30, num_kp=1, fps=30.0)
    duration = seq.frames[-1].timestamp - seq.frames[0].timestamp
    out = resample(seq, target_fps=120.0, source_fps=30.0)
    expected = int(duration * 120.0) + 1
    assert isinstance(out, KeypointSequence)
    assert out.num_frames == expected
    assert out.metadata.get("resampled") is True
    assert out.metadata.get("source_fps") == 30.0
    assert out.metadata.get("target_fps") == 120.0


def test_resample_keypoints_preserves_endpoints() -> None:
    seq = make_keypoint_sequence(num_frames=30, num_kp=1, fps=30.0)
    out = resample(seq, target_fps=100.0)
    assert out.frames[0].timestamp == pytest.approx(seq.frames[0].timestamp)
    assert out.frames[-1].timestamp == pytest.approx(seq.frames[-1].timestamp)


def test_resample_markers_increases_frame_count() -> None:
    traj = make_marker_trajectory(num_frames=30, fps=60.0)
    out = resample(traj, target_fps=1000.0)
    assert isinstance(out, MarkerTrajectory)
    # 1000Hz should produce far more frames than 30
    assert out.num_frames > 30
    assert out.metadata.get("target_fps") == 1000.0


def test_resample_round_trip_preserves_signal_shape() -> None:
    """60Hz -> 120Hz -> 60Hz round-trip should approximate the original."""
    seq = make_keypoint_sequence(num_frames=30, num_kp=1, fps=60.0)
    up = resample(seq, target_fps=120.0, source_fps=60.0)
    down = resample(up, target_fps=60.0, source_fps=120.0)

    raw = np.array([f.keypoints[0].x for f in seq.frames])
    rt = np.array([f.keypoints[0].x for f in down.frames])
    # Allow small numerical drift; both should have same length to within 1
    n = min(len(raw), len(rt))
    np.testing.assert_allclose(raw[:n], rt[:n], atol=1e-6)


def test_resample_keypoints_timestamp_grid_uniform() -> None:
    seq = make_keypoint_sequence(num_frames=30, num_kp=1, fps=30.0)
    out = resample(seq, target_fps=100.0)
    timestamps = np.array([f.timestamp for f in out.frames])
    diffs = np.diff(timestamps)
    np.testing.assert_allclose(diffs, diffs[0], atol=1e-6)


def test_resample_marker_trajectory_id_preserved() -> None:
    traj = make_marker_trajectory(num_frames=10, fps=30.0)
    out = resample(traj, target_fps=60.0)
    assert out.id == traj.id
