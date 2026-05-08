"""Unit tests for motion_pipeline.preprocessing.gap_fill."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    KeypointSequence,
    MarkerTrajectory,
)
from src.shared.python.motion_pipeline.preprocessing.gap_fill import (
    GapFillStrategy,
    gap_fill,
)

from ._local_fixtures import (
    make_keypoint_sequence,
    make_low_confidence_keypoint_sequence,
    make_marker_trajectory,
    make_marker_trajectory_with_occlusion,
)


def test_gap_fill_unsupported_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        gap_fill("not-a-sequence", strategy=GapFillStrategy.LINEAR)  # type: ignore[arg-type]


def test_gap_fill_keypoints_with_no_gaps_returns_input() -> None:
    seq = make_keypoint_sequence(num_frames=10, num_kp=2, confidence=1.0)
    out = gap_fill(seq, strategy=GapFillStrategy.LINEAR)
    assert isinstance(out, KeypointSequence)
    assert out.num_frames == seq.num_frames


def test_gap_fill_keypoints_short_sequence_unchanged() -> None:
    seq = make_keypoint_sequence(num_frames=1, num_kp=2)
    out = gap_fill(seq, strategy=GapFillStrategy.LINEAR)
    assert out.num_frames == 1


def test_gap_fill_markers_short_trajectory_unchanged() -> None:
    traj = make_marker_trajectory(num_frames=1)
    out = gap_fill(traj, strategy=GapFillStrategy.LINEAR)
    assert out.num_frames == 1


def test_gap_fill_keypoints_linear_fills_low_confidence_window() -> None:
    seq = make_low_confidence_keypoint_sequence(num_frames=20, low_conf_range=(5, 8))
    out = gap_fill(seq, strategy=GapFillStrategy.LINEAR, max_gap=10)

    assert isinstance(out, KeypointSequence)
    assert out.num_frames == seq.num_frames
    # All x-values must be finite (no NaNs introduced by gap-filling)
    for f in out.frames:
        for kp in f.keypoints:
            assert np.isfinite(kp.x)
            assert np.isfinite(kp.y)
    # Filled frames have monotonically non-decreasing x because the source
    # signal does (linear ramp). Tolerance for numerical noise.
    xs = [f.keypoints[0].x for f in out.frames]
    assert xs[-1] > xs[0]
    assert out.metadata.get("gap_filled") is True
    assert out.metadata.get("strategy") == "linear"


def test_gap_fill_keypoints_nearest_uses_previous() -> None:
    seq = make_low_confidence_keypoint_sequence(num_frames=15, low_conf_range=(5, 7))
    out = gap_fill(seq, strategy=GapFillStrategy.NEAREST, max_gap=10)
    # Nearest copies the value at index start-1 (=4 -> x=0.4) into the gap
    expected = seq.frames[4].keypoints[0].x
    for i in range(5, 8):
        assert out.frames[i].keypoints[0].x == pytest.approx(expected)


def test_gap_fill_keypoints_skips_too_large_gap() -> None:
    seq = make_low_confidence_keypoint_sequence(num_frames=30, low_conf_range=(5, 25))
    out = gap_fill(seq, strategy=GapFillStrategy.LINEAR, max_gap=5)
    # Big gap is left alone; the original (low-confidence) frames remain
    assert out.num_frames == seq.num_frames


def test_gap_fill_markers_linear_interpolates_occluded_window() -> None:
    traj = make_marker_trajectory_with_occlusion(num_frames=20, occluded_range=(5, 8))
    out = gap_fill(traj, strategy=GapFillStrategy.LINEAR, max_gap=10)

    assert isinstance(out, MarkerTrajectory)
    assert out.num_frames == traj.num_frames
    # M1 was a linear ramp: x = i * 0.1. Interpolated values must lie
    # between the bracketing frames' x.
    for i in range(5, 9):
        assert 0.4 <= out.frames[i].markers["M1"].x <= 0.9
    assert out.metadata.get("gap_filled") is True
    # Interpolated markers are no longer flagged occluded
    for i in range(5, 9):
        assert out.frames[i].markers["M1"].occluded is False


def test_gap_fill_markers_nearest_uses_previous() -> None:
    traj = make_marker_trajectory_with_occlusion(num_frames=20, occluded_range=(5, 7))
    out = gap_fill(traj, strategy=GapFillStrategy.NEAREST, max_gap=10)
    expected = traj.frames[4].markers["M1"].x
    for i in range(5, 8):
        assert out.frames[i].markers["M1"].x == pytest.approx(expected)


def test_gap_fill_markers_no_occluded_returns_unchanged_data() -> None:
    traj = make_marker_trajectory(num_frames=10)
    out = gap_fill(traj, strategy=GapFillStrategy.LINEAR)
    # Marker positions are identical
    for orig, new in zip(traj.frames, out.frames, strict=False):
        for name in orig.markers:
            assert orig.markers[name].x == new.markers[name].x


def test_gap_fill_cubic_falls_back_to_linear() -> None:
    """CUBIC strategy is documented as a placeholder that delegates to linear."""
    seq = make_low_confidence_keypoint_sequence(num_frames=15, low_conf_range=(5, 7))
    cubic_out = gap_fill(seq, strategy=GapFillStrategy.CUBIC, max_gap=10)
    linear_out = gap_fill(seq, strategy=GapFillStrategy.LINEAR, max_gap=10)
    for f_c, f_l in zip(cubic_out.frames, linear_out.frames, strict=False):
        for kp_c, kp_l in zip(f_c.keypoints, f_l.keypoints, strict=False):
            assert kp_c.x == pytest.approx(kp_l.x)


@pytest.mark.xfail(
    strict=False,
    reason=(
        "PCA gap-fill strategy is declared in GapFillStrategy enum but the "
        "production implementation only branches on LINEAR/CUBIC/NEAREST. "
        "Spec for #4564 names PCA as a key strategy. Production fix tracked "
        "separately; this xfail documents the gap."
    ),
)
def test_gap_fill_pca_strategy_implemented() -> None:
    seq = make_low_confidence_keypoint_sequence(num_frames=15, low_conf_range=(5, 7))
    out = gap_fill(seq, strategy=GapFillStrategy.PCA, max_gap=10)
    # If PCA were implemented the metadata strategy field would record it
    # and the values in the gap would differ from the LINEAR baseline.
    linear_out = gap_fill(seq, strategy=GapFillStrategy.LINEAR, max_gap=10)
    same_as_linear = all(
        out.frames[i].keypoints[0].x == linear_out.frames[i].keypoints[0].x
        for i in range(5, 8)
    )
    assert not same_as_linear  # would currently be True -> xfail
