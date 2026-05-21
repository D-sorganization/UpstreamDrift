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
    assert isinstance(out, KeypointSequence)
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
    assert isinstance(out, MarkerTrajectory)
    expected = traj.frames[4].markers["M1"].x
    for i in range(5, 8):
        assert out.frames[i].markers["M1"].x == pytest.approx(expected)


def test_gap_fill_markers_no_occluded_returns_unchanged_data() -> None:
    traj = make_marker_trajectory(num_frames=10)
    out = gap_fill(traj, strategy=GapFillStrategy.LINEAR)
    assert isinstance(out, MarkerTrajectory)
    # Marker positions are identical
    for orig, new in zip(traj.frames, out.frames, strict=False):
        for name in orig.markers:
            assert orig.markers[name].x == new.markers[name].x


def test_gap_fill_cubic_falls_back_to_linear() -> None:
    """CUBIC strategy is documented as a placeholder that delegates to linear."""
    seq = make_low_confidence_keypoint_sequence(num_frames=15, low_conf_range=(5, 7))
    cubic_out = gap_fill(seq, strategy=GapFillStrategy.CUBIC, max_gap=10)
    linear_out = gap_fill(seq, strategy=GapFillStrategy.LINEAR, max_gap=10)
    assert isinstance(cubic_out, KeypointSequence)
    assert isinstance(linear_out, KeypointSequence)
    for f_c, f_l in zip(cubic_out.frames, linear_out.frames, strict=False):
        for kp_c, kp_l in zip(f_c.keypoints, f_l.keypoints, strict=False):
            assert kp_c.x == pytest.approx(kp_l.x)


def test_gap_fill_pca_strategy_implemented() -> None:
    # Build a marker trajectory with 3 correlated markers.
    # The markers are correlated so PCA can reconstruct them.
    from src.shared.python.motion_pipeline.contracts import Marker, MarkerFrame

    frames = []
    for i in range(15):
        # Correlated marker values (linear ramps with different scale/bias)
        # First marker M1 has a gap in [5, 7] (occluded = True)
        occ = 5 <= i <= 7
        m1 = Marker(name="M1", x=float(i) * 0.1, y=0.0, z=0.0, occluded=occ)
        m2 = Marker(name="M2", x=float(i) * 0.2, y=1.0, z=0.0, occluded=False)
        m3 = Marker(name="M3", x=float(i) * -0.05, y=0.0, z=0.5, occluded=False)
        frames.append(
            MarkerFrame(timestamp=i / 30.0, markers={"M1": m1, "M2": m2, "M3": m3}, frame_index=i)
        )
    traj = MarkerTrajectory(id="traj_occ_multi", frames=frames)

    out = gap_fill(traj, strategy=GapFillStrategy.PCA, max_gap=10)
    # If PCA is implemented the metadata strategy field would record it
    assert out.metadata.get("strategy") == "pca"

    linear_out = gap_fill(traj, strategy=GapFillStrategy.LINEAR, max_gap=10)
    assert isinstance(out, MarkerTrajectory)
    assert isinstance(linear_out, MarkerTrajectory)
    same_as_linear = all(
        out.frames[i].markers["M1"].x == linear_out.frames[i].markers["M1"].x for i in range(5, 8)
    )
    assert not same_as_linear


def test_gap_fill_gaps_at_boundaries() -> None:
    # Gap at index 0 (start == 0) and gap at the end (end >= len(frames))
    seq_start = make_low_confidence_keypoint_sequence(num_frames=10, low_conf_range=(0, 2))
    out_start = gap_fill(seq_start, strategy=GapFillStrategy.LINEAR)
    assert isinstance(out_start, KeypointSequence)
    # The low-confidence window is not filled because start == 0
    assert out_start.frames[0].keypoints[0].confidence < 0.5

    seq_end = make_low_confidence_keypoint_sequence(num_frames=10, low_conf_range=(8, 9))
    out_end = gap_fill(seq_end, strategy=GapFillStrategy.LINEAR)
    assert isinstance(out_end, KeypointSequence)
    # The low-confidence window at the end is not filled because end >= len(frames)
    assert out_end.frames[9].keypoints[0].confidence < 0.5

    # Nearest neighbor boundary cases
    out_near_start = gap_fill(seq_start, strategy=GapFillStrategy.NEAREST)
    assert isinstance(out_near_start, KeypointSequence)
    assert out_near_start.frames[0].keypoints[0].confidence < 0.5

    # Same for markers
    traj_start = make_marker_trajectory_with_occlusion(num_frames=10, occluded_range=(0, 2))
    out_m_start = gap_fill(traj_start, strategy=GapFillStrategy.LINEAR)
    assert isinstance(out_m_start, MarkerTrajectory)
    assert out_m_start.frames[0].markers["M1"].occluded is True

    out_m_near_start = gap_fill(traj_start, strategy=GapFillStrategy.NEAREST)
    assert isinstance(out_m_near_start, MarkerTrajectory)
    assert out_m_near_start.frames[0].markers["M1"].occluded is True


def test_gap_fill_pca_edge_cases() -> None:
    from src.shared.python.motion_pipeline.contracts import Marker, MarkerFrame, MarkerTrajectory

    # 1. Less than 2 frames
    m1 = Marker(name="M1", x=0.0, y=0.0, z=0.0, occluded=True)
    m2 = Marker(name="M2", x=1.0, y=0.0, z=0.0, occluded=False)
    frame = MarkerFrame(timestamp=0.0, markers={"M1": m1, "M2": m2}, frame_index=0)
    traj_1 = MarkerTrajectory(id="traj_1", frames=[frame])
    out_1 = gap_fill(traj_1, strategy=GapFillStrategy.PCA)
    assert isinstance(out_1, MarkerTrajectory)
    assert out_1.frames[0].markers["M1"].occluded is True

    # 2. PCA on keypoints falls back to linear
    seq = make_low_confidence_keypoint_sequence(num_frames=10, low_conf_range=(3, 5))
    out_kp_pca = gap_fill(seq, strategy=GapFillStrategy.PCA)
    assert isinstance(out_kp_pca, KeypointSequence)
    assert out_kp_pca.metadata.get("strategy") == "pca"
    # Should fill the gap using linear interpolation
    assert out_kp_pca.frames[3].keypoints[0].confidence == 0.5

    # 3. PCA with less than 2 visible coords in some frames (should fall back to linear for those frames)
    # Build a trajectory where M1/M2 are occluded in frames [3, 4] (middle of sequence)
    # Frames 0-2 and 5-9 are fully visible.
    frames = []
    for i in range(10):
        occ = 3 <= i <= 4
        m1 = Marker(name="M1", x=float(i) * 0.1, y=0.0, z=0.0, occluded=occ)
        m2 = Marker(name="M2", x=float(i) * 0.2, y=1.0, z=0.0, occluded=occ)
        frames.append(MarkerFrame(timestamp=i / 30.0, markers={"M1": m1, "M2": m2}, frame_index=i))
    traj_no_visible = MarkerTrajectory(id="traj_no_visible", frames=frames)
    out_no_visible = gap_fill(traj_no_visible, strategy=GapFillStrategy.PCA)
    assert isinstance(out_no_visible, MarkerTrajectory)
    # Reconstructed using linear fallback because PCA was underdetermined for frames [3, 4]
    assert out_no_visible.frames[3].markers["M1"].occluded is False


def test_pure_python_gap_fill() -> None:
    from src.shared.python.motion_pipeline.preprocessing._gap_fill_pure_python import (
        gap_fill as pure_gap_fill,
        GapFillStrategy as PureGapFillStrategy,
    )
    from src.shared.python.motion_pipeline.contracts import Marker, MarkerFrame, MarkerTrajectory

    # Test linear on keypoint sequence
    seq = make_low_confidence_keypoint_sequence(num_frames=20, low_conf_range=(5, 8))
    out = pure_gap_fill(seq, strategy=PureGapFillStrategy.LINEAR, max_gap=10)
    assert out.metadata.get("gap_filled") is True

    # Test nearest on keypoints
    seq_near = make_low_confidence_keypoint_sequence(num_frames=15, low_conf_range=(5, 7))
    out_near = pure_gap_fill(seq_near, strategy=PureGapFillStrategy.NEAREST, max_gap=10)
    assert out_near.metadata.get("gap_filled") is True

    # Test linear on markers
    traj = make_marker_trajectory_with_occlusion(num_frames=20, occluded_range=(5, 8))
    out_traj = pure_gap_fill(traj, strategy=PureGapFillStrategy.LINEAR, max_gap=10)
    assert out_traj.metadata.get("gap_filled") is True

    # Test nearest on markers
    traj_near = make_marker_trajectory_with_occlusion(num_frames=20, occluded_range=(5, 7))
    out_traj_near = pure_gap_fill(traj_near, strategy=PureGapFillStrategy.NEAREST, max_gap=10)
    assert out_traj_near.metadata.get("gap_filled") is True

    # Test cubic fallback to linear
    out_cubic = pure_gap_fill(seq_near, strategy=PureGapFillStrategy.CUBIC, max_gap=10)
    assert out_cubic.metadata.get("gap_filled") is True

    # Test unsupported type
    with pytest.raises(ValueError, match="Unsupported"):
        pure_gap_fill("invalid", strategy=PureGapFillStrategy.LINEAR)  # type: ignore[arg-type]

    # Test empty or short sequences (length < 2)
    short_seq = make_keypoint_sequence(num_frames=1, num_kp=2)
    assert pure_gap_fill(short_seq) is short_seq

    short_traj = make_marker_trajectory(num_frames=1)
    assert pure_gap_fill(short_traj) is short_traj

    # Test PCA on keypoints (falls back to linear)
    out_kp_pca = pure_gap_fill(seq, strategy=PureGapFillStrategy.PCA, max_gap=10)
    assert out_kp_pca.metadata.get("gap_filled") is True

    # Test PCA on marker trajectory
    frames = []
    for i in range(15):
        occ = 5 <= i <= 7
        m1 = Marker(name="M1", x=float(i) * 0.1, y=0.0, z=0.0, occluded=occ)
        m2 = Marker(name="M2", x=float(i) * 0.2, y=1.0, z=0.0, occluded=False)
        m3 = Marker(name="M3", x=float(i) * -0.05, y=0.0, z=0.5, occluded=False)
        frames.append(
            MarkerFrame(timestamp=i / 30.0, markers={"M1": m1, "M2": m2, "M3": m3}, frame_index=i)
        )
    traj_pca = MarkerTrajectory(id="traj_pca", frames=frames)
    out_pca = pure_gap_fill(traj_pca, strategy=PureGapFillStrategy.PCA, max_gap=10)
    assert out_pca.metadata.get("strategy") == "pca"

    # Test PCA underdetermined fallback (fewer than 2 visible frames)
    frames_no_visible = []
    for i in range(10):
        occ = 3 <= i <= 4
        m1 = Marker(name="M1", x=float(i) * 0.1, y=0.0, z=0.0, occluded=occ)
        m2 = Marker(name="M2", x=float(i) * 0.2, y=1.0, z=0.0, occluded=occ)
        frames_no_visible.append(
            MarkerFrame(timestamp=i / 30.0, markers={"M1": m1, "M2": m2}, frame_index=i)
        )
    traj_no_visible = MarkerTrajectory(id="traj_no_visible", frames=frames_no_visible)
    out_no_visible = pure_gap_fill(traj_no_visible, strategy=PureGapFillStrategy.PCA, max_gap=10)
    assert out_no_visible.metadata.get("gap_filled") is True

    # Test PCA rank-deficient fallback
    from unittest.mock import patch
    import numpy as np

    with patch("numpy.linalg.svd", side_effect=np.linalg.LinAlgError):
        out_svd_error = pure_gap_fill(traj_pca, strategy=PureGapFillStrategy.PCA, max_gap=10)
        assert out_svd_error.metadata.get("strategy") == "pca"  # fell back to linear inside

    # Test PCA with SVD zero/empty singular values
    with patch("numpy.linalg.svd", return_value=(None, np.array([]), None)):
        out_zero_s = pure_gap_fill(traj_pca, strategy=PureGapFillStrategy.PCA, max_gap=10)
        assert out_zero_s.metadata.get("strategy") == "pca"
