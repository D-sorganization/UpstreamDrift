"""Unit tests for fsp_integration.py -- FSP engine integration (Phase 2).

Tests follow TDD: written before implementation, covering:
- detect_md_mf_window: phase-annotated and fallback heuristic paths
- extract_clubhead_trajectory: shape, dtype, key resolution, DbC
- compute_swing_fsp: full pipeline, graceful Rust fallback, planar swing
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.fsp_integration import (
    FspResult,
    compute_swing_fsp,
    detect_md_mf_window,
    extract_clubhead_trajectory,
)

# ---------------------------------------------------------------------------
# Synthetic frame factory
# ---------------------------------------------------------------------------


def make_synthetic_frames(n: int = 100) -> list[dict]:
    """100 frames with clubhead velocity peaking at frame 50."""
    frames: list[dict] = []
    for i in range(n):
        vel = np.exp(-0.02 * (i - 50) ** 2)  # Gaussian peak at 50
        frames.append(
            {
                "clubhead": np.array([float(i), 0.0, float(i) * 0.1]),
                "clubhead_velocity": np.array([0.0, 0.0, vel]),
                "lead_hand": np.array([float(i) * 0.5, 0.0, float(i) * 0.05]),
            }
        )
    return frames


def make_phase_annotated_frames() -> list[dict]:
    """Frames with explicit phase annotations for mid_downswing / mid_follow_through."""
    frames: list[dict] = []
    phases = (
        ["setup"] * 20
        + ["backswing"] * 20
        + ["mid_downswing"] * 1
        + ["impact"] * 10
        + ["mid_follow_through"] * 1
        + ["finish"] * 48
    )
    for i, phase in enumerate(phases):
        frames.append(
            {
                "clubhead": np.array([float(i), 0.0, 0.0]),
                "clubhead_velocity": np.array([0.0, 0.0, 1.0]),
                "lead_hand": np.array([0.0, 0.0, 0.0]),
                "phase": phase,
            }
        )
    return frames


# ---------------------------------------------------------------------------
# detect_md_mf_window
# ---------------------------------------------------------------------------


class TestDetectMdMfWindow:
    def test_basic_velocity_heuristic(self) -> None:
        frames = make_synthetic_frames()
        start, end = detect_md_mf_window(frames)
        assert 0 <= start < end < len(frames)
        assert end - start >= 10, "window should be at least 10 frames wide"

    def test_window_around_impact(self) -> None:
        """Impact at frame 50 -> window should bracket it roughly 30%/30%."""
        frames = make_synthetic_frames(100)
        start, end = detect_md_mf_window(frames)
        # Impact at 50; 30% of 100 = 30 -> start ~= 20, end ~= 80
        assert start <= 40, f"start {start} should be before frame 40"
        assert end >= 60, f"end {end} should be after frame 60"

    def test_too_few_frames_raises(self) -> None:
        with pytest.raises(ValueError, match="5"):
            detect_md_mf_window([{} for _ in range(3)])

    def test_exactly_five_frames_passes(self) -> None:
        """Exactly 5 frames: minimum allowed -- should not raise."""
        frames = make_synthetic_frames(5)
        start, end = detect_md_mf_window(frames)
        assert start < end

    def test_phase_annotated_uses_phase_labels(self) -> None:
        frames = make_phase_annotated_frames()
        start, end = detect_md_mf_window(frames)
        # mid_downswing is at index 40, mid_follow_through is at index 51
        assert start == 40
        assert end == 51

    def test_returns_valid_indices(self) -> None:
        frames = make_synthetic_frames(50)
        start, end = detect_md_mf_window(frames)
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert end < len(frames)


# ---------------------------------------------------------------------------
# extract_clubhead_trajectory
# ---------------------------------------------------------------------------


class TestExtractClubheadTrajectory:
    def test_shape_correct(self) -> None:
        frames = make_synthetic_frames()
        traj = extract_clubhead_trajectory(frames, 30, 70)
        assert traj.shape == (41, 3), f"expected (41, 3), got {traj.shape}"

    def test_dtype_float64(self) -> None:
        frames = make_synthetic_frames()
        traj = extract_clubhead_trajectory(frames, 0, 9)
        assert traj.dtype == np.float64

    def test_bad_indices_raises(self) -> None:
        frames = make_synthetic_frames()
        with pytest.raises(ValueError):
            extract_clubhead_trajectory(frames, 70, 30)

    def test_equal_indices_raises(self) -> None:
        frames = make_synthetic_frames()
        with pytest.raises(ValueError):
            extract_clubhead_trajectory(frames, 50, 50)

    def test_alternate_key_club_head(self) -> None:
        """Frames using 'club_head' key instead of 'clubhead'."""
        frames = [
            {
                "club_head": np.array([float(i), 0.0, 0.0]),
                "clubhead_velocity": np.array([0.0, 0.0, 1.0]),
                "lead_hand": np.array([0.0, 0.0, 0.0]),
            }
            for i in range(10)
        ]
        traj = extract_clubhead_trajectory(frames, 0, 9)
        assert traj.shape == (10, 3)

    def test_full_range(self) -> None:
        frames = make_synthetic_frames(20)
        traj = extract_clubhead_trajectory(frames, 0, 19)
        assert traj.shape == (20, 3)


# ---------------------------------------------------------------------------
# compute_swing_fsp
# ---------------------------------------------------------------------------


class TestComputeSwingFsp:
    def test_returns_fsp_result(self) -> None:
        frames = make_synthetic_frames()
        result = compute_swing_fsp(frames)
        assert isinstance(result, FspResult)

    def test_slope_deg_is_float(self) -> None:
        frames = make_synthetic_frames()
        result = compute_swing_fsp(frames)
        assert isinstance(result.slope_deg, float)

    def test_direction_deg_is_float(self) -> None:
        frames = make_synthetic_frames()
        result = compute_swing_fsp(frames)
        assert isinstance(result.direction_deg, float)

    def test_clubhead_deviations_shape(self) -> None:
        frames = make_synthetic_frames()
        result = compute_swing_fsp(frames)
        assert result.clubhead_deviations.ndim == 1
        assert len(result.clubhead_deviations) > 0

    def test_hand_deviations_shape(self) -> None:
        frames = make_synthetic_frames()
        result = compute_swing_fsp(frames)
        assert result.hand_deviations.ndim == 1
        assert len(result.hand_deviations) > 0

    def test_plane_attribute_present(self) -> None:
        frames = make_synthetic_frames()
        result = compute_swing_fsp(frames)
        assert result.plane is not None
        assert hasattr(result.plane, "normal")
        assert hasattr(result.plane, "centroid")

    def test_fsp_slope_planar_swing(self) -> None:
        """Clubhead on z=0 plane -> slope approx 0 degrees."""
        frames = [
            {
                "clubhead": np.array([float(i), float(i) * 0.1, 0.0]),
                "clubhead_velocity": np.array([0.0, 0.0, float(i)]),
                "lead_hand": np.array([float(i) * 0.5, 0.0, 0.0]),
            }
            for i in range(100)
        ]
        result = compute_swing_fsp(frames)
        assert abs(result.slope_deg) < 5.0, (
            f"planar swing slope should be near 0, got {result.slope_deg}"
        )

    def test_deviations_dtype(self) -> None:
        frames = make_synthetic_frames()
        result = compute_swing_fsp(frames)
        assert result.clubhead_deviations.dtype in (np.float64, np.float32)
        assert result.hand_deviations.dtype in (np.float64, np.float32)

    def test_slope_in_valid_range(self) -> None:
        frames = make_synthetic_frames()
        result = compute_swing_fsp(frames)
        assert 0.0 <= result.slope_deg <= 90.0, (
            f"slope should be in [0, 90], got {result.slope_deg}"
        )
