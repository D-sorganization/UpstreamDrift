"""Unit tests for swing events detection (Issue #11164)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.swing_comparison.events import SwingEvents, detect_events
from src.shared.python.swing_comparison.motion import (
    SwingMotion,
)


def _make_synthetic_swing(
    n_frames: int = 201,
    dt: float = 0.01,
    address_frame: int = 20,
    top_frame: int = 110,
    impact_frame: int = 140,
    finish_frame: int = 180,
) -> SwingMotion:
    """Build a synthetic swing motion with known event timings.

    Motion geometry:
    - 0 to address_frame: stationary at address (theta = 0)
    - address_frame to top_frame: backswing arc from 0 to -140 deg
    - top_frame to impact_frame: downswing arc from -140 to 0 deg (maximum speed at lowest Z)
    - impact_frame to finish_frame: follow-through to +90 deg
    - finish_frame to end: stationary at +90 deg
    """
    t = np.arange(n_frames, dtype=np.float64) * dt
    theta = np.zeros(n_frames, dtype=np.float64)

    # 1. Address
    theta[:address_frame] = 0.0

    # 2. Backswing
    bs_len = top_frame - address_frame
    tau_bs = np.linspace(0.0, 1.0, bs_len, endpoint=False)
    theta[address_frame:top_frame] = -140.0 * 0.5 * (1.0 - np.cos(np.pi * tau_bs))

    # 3. Downswing
    ds_len = impact_frame - top_frame
    tau_ds = np.linspace(0.0, 1.0, ds_len, endpoint=False)
    # Accelerating downswing profile
    theta[top_frame:impact_frame] = -140.0 * (1.0 - tau_ds**2)

    # 4. Follow-through
    ft_len = finish_frame - impact_frame
    tau_ft = np.linspace(0.0, 1.0, ft_len, endpoint=False)
    theta[impact_frame:finish_frame] = 90.0 * np.sin(0.5 * np.pi * tau_ft)

    # 5. Finish stationary
    theta[finish_frame:] = 90.0

    # Clubhead on planar arc (X-Z plane, Z up)
    r_club = 1.0
    th_rad = np.radians(theta)
    # At impact (theta=0), x = 0, z = -r_club (minimum z)
    club_x = r_club * np.sin(th_rad)
    club_y = np.zeros(n_frames, dtype=np.float64)
    club_z = -r_club * np.cos(th_rad) + 0.1  # slightly off ground
    club_head = np.column_stack([club_x, club_y, club_z])

    # Grip points closer to origin
    grip = 0.3 * club_head + np.array([0.0, 0.0, 0.7])

    markers = {
        "club_head": club_head,
        "grip": grip,
    }

    return SwingMotion(
        t=t,
        markers=markers,
        club_head=club_head,
        grip=grip,
    )


@pytest.mark.unit
class TestSwingEvents:
    """Test suite for swing event detection."""

    def test_synthetic_event_detection_exact(self) -> None:
        """Detect events on synthetic swing with known ground-truth events."""
        motion = _make_synthetic_swing(
            n_frames=201,
            dt=0.01,
            address_frame=20,
            top_frame=110,
            impact_frame=140,
            finish_frame=180,
        )
        events = detect_events(motion)

        assert isinstance(events, SwingEvents)
        # Verify invariants
        assert (
            0
            <= events.address_idx
            <= events.top_idx
            <= events.impact_idx
            <= events.finish_idx
            < len(motion.t)
        )
        assert (
            events.address_time
            <= events.top_time
            <= events.impact_time
            <= events.finish_time
        )

        # Verify precision against ground truth (within 3 frames due to discrete sampling)
        assert abs(events.address_idx - 20) <= 3
        assert abs(events.top_idx - 110) <= 2
        assert abs(events.impact_idx - 140) <= 2
        assert abs(events.finish_idx - 180) <= 5

        # Check times
        assert np.isclose(events.address_time, events.address_idx * 0.01)
        assert np.isclose(events.top_time, events.top_idx * 0.01)
        assert np.isclose(events.impact_time, events.impact_idx * 0.01)
        assert np.isclose(events.finish_time, events.finish_idx * 0.01)

    def test_events_with_nan_gaps(self) -> None:
        """Ensure short NaN gaps do not break event detection."""
        motion = _make_synthetic_swing()
        assert motion.club_head is not None and motion.grip is not None
        # Inject short NaN gaps (e.g. 2 frames) in club_head
        club_head_nan = motion.club_head.copy()
        club_head_nan[50:52, :] = np.nan
        club_head_nan[120:122, :] = np.nan

        motion_nan = SwingMotion(
            t=motion.t,
            markers={"club_head": club_head_nan, "grip": motion.grip},
            club_head=club_head_nan,
            grip=motion.grip,
        )
        events = detect_events(motion_nan)
        assert isinstance(events, SwingEvents)
        assert (
            events.address_idx
            <= events.top_idx
            <= events.impact_idx
            <= events.finish_idx
        )

    def test_rejects_non_monotonic_time(self) -> None:
        """Contract error when time is non-monotonic."""
        motion_valid = _make_synthetic_swing()
        bad_t = motion_valid.t.copy()
        bad_t[50] = bad_t[49]  # flat / non-increasing
        with pytest.raises((PreconditionError, ValueError)):
            SwingMotion(t=bad_t, markers=motion_valid.markers)

    def test_rejects_too_few_frames(self) -> None:
        """Contract error when frames count < 4."""
        t_short = np.array([0.0, 0.01, 0.02])
        m_short = {"M1": np.zeros((3, 3))}
        with pytest.raises((PreconditionError, ValueError)):
            SwingMotion(t=t_short, markers=m_short)

    def test_rejects_mismatched_marker_shape(self) -> None:
        """Contract error when marker shape doesn't match t."""
        t = np.array([0.0, 0.01, 0.02, 0.03])
        m = {"M1": np.zeros((5, 3))}  # 5 instead of 4
        with pytest.raises((PreconditionError, ValueError)):
            SwingMotion(t=t, markers=m)
