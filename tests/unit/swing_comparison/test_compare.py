"""Unit tests for swing comparison engine (Issue #11164)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.swing_comparison.metrics import (
    ComparisonReport,
    compare,
)
from src.shared.python.swing_comparison.motion import (
    SwingMotion,
)


def _make_simple_swing(
    n_frames: int = 100, dt: float = 0.01, offset: float = 0.0
) -> SwingMotion:
    t = np.arange(n_frames, dtype=np.float64) * dt

    top_f = int(0.55 * n_frames)
    imp_f = int(0.75 * n_frames)
    theta = np.zeros(n_frames, dtype=np.float64)

    # Backswing: 0 to top_f (slows to 0 at top)
    tau_bs = np.linspace(0.0, 1.0, top_f, endpoint=False)
    theta[:top_f] = -140.0 * 0.5 * (1.0 - np.cos(np.pi * tau_bs))

    # Downswing: top_f to imp_f (accelerates to impact)
    tau_ds = np.linspace(0.0, 1.0, imp_f - top_f, endpoint=False)
    theta[top_f:imp_f] = -140.0 * (1.0 - tau_ds**2)

    # Follow-through
    tau_ft = np.linspace(0.0, 1.0, n_frames - imp_f)
    theta[imp_f:] = 90.0 * np.sin(0.5 * np.pi * tau_ft)

    th_rad = np.radians(theta)
    offset_vec = np.array([offset, 0.0, 0.0])
    club_x = np.sin(th_rad)
    club_y = np.zeros(n_frames)
    club_z = -np.cos(th_rad) + 0.1
    club_head = np.column_stack([club_x, club_y, club_z]) + offset_vec
    grip = (
        0.5 * (np.column_stack([club_x, club_y, club_z]))
        + np.array([0.0, 0.0, 0.5])
        + offset_vec
    )

    # Shoulder and hip markers
    sh_l = (
        np.column_stack(
            [0.2 * np.cos(th_rad), 0.2 * np.sin(th_rad), np.full(n_frames, 1.4)]
        )
        + offset_vec
    )
    sh_r = (
        np.column_stack(
            [-0.2 * np.cos(th_rad), -0.2 * np.sin(th_rad), np.full(n_frames, 1.4)]
        )
        + offset_vec
    )
    w_l = (
        np.column_stack(
            [
                0.15 * np.cos(th_rad * 0.7),
                0.15 * np.sin(th_rad * 0.7),
                np.full(n_frames, 0.9),
            ]
        )
        + offset_vec
    )
    w_r = (
        np.column_stack(
            [
                -0.15 * np.cos(th_rad * 0.7),
                -0.15 * np.sin(th_rad * 0.7),
                np.full(n_frames, 0.9),
            ]
        )
        + offset_vec
    )
    elb = 0.5 * (sh_l + grip)

    markers = {
        "WaistLeft": w_l,
        "WaistRight": w_r,
        "LShoulderBack": sh_l,
        "RShoulderBack": sh_r,
        "LShoulderTop": sh_l,
        "LElbowOut": elb,
        "LWristTop": grip,
        "Marker_2:2:1": club_head,
        "Marker_3:3:1": grip,
    }

    return SwingMotion(
        t=t,
        markers=markers,
        club_head=club_head,
        grip=grip,
    )


@pytest.mark.unit
class TestSwingComparison:
    """Test suite for compare(a, b)."""

    def test_self_comparison_yields_zero_differences(self) -> None:
        """Comparing a swing to itself gives exactly zero differences and zero RMS."""
        motion = _make_simple_swing()
        report = compare(motion, motion)

        assert isinstance(report, ComparisonReport)
        assert np.isclose(report.mean_marker_rms, 0.0, atol=1e-6)
        for marker_name, rms in report.shared_marker_rms.items():
            assert np.isclose(rms, 0.0, atol=1e-6), (
                f"RMS non-zero for {marker_name}: {rms}"
            )

        # Differences dictionary should have 0.0 for numeric metrics
        for key, diff in report.differences.items():
            if isinstance(diff, (int, float)):
                assert np.isclose(diff, 0.0, atol=1e-5), (
                    f"Diff non-zero for {key}: {diff}"
                )

    def test_offset_comparison_analytic_rms(self) -> None:
        """Comparing motion against identical motion offset by delta gives exact RMS = delta."""
        delta = 0.05  # 5 cm offset along X
        motion_a = _make_simple_swing(offset=0.0)
        motion_b = _make_simple_swing(offset=delta)

        report = compare(motion_a, motion_b)
        assert np.isclose(report.mean_marker_rms, delta, rtol=1e-2)
        for marker_name, rms in report.shared_marker_rms.items():
            assert np.isclose(rms, delta, rtol=1e-2), (
                f"RMS mismatch for {marker_name}: {rms}"
            )

    def test_different_sampling_rates_and_durations(self) -> None:
        """Compare swings with different frame counts and sample rates via time-normalization."""
        motion_100hz = _make_simple_swing(n_frames=100, dt=0.01)
        motion_200hz = _make_simple_swing(n_frames=200, dt=0.005)

        report = compare(motion_100hz, motion_200hz)
        assert isinstance(report, ComparisonReport)
        # Should align closely despite different sample rates
        assert report.mean_marker_rms < 0.05

    def test_no_shared_markers_edge_case(self) -> None:
        """Edge case where two motions share zero markers."""
        t = np.arange(50, dtype=np.float64) * 0.01
        m_a = {"MarkerA": np.zeros((50, 3))}
        m_b = {"MarkerB": np.zeros((50, 3))}
        head = np.zeros((50, 3))
        grip = np.zeros((50, 3))

        swing_a = SwingMotion(t=t, markers=m_a, club_head=head, grip=grip)
        swing_b = SwingMotion(t=t, markers=m_b, club_head=head, grip=grip)

        report = compare(swing_a, swing_b)
        assert report.shared_marker_rms == {}
        assert report.mean_marker_rms == 0.0
