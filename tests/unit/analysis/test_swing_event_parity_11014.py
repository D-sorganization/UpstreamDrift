"""Parity test across all four swing event detection call sites (Issue #11014).

Verifies that on a shared synthetic swing fixture:
- All four sites report the same impact/peak frame (within +-1 for _align).
- The three event-level sites agree on address and top.
- The analytics.detect_events frames equal detect_swing_events exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct.analytics import SwingSeries, detect_events
from src.shared.python.analysis.phase_detection import PhaseDetectionMixin
from src.shared.python.analysis.swing_events import detect_swing_events
from src.shared.python.data_io.swing_capture_import import (
    JointTrajectory,
    SwingCaptureImporter,
)
from src.shared.python.motion_matching.loaders._align import detect_impact_index

pytestmark = pytest.mark.unit


def _make_synthetic_swing() -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Generate a realistic 240 fps synthetic swing with known key events.

    Profile characteristics:
    - 480 frames at 240 fps (2.0 seconds).
    - Quiet address: frames 0..100 (> 0.4 s >= 0.3 s) at baseline speed < 0.25 m/s.
    - Smooth backswing: smooth rise peaking near frame 160.
    - Slow top: smooth trough near frame 209 where speed dips (~2.0 m/s).
    - Sharp downswing: accelerating to impact peak at frame 300 (40.3 m/s).
    - Finish decay: smooth decay returning toward quiet baseline.

    Returns:
        (speed, fps, times, clubhead)
    """
    fps = 240.0
    n = 480
    t = np.arange(n, dtype=float)

    g_bs = 6.0 * np.exp(-(((t - 160.0) / 32.0) ** 2))
    g_top = 2.5 * np.exp(-(((t - 240.0) / 40.0) ** 2))
    g_impact = 40.0 * np.exp(-(((t - 300.0) / 16.0) ** 2))

    speed = 0.05 + g_bs + g_top + g_impact

    dt = 1.0 / fps
    times = t * dt

    # Central difference on integrated speed reproduces the velocity profile
    clubhead_x = np.cumsum(speed) * dt
    clubhead = np.column_stack([clubhead_x, np.zeros(n), np.zeros(n)])

    return speed, fps, times, clubhead


class _PhaseDetectionHost(PhaseDetectionMixin):
    """Host object providing required attributes for PhaseDetectionMixin."""

    def __init__(self, speed: np.ndarray, times: np.ndarray) -> None:
        self.club_head_speed = speed
        self.times = times
        self.duration = float(times[-1] - times[0])


def test_swing_event_parity_across_all_four_sites() -> None:
    """All 4 sites delegate to the canonical detector and maintain parity."""
    speed, fps, times, clubhead = _make_synthetic_swing()
    n = len(speed)

    # 1. Canonical detector (source of truth)
    canonical = detect_swing_events(speed, fps)
    assert canonical.peak == 300
    assert canonical.address == 126
    assert canonical.top == 209

    # 2. Site (a): analytics.detect_events
    series = SwingSeries(
        time_s=times,
        pelvis_turn_deg=np.zeros(n),
        shoulder_turn_deg=np.zeros(n),
        x_factor_deg=np.zeros(n),
        hand_speed_mps=speed,
        hand_speed_uncertainty_mps=np.zeros(n),
    )
    analytics_events = detect_events(series, fps)

    # Exact bit-identical match with canonical detector
    assert analytics_events.address_frame == canonical.address
    assert analytics_events.top_frame == canonical.top
    assert analytics_events.peak_speed_frame == canonical.peak
    assert analytics_events.finish_frame == canonical.finish

    # 3. Site (b): PhaseDetectionMixin
    host = _PhaseDetectionHost(speed, times)
    smoothed = host._smooth_speed(speed)
    impact_idx, transition_idx, takeaway_idx, _finish_idx = host._find_key_events(
        smoothed, fps
    )

    # Agreement on impact and key events
    assert impact_idx == canonical.peak
    assert takeaway_idx == canonical.address
    assert transition_idx == canonical.top

    # 4. Site (c): SwingCaptureImporter.detect_swing_phases
    importer = SwingCaptureImporter()
    trajectory = JointTrajectory(
        joint_names=["total"],
        positions=np.zeros((n, 1)),
        velocities=speed[:, None],
        times=times,
        frame_rate=fps,
    )
    import_labels = importer.detect_swing_phases(trajectory)

    assert import_labels.address == 0
    assert import_labels.backswing_start == canonical.address
    assert import_labels.top_of_backswing == canonical.top
    assert import_labels.downswing_start == canonical.top
    assert import_labels.impact == canonical.peak
    assert import_labels.follow_through_end == n - 1

    # 5. Site (d): _align.detect_impact_index
    align_impact = detect_impact_index(times, clubhead)
    # Impact within +-1 frame due to 5-point central differencing
    assert abs(align_impact - canonical.peak) <= 1
