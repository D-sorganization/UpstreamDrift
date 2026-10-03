"""Integration test for swing comparison with capture-A (Issue #11164)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.shared.python.swing_comparison.events import detect_events
from src.shared.python.swing_comparison.metrics import (
    compare,
    compute_all_metrics,
)
from src.shared.python.swing_comparison.motion import (
    SwingMotion,
    swing_motion_from_markers,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
C3D_TA_DRIVER_PATH = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"


def _load_capture_a_swing_motion() -> SwingMotion:
    """Helper to load capture-A from data/C3D_TA_Driver.c3d directly."""
    if not C3D_TA_DRIVER_PATH.exists():
        pytest.skip(f"capture-A file not found at {C3D_TA_DRIVER_PATH}")

    try:
        import ezc3d
    except ImportError:
        pytest.skip("ezc3d not available for loading C3D capture")

    c3d = ezc3d.c3d(str(C3D_TA_DRIVER_PATH))
    points = c3d["data"]["points"]  # (4, markers, frames)
    labels = [str(lbl).strip() for lbl in c3d["parameters"]["POINT"]["LABELS"]["value"]]
    rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    n_frames = points.shape[2]
    t = np.arange(n_frames, dtype=np.float64) / rate

    # Y-up capture frame to right-handed Z-up: (x, -z, y)
    raw_xyz = np.transpose(points[:3, :, :], (2, 1, 0))  # (frames, markers, 3)
    zup_xyz = np.zeros_like(raw_xyz)
    zup_xyz[..., 0] = raw_xyz[..., 0]
    zup_xyz[..., 1] = -raw_xyz[..., 2]
    zup_xyz[..., 2] = raw_xyz[..., 1]

    # Convert mm to m if needed (data/C3D_TA_Driver.c3d is already in m)
    units = str(c3d["parameters"]["POINT"]["UNITS"]["value"][0]).strip().lower()
    if units == "mm":
        zup_xyz *= 0.001

    markers = {labels[i]: zup_xyz[:, i, :] for i in range(len(labels))}
    return swing_motion_from_markers(t, markers)


@pytest.mark.integration
class TestCaptureAIntegration:
    """Integration test with reference driver capture capture-A."""

    def test_capture_a_plausible_ranges(self) -> None:
        """Assert plausible ranges for tempo ratio (2-4.2) and impact speed (35-60 m/s)."""
        motion = _load_capture_a_swing_motion()
        events = detect_events(motion)
        metrics = compute_all_metrics(motion, events)

        # 1. Tempo ratio plausible range (2.0 to 4.2)
        assert 2.0 <= metrics.tempo.tempo_ratio <= 4.2, (
            f"Tempo ratio {metrics.tempo.tempo_ratio} out of range"
        )

        # 2. Impact club head speed plausible range (35 to 60 m/s, ~78-134 mph)
        assert 35.0 <= metrics.club.impact_club_head_speed_m_s <= 60.0, (
            f"Impact speed {metrics.club.impact_club_head_speed_m_s} out of range"
        )

        # 3. Kinematic sequence ordering
        assert metrics.kinematic_sequence.pelvis.peak_speed > 0.0
        assert (
            metrics.kinematic_sequence.club.peak_speed
            > metrics.kinematic_sequence.pelvis.peak_speed
        )

        # 4. Self-comparison produces zero RMS
        report = compare(motion, motion)
        assert np.isclose(report.mean_marker_rms, 0.0, atol=1e-6)
