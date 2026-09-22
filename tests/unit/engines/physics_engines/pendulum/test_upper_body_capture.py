"""Contracts for calibration records used by the TB-06 C3D campaign."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.pendulum.python.motion_matching.upper_body_capture import (
    CaptureSourceClock,
    calibrate_capture_frame,
)
from src.shared.python.motion_matching.body_target import BodyTarget
from src.shared.python.motion_matching.club_target import SourceProvenance


pytestmark = pytest.mark.unit


def _body_target() -> BodyTarget:
    time = np.array([0.0, 0.01, 0.02])
    marker_xyz = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        ]
    )
    return BodyTarget(
        time=time,
        marker_xyz=marker_xyz,
        marker_names=("rs", "re", "rh"),
        impact_idx=1,
        events=(),
        source=SourceProvenance("test.c3d", "c3d", "test", "test", "a" * 64),
    )


def test_capture_clock_keeps_native_and_evaluation_rates_distinct() -> None:
    clock = CaptureSourceClock(raw_frame_count=654, raw_frame_rate_hz=360.0)
    assert clock.raw_frame_count == 654
    assert clock.raw_frame_rate_hz == 360.0


def test_capture_frame_is_rigid_and_reports_plane_residual() -> None:
    calibration = calibrate_capture_frame(_body_target(), ("rs", "re", "rh"))
    assert calibration.marker_names == ("rs", "re", "rh")
    assert calibration.frame.plane_basis.shape == (3, 2)
    assert np.allclose(
        calibration.frame.plane_basis.T @ calibration.frame.plane_basis, np.eye(2)
    )
    assert calibration.plane_rmse_m == pytest.approx(0.0)


def test_capture_frame_rejects_unknown_marker() -> None:
    with pytest.raises(ValueError, match="does not provide"):
        calibrate_capture_frame(_body_target(), ("rs", "missing", "rh"))
