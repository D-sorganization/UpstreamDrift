"""Contracts for calibration records used by the TB-06 C3D campaign."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.pendulum.python.motion_matching.upper_body_capture import (
    CaptureFrameCalibration,
    CaptureSourceClock,
    assess_planarity_lower_bound,
    calibrate_capture_frame,
)
from scripts.motion_capture.upper_body_planarity_receipt import build_planarity_receipt
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


def test_capture_frame_calibration_rejects_nonfinite_residuals() -> None:
    frame = calibrate_capture_frame(_body_target(), ("rs", "re", "rh")).frame
    with pytest.raises(ValueError, match="plane_rmse_m"):
        CaptureFrameCalibration(
            frame=frame,
            marker_names=("rs", "re", "rh"),
            plane_rmse_m=float("nan"),
            plane_max_deviation_m=0.0,
        )


def test_planarity_lower_bound_blocks_an_unattainable_planar_fit() -> None:
    calibration = calibrate_capture_frame(_body_target(), ("rs", "re", "rh"))
    assessment = assess_planarity_lower_bound(calibration, max_marker_rmse_m=0.055)
    assert assessment.planar_fit_is_eligible is True
    assert assessment.marker_rmse_lower_bound_m == pytest.approx(0.0)


def test_planarity_lower_bound_keeps_a_capture_unqualified_when_exceeded() -> None:
    marker_xyz = _body_target().marker_xyz.copy()
    marker_xyz[0, 2, 2] = 0.20
    nonplanar = BodyTarget(
        time=_body_target().time,
        marker_xyz=marker_xyz,
        marker_names=("rs", "re", "rh"),
        impact_idx=1,
        events=(),
        source=SourceProvenance("test.c3d", "c3d", "test", "test", "a" * 64),
    )
    assessment = assess_planarity_lower_bound(
        calibrate_capture_frame(nonplanar, ("rs", "re", "rh")),
        max_marker_rmse_m=0.010,
    )
    assert assessment.planar_fit_is_eligible is False
    assert assessment.marker_rmse_lower_bound_m > assessment.max_marker_rmse_m
    assert "cannot attain" in assessment.reason


def test_planarity_receipt_records_an_unqualified_source_clock() -> None:
    calibration = calibrate_capture_frame(_body_target(), ("rs", "re", "rh"))
    receipt = build_planarity_receipt(
        capture_name="synthetic",
        body_target=_body_target(),
        source_clock=CaptureSourceClock(raw_frame_count=654, raw_frame_rate_hz=360.0),
        calibration=calibration,
        max_marker_rmse_m=0.055,
    )
    assert receipt["schema"] == "upper-body-planarity-preflight/1.0.0"
    assert receipt["source_clock"] == {
        "raw_frame_count": 654,
        "raw_frame_rate_hz": 360.0,
    }
    assert receipt["status"] == "planar_fit_eligible_not_qualified"
    assert receipt["marker_coverage_fraction"] == {"rs": 1.0, "re": 1.0, "rh": 1.0}
    assert receipt["qualification"] == "not_assessed"
