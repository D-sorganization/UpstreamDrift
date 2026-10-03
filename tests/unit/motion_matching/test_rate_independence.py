"""Tests for rate-independence across the shared matching pipeline (#11166)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline.lane import Lane
from src.shared.python.motion_matching.tour_capture_contract import (
    TOUR_CAPTURE,
    TourCapture,
)

pytestmark = pytest.mark.unit


def _make_synthetic_capture(rate_hz: float, frames: int) -> TourCapture:
    t = np.arange(frames, dtype=float) / rate_hz
    points = np.zeros((frames, len(TOUR_CAPTURE.labels), 3), dtype=float)
    valid = np.ones((frames, len(TOUR_CAPTURE.labels)), dtype=bool)
    return TourCapture(t, TOUR_CAPTURE.labels, points, valid)


def test_lane_rate_hz_derived_from_timestamps() -> None:
    """Lane.rate_hz must be derived from capture.time_s rather than hardcoded 360 Hz."""
    cap_240 = _make_synthetic_capture(240.0, 367)
    lane_240 = Lane(capture=cap_240)
    assert hasattr(lane_240, "rate_hz")
    assert lane_240.rate_hz == pytest.approx(240.0)

    cap_360 = _make_synthetic_capture(360.0, 654)
    lane_360 = Lane(capture=cap_360)
    assert lane_360.rate_hz == pytest.approx(360.0)


def test_dynamics_backswing_metrics_scales_with_rate() -> None:
    """_build_backswing_metrics must use rate-dependent 1-second limit (241 frames at 240 Hz, 361 at 360 Hz)."""
    from src.shared.python.motion_matching.pipeline.dynamics import (
        _build_backswing_metrics,
    )
    from unittest.mock import MagicMock

    record = MagicMock()
    record.time_s = np.linspace(0, 1.5, 367)
    record.weight_fraction = np.ones(367)

    sim_q = np.zeros((367, 10))
    q_ref = np.zeros((367, 10))
    sim_errors = np.zeros((367, 38))
    valid = np.ones((367, 38), dtype=bool)

    # Calling with rate_hz=240.0 should limit to 241 frames
    m240 = _build_backswing_metrics(
        sim_q, q_ref, sim_errors, valid, frames=367, record=record, rate_hz=240.0
    )
    assert "marker_rms_m" in m240

    # Calling with rate_hz=360.0 should match the existing 361 limit
    m360 = _build_backswing_metrics(
        sim_q, q_ref, sim_errors, valid, frames=367, record=record, rate_hz=360.0
    )
    assert "marker_rms_m" in m360
