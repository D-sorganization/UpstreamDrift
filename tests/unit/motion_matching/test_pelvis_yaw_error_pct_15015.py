"""Pelvis-yaw error percent must not depend on capture-frame heading (#15015 C7)."""

from __future__ import annotations

import numpy as np
import pytest

import math

from src.shared.python.motion_matching.pelvis_yaw import (
    PELVIS_YAW_ERROR_REFERENCE_DEG,
    compute_pelvis_yaw_metrics,
    pelvis_yaw_error_pct_from_diff_deg,
)
from src.shared.python.motion_matching.replay_metrics import compute_replay_five_metrics

pytestmark = pytest.mark.unit


def _waist_markers_from_target_yaw_deg(yaw_deg: float) -> np.ndarray:
    """Two markers on the horizontal plane with pelvis heading ``yaw_deg``."""
    rad = np.radians(yaw_deg)
    direction = np.array([np.cos(rad), np.sin(rad), 0.0])
    markers = np.zeros((2, 3))
    markers[0] = [-direction[0] * 0.2, -direction[1] * 0.2, 0.9]
    markers[1] = [direction[0] * 0.2, direction[1] * 0.2, 0.9]
    return markers


def _rotate_xy(markers: np.ndarray, delta_deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(delta_deg)), np.sin(np.radians(delta_deg))
    rot = np.array([[c, -s], [s, c]])
    out = markers.copy()
    out[:, :2] = markers[:, :2] @ rot.T
    return out


def test_pelvis_yaw_error_pct_is_rotation_invariant_for_fixed_angular_error() -> None:
    """A rigid z-rotation of the capture frame must not change ``pelvis_yaw_error_pct``."""
    diff_deg = 4.5
    baseline_yaw = 12.0
    target = _waist_markers_from_target_yaw_deg(baseline_yaw)
    pred = _waist_markers_from_target_yaw_deg(baseline_yaw + diff_deg)
    base_metrics = compute_pelvis_yaw_metrics(pred, target, wl_i=0, wr_i=1)
    assert base_metrics.valid
    assert base_metrics.yaw_diff_deg == pytest.approx(diff_deg, abs=0.05)

    for extra_rotation in (0.0, 30.0, 90.0, 150.0, -45.0):
        t_rot = _rotate_xy(target, extra_rotation)
        p_rot = _rotate_xy(pred, extra_rotation)
        metrics = compute_pelvis_yaw_metrics(p_rot, t_rot, wl_i=0, wr_i=1)
        assert metrics.valid
        assert metrics.yaw_diff_deg == pytest.approx(diff_deg, abs=0.05)
        assert metrics.pelvis_yaw_error_pct == pytest.approx(
            base_metrics.pelvis_yaw_error_pct, abs=1e-9
        )


def test_pelvis_yaw_error_pct_uses_fixed_reference_not_target_heading() -> None:
    """Small errors near zero target heading must not explode relative to mid-range headings."""
    small_diff = 0.25
    for target_yaw in (0.0, 5.0, 90.0):
        target = _waist_markers_from_target_yaw_deg(target_yaw)
        pred = _waist_markers_from_target_yaw_deg(target_yaw + small_diff)
        metrics = compute_pelvis_yaw_metrics(pred, target, wl_i=0, wr_i=1)
        expected = pelvis_yaw_error_pct_from_diff_deg(small_diff)
        assert metrics.pelvis_yaw_error_pct == pytest.approx(expected, abs=1e-9)
        assert metrics.pelvis_yaw_error_pct == pytest.approx(
            100.0 * small_diff / PELVIS_YAW_ERROR_REFERENCE_DEG, abs=1e-9
        )


def test_pelvis_yaw_pct_maps_four_point_five_deg_to_five_percent() -> None:
    """At the fixed 90° reference, 4.5° angular error yields a 5% reported value."""
    assert pelvis_yaw_error_pct_from_diff_deg(4.5) == pytest.approx(5.0, abs=1e-9)


def test_pelvis_yaw_error_pct_from_diff_deg_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        pelvis_yaw_error_pct_from_diff_deg(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        pelvis_yaw_error_pct_from_diff_deg(float("inf"))


def test_replay_five_metrics_pelvis_nan_when_waist_invalid_on_terminal_frame() -> None:
    """``compute_replay_five_metrics`` reports NaN (not a sentinel pct) when pelvis is degenerate."""
    time_s = np.array([0.10, 0.85])
    pred = np.zeros((2, 2, 3))
    target = np.ones((2, 2, 3))
    valid = np.array([[True, True], [False, True]], dtype=bool)
    labels = ["WaistLeft", "WaistRight"]
    metrics = compute_replay_five_metrics(
        time_s=time_s,
        pred_markers_m=pred,
        target_markers_m=target,
        valid=valid,
        marker_labels=labels,
    )
    assert math.isnan(metrics.pelvis_yaw_error_pct)
