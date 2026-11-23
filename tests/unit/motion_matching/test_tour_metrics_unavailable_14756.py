"""#14756 P7: empty/missing evidence must be unavailable, never a perfect zero."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.motion_matching.tour_capture_contract import TourCapture
from src.shared.python.motion_matching.tour_metrics import compute_shared_metrics

pytestmark = pytest.mark.unit

FRAMES = 20
LABELS = ("HeadTop", "BackTop", "WaistLeft", "WaistRight")


def _capture(valid: np.ndarray, labels: tuple[str, ...] = LABELS) -> TourCapture:
    time_s = np.arange(FRAMES) / 100.0
    points = np.zeros((FRAMES, len(labels), 3))
    return TourCapture(time_s=time_s, labels=labels, points_m=points, valid=valid)


def _offset_pred(capture: TourCapture, offset_m: float = 0.1) -> np.ndarray:
    return capture.points_m + np.array([offset_m, 0.0, 0.0])


def test_all_invalid_capture_is_refused_not_zero() -> None:
    cap = _capture(np.zeros((FRAMES, len(LABELS)), dtype=bool))
    with pytest.raises(ValueError, match="No valid"):
        compute_shared_metrics(cap, _offset_pred(cap))


def test_missing_club_labels_report_nan_not_zero() -> None:
    cap = _capture(np.ones((FRAMES, len(LABELS)), dtype=bool))
    metrics = compute_shared_metrics(cap, _offset_pred(cap))
    assert metrics.whole_marker_rmse_m == pytest.approx(0.1)
    assert math.isnan(metrics.club_marker_rmse_m)


def test_empty_early_window_reports_nan_not_zero() -> None:
    cap = _capture(np.ones((FRAMES, len(LABELS)), dtype=bool))
    metrics = compute_shared_metrics(cap, _offset_pred(cap), early_cutoff_s=-1.0)
    assert math.isnan(metrics.early_marker_rmse_m)


def test_missing_pelvis_labels_report_nan_yaw() -> None:
    labels = ("HeadTop", "BackTop")
    cap = _capture(np.ones((FRAMES, len(labels)), dtype=bool), labels)
    metrics = compute_shared_metrics(cap, _offset_pred(cap))
    assert math.isnan(metrics.pelvis_yaw_rmse_rad)


def test_invalid_waist_frames_report_nan_yaw() -> None:
    valid = np.ones((FRAMES, len(LABELS)), dtype=bool)
    valid[:, 2] = False
    cap = _capture(valid)
    metrics = compute_shared_metrics(cap, _offset_pred(cap))
    assert math.isnan(metrics.pelvis_yaw_rmse_rad)


def test_head_only_terminal_window_marks_body_nan_and_head_finite() -> None:
    valid = np.zeros((FRAMES, len(LABELS)), dtype=bool)
    valid[:, 0] = True  # only HeadTop observed
    cap = _capture(valid)
    metrics = compute_shared_metrics(cap, _offset_pred(cap))
    assert metrics.terminal_head_cluster_rmse_m == pytest.approx(0.1)
    assert math.isnan(metrics.terminal_body_excluding_head_rmse_m)


def test_fully_observed_metrics_stay_finite() -> None:
    valid = np.ones((FRAMES, len(LABELS)), dtype=bool)
    cap = _capture(valid)
    metrics = compute_shared_metrics(cap, _offset_pred(cap))
    assert metrics.whole_marker_rmse_m == pytest.approx(0.1)
    assert metrics.early_marker_rmse_m == pytest.approx(0.1)
    assert metrics.pelvis_yaw_rmse_rad == pytest.approx(0.0)
    assert metrics.terminal_marker_rmse_m == pytest.approx(0.1)
