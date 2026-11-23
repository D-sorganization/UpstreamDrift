"""Missing capture evidence must never masquerade as a perfect fit (#14756 P7)."""

import math

import numpy as np
import pytest

from src.shared.python.motion_matching.acceptance import GateStatus, Horizon, evaluate
from src.shared.python.motion_matching.tour_capture_contract import TourCapture
from src.shared.python.motion_matching.tour_metrics import compute_shared_metrics

pytestmark = pytest.mark.unit


def _capture(labels: tuple[str, ...], valid: np.ndarray | None = None) -> TourCapture:
    points = np.zeros((4, len(labels), 3))
    if "WaistLeft" in labels:
        points[:, labels.index("WaistLeft"), 2] = 0.2
    return TourCapture(
        time_s=np.arange(4) * 0.25,
        labels=labels,
        points_m=points,
        valid=np.ones(points.shape[:2], dtype=bool) if valid is None else valid,
    )


@pytest.mark.parametrize("selection", [None, (), ("absent",)])
def test_no_valid_tracked_evidence_is_unmeasured(selection: tuple | None) -> None:
    capture = _capture(("HeadTop",), np.zeros((4, 1), dtype=bool))
    metrics = compute_shared_metrics(
        capture, capture.points_m, tracked_labels=selection
    )
    assert all(math.isnan(value) for value in metrics.as_dict().values())
    verdict = evaluate({"shared_metrics": metrics.as_dict()}, horizon=Horizon.G1)
    whole_gate = next(g for g in verdict.gates if g.name == "whole_marker_rmse_m")
    assert whole_gate.status == GateStatus.FAILED
    assert not verdict.is_physically_accepted


def test_empty_selection_with_valid_capture_is_unmeasured() -> None:
    capture = _capture(("HeadTop",))
    metrics = compute_shared_metrics(capture, capture.points_m, tracked_labels=())
    assert all(math.isnan(value) for value in metrics.as_dict().values())


@pytest.mark.parametrize("labels", [("HeadTop",), ("WaistLeft",), ("WaistRight",)])
def test_missing_club_or_pelvis_labels_are_unmeasured(labels: tuple[str, ...]) -> None:
    capture = _capture(labels)
    metrics = compute_shared_metrics(capture, capture.points_m)
    assert metrics.whole_marker_rmse_m == 0.0
    assert math.isnan(metrics.club_marker_rmse_m)
    assert math.isnan(metrics.pelvis_yaw_rmse_rad)


def test_unobserved_windows_and_groups_are_unmeasured() -> None:
    labels = ("HeadTop", "WaistLeft", "WaistRight", "Marker_2:2:1")
    valid = np.zeros((4, 4), dtype=bool)
    valid[1, 0] = True
    capture = _capture(labels, valid)
    predicted = capture.points_m.copy()
    predicted[1, 0, 0] += 0.03
    metrics = compute_shared_metrics(capture, predicted, early_cutoff_s=0.0)
    assert metrics.whole_marker_rmse_m == pytest.approx(0.03)
    for name, value in metrics.as_dict().items():
        if name != "whole_marker_rmse_m":
            assert math.isnan(value), name


def test_observed_zero_and_nonzero_errors_preserve_formulas() -> None:
    capture = _capture(("HeadTop", "WaistLeft", "WaistRight", "Marker_2:2:1"))
    zero = compute_shared_metrics(capture, capture.points_m)
    assert all(value == 0.0 for value in zero.as_dict().values())
    predicted = capture.points_m.copy()
    predicted[:, :, 0] += 0.03
    predicted[:, :, 1] += 0.04
    measured = compute_shared_metrics(capture, predicted)
    for name, value in measured.as_dict().items():
        assert value == pytest.approx(0.0 if name == "pelvis_yaw_rmse_rad" else 0.05)
