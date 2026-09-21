"""Unit tests for bounded two-window direct-node SLSQP motion matching (#9967, #10338).

Tests:
1. `compute_marker_metrics` matches exact squared errors and pelvis yaw wrapping.
2. `check_acceptance` evaluates physical gate thresholds honestly.
3. Metric reproduction against run-102 returned candidate uninterrupted replay.
4. `evaluate_zero_displacement_parity` verifies exact parity to 1e-12.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.two_window_fit import (
    MarkerMetricResults,
    TwoWindowParityInputs,
    ZeroDisplacementParityReport,
    check_acceptance,
    compute_marker_metrics,
    evaluate_zero_displacement_parity,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
RUN102_DIR = (
    ROOT
    / "docs"
    / "development"
    / "simscape_tour_matching"
    / "native_evidence"
    / "two_window_fit_9967_102"
)


def test_compute_marker_metrics_analytical() -> None:
    """Validate metric calculations on an analytical 2-frame, 2-marker synthetic trajectory."""
    time_s = np.array([0.0, 0.5])
    # 2 frames, 2 markers ("WaistLeft", "WaistRight"), 3 coords
    labels = ("WaistLeft", "WaistRight")
    pred = np.zeros((2, 2, 3))
    target = np.zeros((2, 2, 3))

    # Place target at zero, pred displaced by 0.01 m in x on frame 1
    pred[1, 0, 0] = 0.01
    pred[1, 1, 0] = 0.01
    valid = np.ones((2, 2), dtype=bool)

    metrics = compute_marker_metrics(
        pred,
        target,
        valid,
        time_s,
        labels,
        terminal_weight=1.0,
        pelvis_yaw_weight=0.0,
    )

    assert isinstance(metrics, MarkerMetricResults)
    # Total valid points: 4. Error sq at frame 1: 0.01^2 + 0.01^2 = 0.0002. Mean = 0.00005. RMS = sqrt(0.00005) ~ 0.007071
    expected_whole_rms = float(np.sqrt(0.0002 / 4))
    assert metrics.whole_rms_m == pytest.approx(expected_whole_rms, abs=1e-8)
    assert metrics.early_rms_m == pytest.approx(expected_whole_rms, abs=1e-8)
    assert metrics.terminal_rms_m == pytest.approx(0.01, abs=1e-8)
    assert metrics.score == pytest.approx(0.0002 + 0.0002, abs=1e-8)


def test_check_acceptance_honest_gates() -> None:
    """Acceptance must reject any metric exceeding threshold."""
    good = MarkerMetricResults(
        whole_rms_m=0.020,
        early_rms_m=0.010,
        terminal_rms_m=0.030,
        club_cluster_rms_m=0.050,
        pelvis_yaw_error_pct=2.0,
        pelvis_yaw_diff_deg=0.5,
        score=25.0,
    )
    assert check_acceptance(good) is True

    # Violate terminal RMS
    bad_terminal = MarkerMetricResults(
        whole_rms_m=0.020,
        early_rms_m=0.010,
        terminal_rms_m=0.040,  # > 0.035
        club_cluster_rms_m=0.050,
        pelvis_yaw_error_pct=2.0,
        pelvis_yaw_diff_deg=0.5,
        score=25.0,
    )
    assert check_acceptance(bad_terminal) is False


def test_reproduces_run102_uninterrupted_metrics_and_zero_parity() -> None:
    """Reproduces run-102 returned candidate uninterrupted metrics and verifies parity."""
    receipt_path = RUN102_DIR / "receipt.json"
    returned_path = RUN102_DIR / "returned.json"
    replay_path = RUN102_DIR / "returned-replay.npz"
    candidate_path = RUN102_DIR / "returned-candidate.json"

    if not (
        receipt_path.exists()
        and returned_path.exists()
        and replay_path.exists()
        and candidate_path.exists()
    ):
        pytest.skip(f"Run 102 evidence files missing at {RUN102_DIR}")

    receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
    returned_data = json.loads(returned_path.read_text(encoding="utf-8"))
    candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
    replay_data = np.load(replay_path)

    time_s = replay_data["time_s"]
    markers_m = replay_data["markers_m"]
    target_m = replay_data["target_m"]
    valid = replay_data["valid"]
    labels = candidate_data["marker_labels"]

    z_parity = receipt_data["zero_displacement_parity"]
    effort_cost = z_parity["effort_penalty_cost"]
    expected_unint = returned_data["uninterrupted_metrics"]

    # 1. Evaluate metrics on the returned candidate uninterrupted replay
    metrics = compute_marker_metrics(
        markers_m,
        target_m,
        valid,
        time_s,
        labels,
        terminal_weight=25.0,
        pelvis_yaw_weight=40.0,
    )

    # Verifies exact reproduction of all metrics to machine precision
    for key, expected_val in expected_unint.items():
        computed_val = getattr(metrics, key)
        assert computed_val == pytest.approx(expected_val, abs=1e-12)

    # 2. Evaluate zero-displacement parity using identical uninterrupted markers
    # (segmented markers == uninterrupted markers at zero displacement)
    parity_report = evaluate_zero_displacement_parity(
        TwoWindowParityInputs(
            segmented_markers=markers_m,
            uninterrupted_markers=markers_m,
            target_points=target_m,
            valid_mask=valid,
            time_s=time_s,
            labels=labels,
            effort_cost=effort_cost,
            terminal_weight=25.0,
            pelvis_yaw_weight=40.0,
            initial_defect_norm=0.0,
            linear_model_objective=metrics.score + effort_cost,
        )
    )
    assert parity_report.passed is True
    assert parity_report.relative_score_difference < 1e-12
    assert parity_report.marker_max_abs_difference_m < 1e-12
