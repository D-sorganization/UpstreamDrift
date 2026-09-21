"""Tests for tour fit metrics calculations and physical 3D error formulas (TB-02 #10587)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.tour_baselines.fit_metrics import (
    MarkerMetricSummary,
    PhaseMetricSummary,
    PhysicalFitMetrics,
    compute_fit_metrics,
    compute_landmark_signature,
)


def test_hand_calculated_masked_rmse_fixture() -> None:
    """Verify physical 3D Euclidean RMSE against an independently calculated fixture.

    Fixture setup:
    2 frames, 2 markers ("Marker_A", "Marker_B").
    Frame 0:
      Marker_A: pred=(3, 4, 0), obs=(0, 0, 0) => error=5.0, sq=25.0. Valid.
      Marker_B: pred=(1, 1, 1), obs=(1, 1, 1) => error=0.0, sq=0.0. Valid.
    Frame 1:
      Marker_A: pred=(100, 100, 100), obs=(0, 0, 0) => INVALID (masked).
      Marker_B: pred=(1, 2, 2), obs=(0, 0, 0) => error=3.0, sq=9.0. Valid.

    Sum sq = 25.0 + 0.0 + 9.0 = 34.0
    N_valid = 3
    Expected RMSE = sqrt(34 / 3) = 3.366501646...
    """
    pred = np.array(
        [
            [[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]],
            [[100.0, 100.0, 100.0], [1.0, 2.0, 2.0]],
        ],
        dtype=np.float64,
    )
    obs = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    valid = np.array(
        [
            [True, True],
            [False, True],
        ],
        dtype=bool,
    )
    labels = ("Marker_A", "Marker_B")
    time_s = np.array([0.0, 0.1], dtype=np.float64)

    metrics = compute_fit_metrics(
        predicted=pred,
        observed=obs,
        valid=valid,
        labels=labels,
        time_s=time_s,
        optimizer_loss=12.345,
    )

    expected_rmse = math.sqrt(34.0 / 3.0)
    assert math.isclose(metrics.whole_marker_rmse_m, expected_rmse, rel_tol=1e-7)
    assert math.isclose(metrics.max_marker_error_m, 5.0, rel_tol=1e-7)
    assert metrics.n_valid == 3
    assert metrics.n_excluded == 1
    assert metrics.total_observations == 4
    assert math.isclose(metrics.coverage_fraction, 0.75, rel_tol=1e-7)

    # Per-marker metrics
    marker_a = metrics.per_marker["Marker_A"]
    assert marker_a.valid_count == 1
    assert math.isclose(marker_a.rmse_m, 5.0, rel_tol=1e-7)
    assert math.isclose(marker_a.max_m, 5.0, rel_tol=1e-7)

    marker_b = metrics.per_marker["Marker_B"]
    assert marker_b.valid_count == 2
    assert math.isclose(marker_b.rmse_m, math.sqrt(4.5), rel_tol=1e-7)
    assert math.isclose(marker_b.max_m, 3.0, rel_tol=1e-7)

    # Optimizer loss reported separately
    assert metrics.optimizer_weighted_loss == 12.345


def test_empty_valid_set_raises_value_error() -> None:
    """Empty valid set must be rejected, never returned as 0.0 RMSE."""
    pred = np.zeros((2, 2, 3), dtype=np.float64)
    obs = np.zeros((2, 2, 3), dtype=np.float64)
    valid = np.zeros((2, 2), dtype=bool)  # all invalid
    labels = ("M1", "M2")
    time_s = np.array([0.0, 0.1], dtype=np.float64)

    with pytest.raises(ValueError, match="empty valid set"):
        compute_fit_metrics(
            predicted=pred,
            observed=obs,
            valid=valid,
            labels=labels,
            time_s=time_s,
        )


def test_nan_in_valid_predicted_raises_value_error() -> None:
    """NaN in predicted coordinates at a valid index must be rejected."""
    pred = np.array(
        [
            [[np.nan, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    obs = np.zeros((2, 2, 3), dtype=np.float64)
    valid = np.ones((2, 2), dtype=bool)
    labels = ("M1", "M2")
    time_s = np.array([0.0, 0.1], dtype=np.float64)

    with pytest.raises(ValueError, match="non-finite"):
        compute_fit_metrics(
            predicted=pred,
            observed=obs,
            valid=valid,
            labels=labels,
            time_s=time_s,
        )


def test_nan_in_masked_out_predicted_is_ignored() -> None:
    """NaN in predicted coordinates at an INVALID index must NOT trigger failure."""
    pred = np.array(
        [
            [[np.nan, np.nan, np.nan], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    obs = np.zeros((2, 2, 3), dtype=np.float64)
    valid = np.array(
        [
            [False, True],
            [True, True],
        ],
        dtype=bool,
    )
    labels = ("M1", "M2")
    time_s = np.array([0.0, 0.1], dtype=np.float64)

    metrics = compute_fit_metrics(
        predicted=pred,
        observed=obs,
        valid=valid,
        labels=labels,
        time_s=time_s,
    )
    assert metrics.n_valid == 3
    assert np.isfinite(metrics.whole_marker_rmse_m)


def test_landmark_signature_distinction() -> None:
    """Different landmark sets must have distinct cryptographic signatures.

    Landmark sets can never be ranked as equivalent if their observed markers differ.
    """
    sig1 = compute_landmark_signature(["HeadFront", "WaistLeft", "WaistRight"])
    sig2 = compute_landmark_signature(["WaistLeft", "WaistRight", "HeadFront"])
    sig3 = compute_landmark_signature(["WaistLeft", "WaistRight"])

    # Order-independent for the same set
    assert sig1 == sig2
    # Distinct when sets differ
    assert sig1 != sig3
    assert len(sig1) == 64  # SHA-256 hex digest


def test_phase_breakdown_metrics() -> None:
    """Metrics breakdown per biomechanical swing phase."""
    n_frames = 10
    pred = np.ones((n_frames, 2, 3), dtype=np.float64) * 0.05
    obs = np.zeros((n_frames, 2, 3), dtype=np.float64)
    valid = np.ones((n_frames, 2), dtype=bool)
    labels = ("Grip", "Club")
    time_s = np.linspace(0.0, 1.0, n_frames)

    phase_indices = {
        "address": (0, 2),
        "backswing": (2, 5),
        "downswing": (5, 7),
        "impact": (7, 8),
        "follow_through": (8, 10),
    }

    metrics = compute_fit_metrics(
        predicted=pred,
        observed=obs,
        valid=valid,
        labels=labels,
        time_s=time_s,
        phase_indices=phase_indices,
    )

    assert "address" in metrics.per_phase
    assert "impact" in metrics.per_phase
    address_phase = metrics.per_phase["address"]
    assert address_phase.valid_count == 4  # 2 frames * 2 markers
    assert math.isclose(address_phase.rmse_m, math.sqrt(3 * 0.05**2), rel_tol=1e-5)


def test_in_plane_and_out_of_plane_decomposition() -> None:
    """Errors decomposed along normal plane vector."""
    # Plane normal is Z axis (0, 0, 1)
    pred = np.array([[[1.0, 0.0, 2.0]]], dtype=np.float64)
    obs = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    valid = np.array([[True]], dtype=bool)
    labels = ("Marker1",)
    time_s = np.array([0.0], dtype=np.float64)

    metrics = compute_fit_metrics(
        predicted=pred,
        observed=obs,
        valid=valid,
        labels=labels,
        time_s=time_s,
        plane_normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )

    # In-plane error vector = (1.0, 0.0, 0.0) => norm = 1.0
    # Out-of-plane residual vector = (0.0, 0.0, 2.0) => norm = 2.0
    # 3D error = sqrt(1^2 + 2^2) = sqrt(5)
    assert math.isclose(metrics.in_plane_rmse_m or 0.0, 1.0, rel_tol=1e-5)
    assert math.isclose(metrics.out_of_plane_residual_m or 0.0, 2.0, rel_tol=1e-5)
    assert math.isclose(metrics.whole_marker_rmse_m, math.sqrt(5.0), rel_tol=1e-5)
