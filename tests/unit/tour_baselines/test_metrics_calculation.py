"""Tests for physical 3D marker tracking error metrics and formulas (TB-02 #10587)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.tour_baselines.metrics import (
    DynamicFeasibilityStatus,
    KinematicAccuracyStatus,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverStatus,
    TourFitMetrics,
    compute_tour_fit_metrics,
    hash_landmark_set,
)

pytestmark = pytest.mark.unit


def test_tiny_masked_rmse_fixture_independent_calculation() -> None:
    """Validate physical 3D Euclidean marker RMSE formula with a hand-calculated fixture."""
    # 2 frames, 2 markers, 3D
    # Predicted positions:
    # Frame 0: M1=[0.0, 0.0, 0.0], M2=[1.0, 1.0, 1.0]
    # Frame 1: M1=[0.0, 0.0, 1.0], M2=[2.0, 2.0, 2.0]
    pred = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 1.0], [2.0, 2.0, 2.0]],
        ],
        dtype=np.float64,
    )

    # Observed positions:
    # Frame 0: M1=[0.0, 0.0, 0.0] (diff=0), M2=[1.0, 2.0, 1.0] (diff=[0, 1, 0], err=1.0)
    # Frame 1: M1=[0.0, 0.0, 3.0] (diff=[0, 0, 2], err=2.0), M2=[np.nan, np.nan, np.nan] (masked)
    obs = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            [[0.0, 0.0, 3.0], [np.nan, np.nan, np.nan]],
        ],
        dtype=np.float64,
    )

    # Valid points:
    # (0, 0): err^2 = 0^2 = 0
    # (0, 1): err^2 = 1.0^2 = 1.0
    # (1, 0): err^2 = 2.0^2 = 4.0
    # (1, 1): masked (excluded)
    # Sum of squared errors = 0 + 1 + 4 = 5.0
    # N_valid = 3
    # Whole RMSE = sqrt(5.0 / 3) ~= 1.2909944487358056
    expected_rmse = math.sqrt(5.0 / 3.0)
    marker_labels = ("M1", "M2")
    time_s = (0.0, 0.01)

    metrics = compute_tour_fit_metrics(
        predicted_points_m=pred,
        observed_points_m=obs,
        marker_labels=marker_labels,
        time_s=time_s,
        optimizer_loss=42.5,
    )

    assert metrics.observed_valid_denominator == 3
    assert metrics.excluded_sample_count == 1
    assert math.isclose(metrics.coverage_fraction, 3 / 4, rel_tol=1e-6)
    assert math.isclose(metrics.whole_marker_rmse_m, expected_rmse, rel_tol=1e-6)
    assert math.isclose(metrics.max_marker_error_m, 2.0, rel_tol=1e-6)
    assert metrics.optimizer_weighted_loss == 42.5

    # Per-marker RMSE:
    # M1: errs = [0.0, 2.0], sum sq = 4.0, count = 2 -> sqrt(4/2) = sqrt(2) ~= 1.41421356
    # M2: errs = [1.0], sum sq = 1.0, count = 1 -> sqrt(1/1) = 1.0
    assert math.isclose(metrics.per_marker_rmse_m["M1"], math.sqrt(2.0), rel_tol=1e-6)
    assert math.isclose(metrics.per_marker_rmse_m["M2"], 1.0, rel_tol=1e-6)


def test_landmark_set_hash_and_inequivalence() -> None:
    """Verify different landmark sets produce different hashes and cannot be conflated."""
    set_a = ("Clavicle", "RShoulder", "LShoulder")
    set_b = ("Clavicle", "RShoulder", "Grip")
    set_a_reordered = ("LShoulder", "Clavicle", "RShoulder")

    hash_a = hash_landmark_set(set_a)
    hash_b = hash_landmark_set(set_b)
    hash_a_reordered = hash_landmark_set(set_a_reordered)

    assert hash_a != hash_b
    # Order-independent canonical landmark set identity
    assert hash_a == hash_a_reordered
    assert len(hash_a) == 64


def test_reject_empty_valid_set() -> None:
    """Empty valid observations set must raise ValueError."""
    pred = np.zeros((2, 2, 3), dtype=np.float64)
    obs = np.full((2, 2, 3), np.nan, dtype=np.float64)

    with pytest.raises(ValueError, match="No valid observations"):
        compute_tour_fit_metrics(
            predicted_points_m=pred,
            observed_points_m=obs,
            marker_labels=("M1", "M2"),
            time_s=(0.0, 0.01),
        )


def test_reject_dimension_mismatch() -> None:
    """Mismatched shapes between predicted and observed points must raise ValueError."""
    pred = np.zeros((3, 2, 3), dtype=np.float64)
    obs = np.zeros((2, 2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_tour_fit_metrics(
            predicted_points_m=pred,
            observed_points_m=obs,
            marker_labels=("M1", "M2"),
            time_s=(0.0, 0.01, 0.02),
        )


def test_unconflated_status_enums() -> None:
    """Ensure the 5 statuses remain strictly separate types and values."""
    assert SolverStatus.CONVERGED.value == "converged"
    assert KinematicAccuracyStatus.ACCURATE.value == "accurate"
    assert DynamicFeasibilityStatus.FEASIBLE.value == "feasible"
    assert ScientificQualificationStatus.QUALIFIED.value == "qualified"
    assert ProductPromotionStatus.PROMOTED.value == "promoted"

    # Verify they cannot be mistakenly equated
    assert SolverStatus.CONVERGED != KinematicAccuracyStatus.ACCURATE
    assert ScientificQualificationStatus.QUALIFIED != ProductPromotionStatus.PROMOTED
