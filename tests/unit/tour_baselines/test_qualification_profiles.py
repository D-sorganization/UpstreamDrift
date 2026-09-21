"""Tests for versioned baseline qualification profiles (TB-02 #10587)."""

from __future__ import annotations

import pytest

from src.shared.python.tour_baselines.metrics import (
    DynamicFeasibilityStatus,
    KinematicAccuracyStatus,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverStatus,
    TourFitMetrics,
)
from src.shared.python.tour_baselines.models import ModelClass
from src.shared.python.tour_baselines.packages import NativeReplayEvidence
from src.shared.python.tour_baselines.qualification import (
    DoublePendulumPlanarProfile,
    FullBodyAuthoritativeProfile,
    QualificationVerdict,
    TriplePendulumPlanarProfile,
    UpperBodyGolferProfile,
    evaluate_qualification,
)

pytestmark = pytest.mark.unit


def _make_dummy_metrics(
    whole_rmse: float = 0.030,
    in_plane_rmse: float = 0.025,
    clubhead_rmse: float = 0.040,
) -> TourFitMetrics:
    return TourFitMetrics(
        observed_valid_denominator=1000,
        excluded_sample_count=20,
        coverage_fraction=0.98,
        whole_marker_rmse_m=whole_rmse,
        p95_marker_error_m=whole_rmse * 1.5,
        max_marker_error_m=whole_rmse * 2.0,
        per_marker_rmse_m={"ClubHead": clubhead_rmse, "Grip": 0.020},
        per_phase_rmse_m={"downswing": whole_rmse, "impact": whole_rmse},
        impact_marker_error_m=whole_rmse,
        endpoint_clubhead_rmse_m=clubhead_rmse,
        optimizer_weighted_loss=0.123,
        original_frame_rmse_m=whole_rmse,
        in_plane_rmse_m=in_plane_rmse,
        out_of_plane_residual_m=0.015,
        landmarks_hash="0" * 64,
    )


def test_double_pendulum_profile_qualification() -> None:
    """Double pendulum planar model qualifies under its own DoF-calibrated profile."""
    profile = DoublePendulumPlanarProfile()
    assert profile.model_class == ModelClass.DOUBLE_PENDULUM_PLANAR
    assert profile.gate_version == "tour-qualification/1.0.0"

    metrics = _make_dummy_metrics(in_plane_rmse=0.080, clubhead_rmse=0.090)
    replay = NativeReplayEvidence(
        engine="mujoco",
        replay_command="python -m src.engines.physics_engines.mujoco.replay",
        replay_sha256="a" * 64,
        verified_reproduced=True,
    )

    verdict = evaluate_qualification(profile, metrics, replay)
    assert verdict.scientific_status == ScientificQualificationStatus.QUALIFIED
    assert verdict.is_qualified is True


def test_double_pendulum_fails_when_clubhead_rmse_exceeds_gate() -> None:
    """Double pendulum fails qualification if in-plane clubhead error exceeds threshold."""
    profile = DoublePendulumPlanarProfile()
    metrics = _make_dummy_metrics(in_plane_rmse=0.150, clubhead_rmse=0.160)
    replay = NativeReplayEvidence(
        engine="mujoco",
        replay_command="python -m replay",
        replay_sha256="a" * 64,
        verified_reproduced=True,
    )

    verdict = evaluate_qualification(profile, metrics, replay)
    assert verdict.scientific_status == ScientificQualificationStatus.UNQUALIFIED
    assert verdict.is_qualified is False


def test_missing_native_replay_cannot_qualify() -> None:
    """A complete manifest with missing or unverified native replay cannot qualify."""
    profile = DoublePendulumPlanarProfile()
    metrics = _make_dummy_metrics(in_plane_rmse=0.050, clubhead_rmse=0.050)
    # Replay is unverified
    replay = NativeReplayEvidence(
        engine="mujoco",
        replay_command="",
        replay_sha256="",
        verified_reproduced=False,
    )

    verdict = evaluate_qualification(profile, metrics, replay)
    assert verdict.scientific_status == ScientificQualificationStatus.UNQUALIFIED
    assert "missing native replay" in verdict.reason.lower()


def test_synthetic_test_data_cannot_be_promoted() -> None:
    """Synthetic packages are visibly test data and cannot be promoted to tour baselines."""
    profile = UpperBodyGolferProfile()
    metrics = _make_dummy_metrics(whole_rmse=0.035)
    replay = NativeReplayEvidence(
        engine="mujoco",
        replay_command="python -m replay",
        replay_sha256="a" * 64,
        verified_reproduced=True,
    )

    verdict = evaluate_qualification(
        profile, metrics, replay, is_synthetic_test_data=True
    )
    assert verdict.product_status != ProductPromotionStatus.PROMOTED
    assert verdict.product_status == ProductPromotionStatus.REJECTED
    assert "synthetic" in verdict.reason.lower()


def test_triple_pendulum_and_upper_body_profiles_have_distinct_thresholds() -> None:
    """Reduced models have distinct frozen gates that reflect their modeled kinematics."""
    p_triple = TriplePendulumPlanarProfile()
    p_upper = UpperBodyGolferProfile()

    assert p_triple.thresholds["max_in_plane_clubhead_rmse_m"] == 0.085
    assert p_upper.thresholds["max_upper_body_rmse_m"] == 0.050
