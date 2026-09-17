"""Unit tests for Shadow Tracker ambiguity quantification and evidence gates (ST-09, #10132)."""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.shadow_tracker.contracts import (
    CandidateResult,
    FitRequest,
    FrameObservation,
    ReplayAudit,
    ResultBundle,
)
from shared.python.shadow_tracker.evaluation import (
    AblationResult,
    AmbiguityReport,
    CoverageMetric,
    GateProfile,
    QuantityConfidence,
    ablate_parameter_sensitivity,
    audit_gate_profile,
    classify_evidence_quality,
    compute_empirical_coverage,
    create_evaluated_result_bundle,
    detect_silhouette_ambiguity,
    evaluate_candidate_evidence,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


def _dummy_replay_audit(
    *,
    is_physically_accepted: bool = True,
    reset_count: int = 1,
    max_grip_translation_error_m: float = 0.002,
    max_grip_rotation_error_rad: float = 0.01,
) -> ReplayAudit:
    return ReplayAudit(
        schema_version="shadow-tracker/replay-audit/1.0.0",
        candidate_id="cand-001",
        reset_count=reset_count,
        integrator_name="rk4",
        integrator_version="1.0",
        coverage_start_s=0.0,
        coverage_end_s=0.1,
        max_grip_translation_error_m=max_grip_translation_error_m,
        max_grip_rotation_error_rad=max_grip_rotation_error_rad,
        is_physically_accepted=is_physically_accepted,
    )


def _dummy_candidate(
    *,
    candidate_id: str = "cand-001",
    is_accepted: bool = True,
    replay_audit: ReplayAudit | None = None,
    trajectory: tuple[tuple[float, ...], ...] | None = None,
) -> CandidateResult:
    audit = (
        replay_audit
        if replay_audit is not None
        else (_dummy_replay_audit() if is_accepted else None)
    )
    traj = (
        trajectory
        if trajectory is not None
        else ((0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.02, 0.0, 0.0))
    )
    return CandidateResult(
        schema_version="shadow-tracker/candidate-result/1.0.0",
        candidate_id=candidate_id,
        request_id="req-001",
        initial_state=(0.0, 0.0, 0.0),
        trajectory=traj,
        diagnostics={"mean_iou": 0.96, "club_error": 0.005},
        uncertainty_method="empirical_holdout",
        replay_audit=audit,
        is_accepted=is_accepted,
    )


def _dummy_fit_request() -> FitRequest:
    return FitRequest(
        schema_version="shadow-tracker/fit-request/1.0.0",
        request_id="req-001",
        shot_id="shot-001",
        model_hash="0" * 64,
        candidate_count=1,
        objective_profile="silhouette_and_physics_v1",
        time_window_start_pts=0,
        time_window_end_pts=1000,
        budget_seconds=10.0,
        engine_capability_requirement=("forward_dynamics",),
    )


def _dummy_observation(
    *,
    frame_id: str = "frame-001",
    is_timing_exact: bool = True,
    physical_time_s: float | None = 0.01,
    has_club_mask: bool = True,
) -> FrameObservation:
    return FrameObservation(
        schema_version="shadow-tracker/frame-observation/1.0.0",
        shot_id="shot-001",
        camera_id="cam-001",
        frame_id=frame_id,
        pts_ticks=100,
        timebase_numerator=1,
        timebase_denominator=10000,
        physical_time_s=physical_time_s,
        physical_time_reason="exact_sync" if is_timing_exact else "uncalibrated_video",
        body_mask_ref="mask-body-001",
        club_mask_ref="mask-club-001" if has_club_mask else "mask-club-unobserved",
        valid_mask_ref="mask-valid-001",
        confidence_provenance="ground_truth",
        timing_mode="hardware_genlock" if is_timing_exact else "nominal_video",
        is_timing_exact=is_timing_exact,
        clock_evidence="hardware_sync",
        decoder_name="nvdec",
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_silhouette_ambiguity_distinct_3d_poses_indistinguishable_silhouette() -> None:
    """ST-09 First Test 1: Two different 3D poses with indistinguishable silhouette."""
    cand_a = _dummy_candidate(
        candidate_id="cand-A",
        trajectory=((0.0, 0.0, 2.0), (0.1, 0.0, 2.0)),
    )
    cand_b = _dummy_candidate(
        candidate_id="cand-B",
        trajectory=((0.0, 0.0, 2.5), (0.1, 0.0, 2.5)),
    )

    class MockAmbiguousRenderer:
        def render_trajectory_silhouettes(
            self, traj: tuple[tuple[float, ...], ...]
        ) -> tuple[np.ndarray, ...]:
            mask = np.zeros((10, 10), dtype=bool)
            mask[3:7, 3:7] = True
            return (mask, mask)

    report = detect_silhouette_ambiguity(
        candidate_a=cand_a,
        candidate_b=cand_b,
        renderer=MockAmbiguousRenderer(),
        pose_divergence_threshold_m=0.05,
        silhouette_iou_threshold=0.02,
    )

    assert isinstance(report, AmbiguityReport)
    assert report.has_silhouette_ambiguity is True
    assert report.max_pose_divergence_m >= 0.50
    assert report.silhouette_difference_iou <= 0.01
    assert "cand-A" in report.candidate_ids
    assert "cand-B" in report.candidate_ids


def test_unknown_time_blocks_si_kinetics() -> None:
    """ST-09 First Test 2: Unknown physical time strictly blocks SI kinetics qualification."""
    req = _dummy_fit_request()
    cand = _dummy_candidate(is_accepted=True)
    audits = (cand.replay_audit,)

    inexact_observations = (
        _dummy_observation(
            frame_id="frame-001", is_timing_exact=False, physical_time_s=None
        ),
        _dummy_observation(
            frame_id="frame-002", is_timing_exact=False, physical_time_s=None
        ),
    )

    quality = classify_evidence_quality(
        request=req,
        candidate=cand,
        audits=audits,  # type: ignore[arg-type]
        observations=inexact_observations,
    )

    assert quality != "validated_profile"
    assert quality == "kinematic_only"

    exact_observations = (
        _dummy_observation(
            frame_id="frame-001", is_timing_exact=True, physical_time_s=0.01
        ),
        _dummy_observation(
            frame_id="frame-002", is_timing_exact=True, physical_time_s=0.02
        ),
    )
    quality_exact = classify_evidence_quality(
        request=req,
        candidate=cand,
        audits=audits,  # type: ignore[arg-type]
        observations=exact_observations,
    )
    assert quality_exact == "validated_profile"


def test_hidden_club_case_does_not_return_zero_error() -> None:
    """ST-09 First Test 3: Hidden / unobserved club does not return zero error."""
    obs_no_club = (
        _dummy_observation(frame_id="frame-001", has_club_mask=False),
        _dummy_observation(frame_id="frame-002", has_club_mask=False),
    )
    cand = _dummy_candidate(is_accepted=True)

    report = evaluate_candidate_evidence(
        candidate=cand,
        observations=obs_no_club,
        profile=GateProfile.default_development_profile(),
    )

    club_error = report.metrics.get("club_contour_error")
    assert club_error is not None
    assert club_error > 0.0
    assert any("club" in reason.lower() for reason in report.abstention_reasons)


def test_failed_physical_candidate_cannot_be_exported_as_validated() -> None:
    """Gate G4: Failed physical candidates cannot be exported as validated."""
    failed_audit = _dummy_replay_audit(
        is_physically_accepted=False,
        max_grip_translation_error_m=0.08,
    )

    with pytest.raises(
        ValueError,
        match="is_accepted cannot be true when replay audit is not physically accepted",
    ):
        _dummy_candidate(is_accepted=True, replay_audit=failed_audit)

    rejected_cand = _dummy_candidate(is_accepted=False, replay_audit=failed_audit)
    req = _dummy_fit_request()
    obs = (_dummy_observation(frame_id="frame-001"),)

    quality = classify_evidence_quality(
        request=req,
        candidate=rejected_cand,
        audits=(failed_audit,),
        observations=obs,
    )
    assert quality in ("insufficient_evidence", "kinematic_only")
    assert quality != "validated_profile"


def test_curvature_or_multistart_cannot_be_labeled_calibrated_confidence() -> None:
    """Do Not: Call optimizer curvature or multistart spread calibrated confidence."""
    conf = QuantityConfidence(
        quantity_name="lead_wrist_torque_nm",
        nominal_value=45.2,
        posterior_lower=42.0,
        posterior_upper=48.5,
        sensitivity_min=35.0,
        sensitivity_max=55.0,
        is_calibrated=False,
    )

    assert conf.is_calibrated is False
    assert conf.posterior_width == pytest.approx(6.5)
    assert conf.sensitivity_width == pytest.approx(20.0)
    assert conf.sensitivity_min < conf.posterior_lower
    assert conf.sensitivity_max > conf.posterior_upper


def test_parameter_ablations_camera_timing_mass_contact() -> None:
    """ST-09 Acceptance: Camera, timing, mass, and contact ablations."""
    nominal = _dummy_candidate(candidate_id="nominal")

    perturbed = {
        "camera": _dummy_candidate(
            candidate_id="ablate-camera", trajectory=((0.0, 0.01, 0.0),)
        ),
        "timing": _dummy_candidate(
            candidate_id="ablate-timing", trajectory=((0.0, 0.0, 0.01),)
        ),
        "mass": _dummy_candidate(
            candidate_id="ablate-mass", trajectory=((0.02, 0.0, 0.0),)
        ),
        "contact": _dummy_candidate(
            candidate_id="ablate-contact", trajectory=((0.01, 0.01, 0.0),)
        ),
    }

    ablations = ablate_parameter_sensitivity(nominal, perturbed)
    assert len(ablations) == 4
    params = {a.parameter for a in ablations}
    assert params == {"camera", "timing", "mass", "contact"}
    for a in ablations:
        assert isinstance(a, AblationResult)
        assert a.relative_metric_shift >= 0.0


def test_empirical_coverage_and_interval_width() -> None:
    """Gate G5: Empirical coverage and interval width evaluation on held-out truth."""
    intervals = tuple((float(i) - 1.0, float(i) + 1.0) for i in range(10))
    truths = tuple(float(i) if i < 9 else 15.0 for i in range(10))

    metric = compute_empirical_coverage(intervals, truths, nominal_rate=0.90)
    assert isinstance(metric, CoverageMetric)
    assert metric.empirical_coverage == pytest.approx(0.90)
    assert metric.nominal_rate == pytest.approx(0.90)
    assert metric.average_interval_width == pytest.approx(2.0)
    assert metric.is_well_calibrated is True


def test_gate_audit_g0_through_g5() -> None:
    """Validate systematic gate profile checks from G0 through G5."""
    profile = GateProfile.default_development_profile()
    cand = _dummy_candidate(is_accepted=True)
    obs = (
        _dummy_observation(frame_id="frame-001"),
        _dummy_observation(frame_id="frame-002"),
    )

    passed, statuses = audit_gate_profile(cand, obs, profile)
    assert passed is True
    gate_ids = {s.gate_id for s in statuses}
    assert {"G0", "G1", "G2", "G4", "G5"}.issubset(gate_ids)


def test_evaluated_result_bundle_roundtrip() -> None:
    """Atomic ResultBundle creation and serialization round trip."""
    req = _dummy_fit_request()
    cand = _dummy_candidate(is_accepted=True)
    obs = (_dummy_observation(frame_id="frame-001"),)

    bundle = create_evaluated_result_bundle(
        bundle_id="bundle-001",
        request=req,
        candidates=(cand,),
        observations=obs,
    )

    assert isinstance(bundle, ResultBundle)
    assert bundle.evidence_quality == "validated_profile"
    assert bundle.execution_status == "completed"

    payload = bundle.to_dict()
    reloaded = ResultBundle.from_dict(payload)
    assert reloaded.bundle_id == bundle.bundle_id
    assert reloaded.evidence_quality == bundle.evidence_quality
