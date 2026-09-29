"""Unit tests for Shadow Tracker immutable evidence contracts and service protocols."""

from __future__ import annotations

from fractions import Fraction
import pytest

pytestmark = pytest.mark.unit

from shared.python.shadow_tracker.contracts import (
    CANDIDATE_RESULT_SCHEMA_VERSION,
    CAMERA_TRACK_SCHEMA_VERSION,
    FIT_REQUEST_SCHEMA_VERSION,
    FRAME_OBSERVATION_SCHEMA_VERSION,
    REPLAY_AUDIT_SCHEMA_VERSION,
    RESULT_BUNDLE_SCHEMA_VERSION,
    SHOT_SCHEMA_VERSION,
    SUBJECT_BINDING_SCHEMA_VERSION,
    CameraTrack,
    CandidateResult,
    FitRequest,
    ForwardModel,
    FrameObservation,
    ModelCapabilities,
    RenderRequest,
    RenderResult,
    ReplayAudit,
    ResultBundle,
    RolloutRequest,
    RolloutResult,
    SegmentationRequest,
    SegmentationResult,
    Segmenter,
    ShadowTrackerService,
    Shot,
    SilhouetteRenderer,
    SubjectModelBinding,
)

SAMPLE_SHA256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


# ---------------------------------------------------------------------------
# Shot Contract Tests
# ---------------------------------------------------------------------------


def test_shot_contract_valid_and_roundtrip() -> None:
    shot = Shot(
        schema_version=SHOT_SCHEMA_VERSION,
        asset_id="asset-1",
        shot_id="shot-1",
        start_pts=0,
        end_pts=100,
        start_frame_id="f0",
        end_frame_id="f100",
        subject_id="sub-1",
        swing_id="swing-1",
        camera_id="cam-1",
        cuts=((20, 25),),
        transforms=("mirror_x",),
    )
    assert shot.schema_version == SHOT_SCHEMA_VERSION
    assert shot.duration_pts == 100
    d = shot.to_dict()
    assert d["start_pts"] == 0
    assert d["end_pts"] == 100
    assert d["cuts"] == [[20, 25]]
    assert d["transforms"] == ["mirror_x"]

    restored = Shot.from_dict(d)
    assert restored == shot
    assert restored.cuts == ((20, 25),)
    assert restored.transforms == ("mirror_x",)


def test_shot_validation_rejects_malformed_data() -> None:
    # Invalid schema version
    with pytest.raises(ValueError, match="schema_version"):
        Shot(
            schema_version="invalid-version",
            asset_id="a",
            shot_id="s",
            start_pts=0,
            end_pts=10,
            start_frame_id="f0",
            end_frame_id="f10",
            subject_id="sub",
            swing_id="sw",
            camera_id="cam",
            cuts=(),
            transforms=(),
        )

    # Inverted PTS interval
    with pytest.raises(ValueError, match="start_pts"):
        Shot(
            schema_version=SHOT_SCHEMA_VERSION,
            asset_id="a",
            shot_id="s",
            start_pts=10,
            end_pts=5,
            start_frame_id="f0",
            end_frame_id="f10",
            subject_id="sub",
            swing_id="sw",
            camera_id="cam",
            cuts=(),
            transforms=(),
        )

    # Cut outside interval
    with pytest.raises(ValueError, match="cuts"):
        Shot(
            schema_version=SHOT_SCHEMA_VERSION,
            asset_id="a",
            shot_id="s",
            start_pts=0,
            end_pts=10,
            start_frame_id="f0",
            end_frame_id="f10",
            subject_id="sub",
            swing_id="sw",
            camera_id="cam",
            cuts=((5, 15),),
            transforms=(),
        )


# ---------------------------------------------------------------------------
# FrameObservation Contract Tests
# ---------------------------------------------------------------------------


def test_frame_observation_contract_valid_and_roundtrip() -> None:
    obs = FrameObservation(
        schema_version=FRAME_OBSERVATION_SCHEMA_VERSION,
        shot_id="shot-1",
        camera_id="cam-1",
        frame_id="frame-1",
        pts_ticks=12,
        timebase_numerator=1,
        timebase_denominator=24,
        physical_time_s=0.5,
        physical_time_reason="measured_strobe",
        body_mask_ref="mask-body-1",
        club_mask_ref="mask-club-1",
        valid_mask_ref="mask-valid-1",
        confidence_provenance="manual_review",
    )
    assert obs.presentation_time == Fraction(12, 24)
    assert obs.physical_time_s == 0.5

    d = obs.to_dict()
    assert d["pts_ticks"] == 12
    assert d["timebase_numerator"] == 1
    assert d["timebase_denominator"] == 24
    restored = FrameObservation.from_dict(d)
    assert restored == obs


def test_frame_observation_validation_rejects_malformed_data() -> None:
    # Unreduced fraction
    with pytest.raises(ValueError, match="timebase"):
        FrameObservation(
            schema_version=FRAME_OBSERVATION_SCHEMA_VERSION,
            shot_id="s",
            camera_id="c",
            frame_id="f",
            pts_ticks=0,
            timebase_numerator=2,
            timebase_denominator=48,
            physical_time_s=None,
            physical_time_reason="unknown_sync",
            body_mask_ref="b",
            club_mask_ref="cl",
            valid_mask_ref="v",
            confidence_provenance="test",
        )

    # Empty reason when physical_time_s is None
    with pytest.raises(ValueError, match="physical_time_reason"):
        FrameObservation(
            schema_version=FRAME_OBSERVATION_SCHEMA_VERSION,
            shot_id="s",
            camera_id="c",
            frame_id="f",
            pts_ticks=0,
            timebase_numerator=1,
            timebase_denominator=24,
            physical_time_s=None,
            physical_time_reason="",
            body_mask_ref="b",
            club_mask_ref="cl",
            valid_mask_ref="v",
            confidence_provenance="test",
        )

    # Rejection of string or boolean physical_time_s
    with pytest.raises(TypeError, match="physical_time_s"):
        FrameObservation(
            schema_version=FRAME_OBSERVATION_SCHEMA_VERSION,
            shot_id="s",
            camera_id="c",
            frame_id="f",
            pts_ticks=0,
            timebase_numerator=1,
            timebase_denominator=24,
            physical_time_s="0.5",  # type: ignore[arg-type]
            physical_time_reason="test",
            body_mask_ref="b",
            club_mask_ref="cl",
            valid_mask_ref="v",
            confidence_provenance="test",
        )


# ---------------------------------------------------------------------------
# CameraTrack Contract Tests
# ---------------------------------------------------------------------------


def test_camera_track_contract_valid_and_roundtrip() -> None:
    track = CameraTrack(
        schema_version=CAMERA_TRACK_SCHEMA_VERSION,
        camera_id="cam-1",
        shot_id="shot-1",
        frame_convention="world_to_camera",
        track_times=(0, 1, 2),
        status="measured",
        uncertainty=0.01,
        image_transforms=("identity",),
    )
    assert track.status == "measured"
    d = track.to_dict()
    assert d["track_times"] == [0, 1, 2]
    restored = CameraTrack.from_dict(d)
    assert restored == track


def test_camera_track_validation_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="status"):
        CameraTrack(
            schema_version=CAMERA_TRACK_SCHEMA_VERSION,
            camera_id="c",
            shot_id="s",
            frame_convention="world_to_camera",
            track_times=(),
            status="invalid_status",  # type: ignore[arg-type]
            uncertainty=None,
            image_transforms=(),
        )

    with pytest.raises(ValueError, match="uncertainty"):
        CameraTrack(
            schema_version=CAMERA_TRACK_SCHEMA_VERSION,
            camera_id="c",
            shot_id="s",
            frame_convention="world_to_camera",
            track_times=(),
            status="estimated",
            uncertainty=-0.5,
            image_transforms=(),
        )


# ---------------------------------------------------------------------------
# SubjectModelBinding Contract Tests
# ---------------------------------------------------------------------------


def test_subject_model_binding_valid_and_roundtrip() -> None:
    binding = SubjectModelBinding(
        schema_version=SUBJECT_BINDING_SCHEMA_VERSION,
        subject_id="sub-1",
        model_hash=SAMPLE_SHA256,
        joint_ids=("pelvis", "torso"),
        body_ids=("pelvis_body", "torso_body"),
        visual_envelope={"height_m": 1.82, "shoulder_width_m": 0.45},
        mass_kg=78.5,
        scale_evidence="calibrated_wand",
        handedness="right",
    )
    assert binding.handedness == "right"
    assert binding.mass_kg == 78.5
    d = binding.to_dict()
    restored = SubjectModelBinding.from_dict(d)
    assert restored == binding


def test_subject_model_binding_validation() -> None:
    with pytest.raises(ValueError, match="mass_kg"):
        SubjectModelBinding(
            schema_version=SUBJECT_BINDING_SCHEMA_VERSION,
            subject_id="sub",
            model_hash=SAMPLE_SHA256,
            joint_ids=(),
            body_ids=(),
            visual_envelope={},
            mass_kg=-10.0,
            scale_evidence="test",
            handedness="right",
        )

    with pytest.raises(ValueError, match="handedness"):
        SubjectModelBinding(
            schema_version=SUBJECT_BINDING_SCHEMA_VERSION,
            subject_id="sub",
            model_hash=SAMPLE_SHA256,
            joint_ids=(),
            body_ids=(),
            visual_envelope={},
            mass_kg=75.0,
            scale_evidence="test",
            handedness="ambidextrous",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# ReplayAudit & CandidateResult Contract Tests (DbC Invariant Enforcement)
# ---------------------------------------------------------------------------


def test_replay_audit_contract_and_invariants() -> None:
    audit = ReplayAudit(
        schema_version=REPLAY_AUDIT_SCHEMA_VERSION,
        candidate_id="cand-1",
        reset_count=1,
        integrator_name="rk45",
        integrator_version="1.0",
        coverage_start_s=0.0,
        coverage_end_s=1.2,
        max_grip_translation_error_m=0.005,
        max_grip_rotation_error_rad=0.02,
        is_physically_accepted=True,
    )
    assert audit.is_physically_accepted is True
    d = audit.to_dict()
    restored = ReplayAudit.from_dict(d)
    assert restored == audit

    # Invariant: reset_count != 1 cannot be accepted for reference-free execution
    with pytest.raises(ValueError, match="reset_count"):
        ReplayAudit(
            schema_version=REPLAY_AUDIT_SCHEMA_VERSION,
            candidate_id="cand-1",
            reset_count=5,  # Multiple resets / per-frame state injection
            integrator_name="rk45",
            integrator_version="1.0",
            coverage_start_s=0.0,
            coverage_end_s=1.2,
            max_grip_translation_error_m=0.005,
            max_grip_rotation_error_rad=0.02,
            is_physically_accepted=True,
        )


def test_candidate_result_acceptance_requires_valid_replay_audit() -> None:
    # Case 1: is_accepted=False is allowed without replay audit
    res_unreviewed = CandidateResult(
        schema_version=CANDIDATE_RESULT_SCHEMA_VERSION,
        candidate_id="cand-1",
        request_id="req-1",
        initial_state=(0.0, 0.0, 0.0),
        trajectory=((0.0, 0.0), (1.0, 1.0)),
        diagnostics={"iterations": 50},
        uncertainty_method="laplace",
        replay_audit=None,
        is_accepted=False,
    )
    assert res_unreviewed.is_accepted is False

    # Case 2: Invariant violation - is_accepted=True without replay audit MUST fail
    with pytest.raises(ValueError, match="replay audit"):
        CandidateResult(
            schema_version=CANDIDATE_RESULT_SCHEMA_VERSION,
            candidate_id="cand-1",
            request_id="req-1",
            initial_state=(0.0, 0.0, 0.0),
            trajectory=((0.0, 0.0), (1.0, 1.0)),
            diagnostics={},
            uncertainty_method="laplace",
            replay_audit=None,
            is_accepted=True,
        )

    # Case 3: Invariant violation - is_accepted=True with unaccepted replay audit MUST fail
    unaccepted_audit = ReplayAudit(
        schema_version=REPLAY_AUDIT_SCHEMA_VERSION,
        candidate_id="cand-1",
        reset_count=1,
        integrator_name="rk45",
        integrator_version="1.0",
        coverage_start_s=0.0,
        coverage_end_s=1.2,
        max_grip_translation_error_m=1.266,
        max_grip_rotation_error_rad=0.5,
        is_physically_accepted=False,
    )
    with pytest.raises(ValueError, match="replay audit"):
        CandidateResult(
            schema_version=CANDIDATE_RESULT_SCHEMA_VERSION,
            candidate_id="cand-1",
            request_id="req-1",
            initial_state=(0.0, 0.0, 0.0),
            trajectory=((0.0, 0.0), (1.0, 1.0)),
            diagnostics={},
            uncertainty_method="laplace",
            replay_audit=unaccepted_audit,
            is_accepted=True,
        )

    # Case 4: is_accepted=True with valid passing replay audit succeeds
    accepted_audit = ReplayAudit(
        schema_version=REPLAY_AUDIT_SCHEMA_VERSION,
        candidate_id="cand-1",
        reset_count=1,
        integrator_name="rk45",
        integrator_version="1.0",
        coverage_start_s=0.0,
        coverage_end_s=1.2,
        max_grip_translation_error_m=0.005,
        max_grip_rotation_error_rad=0.02,
        is_physically_accepted=True,
    )
    res_accepted = CandidateResult(
        schema_version=CANDIDATE_RESULT_SCHEMA_VERSION,
        candidate_id="cand-1",
        request_id="req-1",
        initial_state=(0.0, 0.0, 0.0),
        trajectory=((0.0, 0.0), (1.0, 1.0)),
        diagnostics={},
        uncertainty_method="laplace",
        replay_audit=accepted_audit,
        is_accepted=True,
    )
    assert res_accepted.is_accepted is True
    d = res_accepted.to_dict()
    restored = CandidateResult.from_dict(d)
    assert restored == res_accepted


# ---------------------------------------------------------------------------
# FitRequest & ResultBundle Contract Tests
# ---------------------------------------------------------------------------


def test_fit_request_contract_valid_and_roundtrip() -> None:
    req = FitRequest(
        schema_version=FIT_REQUEST_SCHEMA_VERSION,
        request_id="req-1",
        shot_id="shot-1",
        model_hash=SAMPLE_SHA256,
        candidate_count=3,
        objective_profile="dual_grip_profile_v1",
        time_window_start_pts=0,
        time_window_end_pts=50,
        budget_seconds=120.0,
        engine_capability_requirement=("rk45", "dual_grip_closure"),
    )
    assert req.candidate_count == 3
    d = req.to_dict()
    assert d["time_window_start_pts"] == 0
    restored = FitRequest.from_dict(d)
    assert restored == req


def test_result_bundle_contract_valid_and_roundtrip() -> None:
    req = FitRequest(
        schema_version=FIT_REQUEST_SCHEMA_VERSION,
        request_id="req-1",
        shot_id="shot-1",
        model_hash=SAMPLE_SHA256,
        candidate_count=1,
        objective_profile="dual_grip_profile_v1",
        time_window_start_pts=0,
        time_window_end_pts=50,
        budget_seconds=120.0,
        engine_capability_requirement=("rk45",),
    )
    bundle = ResultBundle(
        schema_version=RESULT_BUNDLE_SCHEMA_VERSION,
        bundle_id="bundle-1",
        request=req,
        candidates=(),
        replay_audits=(),
        execution_status="completed",
        evidence_quality="kinematic_only",
        metrics={"score": 0.95},
        hashes={"model": SAMPLE_SHA256},
    )
    assert bundle.execution_status == "completed"
    assert bundle.evidence_quality == "kinematic_only"
    d = bundle.to_dict()
    restored = ResultBundle.from_dict(d)
    assert restored == bundle


# ---------------------------------------------------------------------------
# Service Protocols Tests
# ---------------------------------------------------------------------------


def test_service_protocols_and_data_types() -> None:
    # Test duck-typing and structural conformance of protocols

    class MockSegmenter:
        def segment(self, request: SegmentationRequest) -> SegmentationResult:
            return SegmentationResult(
                shot_id=request.shot_id,
                mask_count=len(request.frame_ids),
                provenance="mock",
            )

    class MockRenderer:
        def render(self, request: RenderRequest) -> RenderResult:
            return RenderResult(body_mask=(1,), club_mask=(0,), visibility_mask=(1,))

    class MockForwardModel:
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(
                supported_bodies=("torso",),
                state_convention="canonical-v2",
                actuator_modes=("torque",),
                contact_modes=("none",),
                is_available=True,
            )

        def rollout(self, request: RolloutRequest) -> RolloutResult:
            audit = ReplayAudit(
                schema_version=REPLAY_AUDIT_SCHEMA_VERSION,
                candidate_id="c1",
                reset_count=1,
                integrator_name="test",
                integrator_version="1",
                coverage_start_s=0.0,
                coverage_end_s=1.0,
                max_grip_translation_error_m=0.01,
                max_grip_rotation_error_rad=0.01,
                is_physically_accepted=True,
            )
            return RolloutResult(
                trajectory=((0.0,),),
                realized_controls=((0.0,),),
                time_points_s=(0.0,),
                audit=audit,
            )

    class MockService:
        def fit(self, request: FitRequest) -> ResultBundle:
            return ResultBundle(
                schema_version=RESULT_BUNDLE_SCHEMA_VERSION,
                bundle_id="b1",
                request=request,
                candidates=(),
                replay_audits=(),
                execution_status="completed",
                evidence_quality="kinematic_only",
                metrics={},
                hashes={},
            )

    assert isinstance(MockSegmenter(), Segmenter)
    assert isinstance(MockRenderer(), SilhouetteRenderer)
    assert isinstance(MockForwardModel(), ForwardModel)
    assert isinstance(MockService(), ShadowTrackerService)


# ---------------------------------------------------------------------------
# Boundary & Malformed Deserialization Tests
# ---------------------------------------------------------------------------


def test_deserialization_rejects_non_dict_payloads() -> None:
    for cls in (
        Shot,
        FrameObservation,
        CameraTrack,
        SubjectModelBinding,
        FitRequest,
        ReplayAudit,
        CandidateResult,
        ResultBundle,
    ):
        with pytest.raises(TypeError, match="Payload must be a dict"):
            cls.from_dict(["not", "a", "dict"])  # type: ignore[arg-type]


def test_deserialization_rejects_unknown_and_missing_fields() -> None:
    valid_shot = Shot(
        schema_version=SHOT_SCHEMA_VERSION,
        asset_id="asset-1",
        shot_id="shot-1",
        start_pts=0,
        end_pts=100,
        start_frame_id="f0",
        end_frame_id="f100",
        subject_id="sub-1",
        swing_id="swing-1",
        camera_id="cam-1",
        cuts=(),
        transforms=(),
    )
    d = valid_shot.to_dict()

    # Unknown field
    d_extra = dict(d, extra_field=123)
    with pytest.raises(ValueError, match="Unknown fields rejected"):
        Shot.from_dict(d_extra)

    # Missing field
    d_missing = dict(d)
    del d_missing["camera_id"]
    with pytest.raises(ValueError, match="Missing required fields"):
        Shot.from_dict(d_missing)


def test_immutable_ownership() -> None:
    mutable_envelope = {"height_m": 1.80}
    binding = SubjectModelBinding(
        schema_version=SUBJECT_BINDING_SCHEMA_VERSION,
        subject_id="sub-1",
        model_hash=SAMPLE_SHA256,
        joint_ids=("pelvis",),
        body_ids=("pelvis_body",),
        visual_envelope=mutable_envelope,
        mass_kg=80.0,
        scale_evidence="wand",
        handedness="right",
    )
    # Mutate the input dictionary
    mutable_envelope["height_m"] = 999.0
    assert binding.visual_envelope["height_m"] == 1.80

    # Mutate the dictionary returned by to_dict
    exported = binding.to_dict()
    exported["visual_envelope"]["height_m"] = 999.0
    assert binding.visual_envelope["height_m"] == 1.80


def test_shot_with_negative_start_pts() -> None:
    shot = Shot(
        schema_version=SHOT_SCHEMA_VERSION,
        asset_id="asset-1",
        shot_id="shot-1",
        start_pts=-30,
        end_pts=20,
        start_frame_id="f_neg30",
        end_frame_id="f_pos20",
        subject_id="sub-1",
        swing_id="swing-1",
        camera_id="cam-1",
        cuts=((-10, -5),),
        transforms=(),
    )
    assert shot.duration_pts == 50
    d = shot.to_dict()
    restored = Shot.from_dict(d)
    assert restored == shot


def test_lazy_package_exports() -> None:
    import shared.python.shadow_tracker as st

    assert st.Shot is Shot
    assert st.FrameObservation is FrameObservation
    assert st.CameraTrack is CameraTrack
    assert st.SubjectModelBinding is SubjectModelBinding
    assert st.FitRequest is FitRequest
    assert st.ReplayAudit is ReplayAudit
    assert st.CandidateResult is CandidateResult
    assert st.ResultBundle is ResultBundle
    assert st.Segmenter is Segmenter
    assert st.SilhouetteRenderer is SilhouetteRenderer
    assert st.ForwardModel is ForwardModel
    assert st.ShadowTrackerService is ShadowTrackerService
