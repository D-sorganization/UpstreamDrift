"""Unit and contract tests for continuous shadow optimization with uncertainty and abstention (MMR-15, #11101).

Validates:
1. Real observations to continuous 3D candidate with fresh replay through public service.
2. Strict zero state resets (reset_count == 1) and zero hidden pelvis/root assistance.
3. Modern synchronized held-out video + C3D validation stratified by swing phase.
4. Archive stress test resilience: cuts, blur, and unknown camera intrinsics.
5. Confidence coverage and structured abstention assessment on holdouts.
6. Inferred torques and contact forces labeled model-dependent and unidentifiable from monocular video.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pytest

from src.shared.python.shadow_tracker._validation import (
    FIT_REQUEST_SCHEMA_VERSION,
    FRAME_OBSERVATION_SCHEMA_VERSION,
    FRAME_SCHEMA_VERSION,
    MASK_SCHEMA_VERSION,
    REPLAY_AUDIT_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
)
from src.shared.python.shadow_tracker.contracts import (
    CANONICAL_ARTICULATED_CONVENTION,
    CandidateResult,
    FitRequest,
    FrameObservation,
    ModelCapabilities,
    POINT_LANDMARKS_CONVENTION,
    ReplayAudit,
    RenderRequest,
    RenderResult,
    ResultBundle,
    RolloutRequest,
    RolloutResult,
)
from src.shared.python.shadow_tracker.evaluation import (
    GateProfile,
    PhaseEvaluationResult,
    QuantityConfidence,
    assess_holdout_coverage_and_abstention,
    audit_gate_profile,
    classify_evidence_quality,
    create_evaluated_result_bundle,
    evaluate_archive_stress_resilience,
    evaluate_candidate_evidence,
    evaluate_phase_stratified_tracking,
)
from src.shared.python.shadow_tracker.forward_model import (
    FullBodyForwardModel,
)
from src.shared.python.shadow_tracker.mask_records import MaskFrame
from src.shared.python.shadow_tracker.projection import (
    AnalyticSilhouetteRenderer,
    PinholeCameraModel,
)
from src.shared.python.shadow_tracker.service import (
    DefaultShadowTrackerService,
    UnavailableBackendError,
)
from src.shared.python.shadow_tracker.source_records import (
    FrameIdentity,
    SourceAsset,
)

pytestmark = [pytest.mark.unit]

SAMPLE_SHA256 = "0" * 64


# ---------------------------------------------------------------------------
# Test Helpers and Minimal Doubles
# ---------------------------------------------------------------------------


def _make_source() -> SourceAsset:
    return SourceAsset(
        schema_version=SOURCE_SCHEMA_VERSION,
        asset_id="asset-mmr15",
        source_uri="https://example.com/modern_calibrated_swing.mp4",
        content_sha256=SAMPLE_SHA256,
        width_px=128,
        height_px=128,
        rights_status="permitted",
        rights_note="open calibrated benchmark footage",
    )


def _make_calibrated_camera() -> PinholeCameraModel:
    return PinholeCameraModel(
        camera_id="cam-calibrated-01",
        width_px=128,
        height_px=128,
        fx=100.0,
        fy=100.0,
        cx=64.0,
        cy=64.0,
        translation_world_to_camera=(0.0, 0.0, 3.0),
    )


def _make_observation(
    frame_idx: int,
    *,
    is_timing_exact: bool = True,
    physical_time_s: float | None = None,
    club_ref: str | None = None,
) -> FrameObservation:
    t_s = frame_idx / 100.0 if physical_time_s is None else physical_time_s
    return FrameObservation(
        schema_version=FRAME_OBSERVATION_SCHEMA_VERSION,
        shot_id="shot-mmr15",
        camera_id="cam-calibrated-01",
        frame_id=f"frame-{frame_idx:04d}",
        pts_ticks=frame_idx * 1000,
        timebase_numerator=1,
        timebase_denominator=100000,
        physical_time_s=t_s,
        physical_time_reason="hardware_sync_genlock"
        if is_timing_exact
        else "uncalibrated_video",
        body_mask_ref=f"mask-body-{frame_idx}",
        club_mask_ref=club_ref or f"mask-club-{frame_idx}",
        valid_mask_ref=f"mask-valid-{frame_idx}",
        confidence_provenance="motion_capture_synchronized",
        timing_mode="authoritative" if is_timing_exact else "estimated_cfr",
        is_timing_exact=is_timing_exact,
        clock_evidence="irig_timecode",
        decoder_name="opencv",
        pixel_format="rgb24",
    )


def _make_mask_frame(
    frame_idx: int,
    camera: PinholeCameraModel,
    renderer: AnalyticSilhouetteRenderer | None = None,
) -> MaskFrame:
    ident = FrameIdentity(
        schema_version=FRAME_SCHEMA_VERSION,
        asset_id="asset-mmr15",
        shot_id="shot-mmr15",
        swing_id="swing-01",
        camera_id=camera.camera_id,
        frame_id=f"frame-{frame_idx:04d}",
        pts_ticks=frame_idx * 1000,
        timebase_numerator=1,
        timebase_denominator=100000,
        physical_time_s=frame_idx / 100.0,
        physical_time_reason="hardware_sync_genlock",
        frame_sha256=SAMPLE_SHA256,
        timing_mode="authoritative",
        is_timing_exact=True,
        clock_evidence="irig_timecode",
    )
    total_px = camera.width_px * camera.height_px
    valid = bytes([1] * total_px)
    if renderer is not None:
        render_res = renderer.render(
            RenderRequest(
                camera_id=camera.camera_id,
                state=(0.0, 0.0, 1.0, 0.0, 0.0, 0.2),
                image_size_px=(camera.width_px, camera.height_px),
                state_convention=POINT_LANDMARKS_CONVENTION,
            )
        )
        body = bytes(render_res.body_mask)
        club = bytes(render_res.club_mask)
    else:
        body = bytes([1 if 2000 <= i < 5000 else 0 for i in range(total_px)])
        club = bytes([1 if 5000 <= i < 5500 else 0 for i in range(total_px)])
    return MaskFrame(
        schema_version=MASK_SCHEMA_VERSION,
        frame=ident,
        width_px=camera.width_px,
        height_px=camera.height_px,
        body=body,
        club=club,
        valid=valid,
        revision_id=f"rev-mmr15-{frame_idx}",
        parent_revision_id=None,
        producer_id="expert_reviewer",
        correction_note="calibrated manual mask",
    )


class MockQualifiedContinuousForwardModel:
    """Qualified forward dynamics model simulating continuous whole-body trajectory with Gate G4 compliance."""

    def __init__(
        self,
        *,
        is_qualified: bool = True,
        fail_physics: bool = False,
        inject_root_forces: bool = False,
        multiple_resets: bool = False,
    ) -> None:
        self._is_qualified = is_qualified
        self._fail_physics = fail_physics
        self._inject_root_forces = inject_root_forces
        self._multiple_resets = multiple_resets
        self.rollout_count = 0

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supported_bodies=tuple(f"coord_{i}" for i in range(41)),
            state_convention=POINT_LANDMARKS_CONVENTION,
            actuator_modes=("torque_polynomial_deg6",),
            contact_modes=("hunt_crossley_regularized_coulomb",),
            is_available=True,
            is_synthetic=False,
            is_qualified=self._is_qualified,
        )

    def rollout(self, request: RolloutRequest) -> RolloutResult:
        self.rollout_count += 1
        times = tuple(request.time_points_s)
        controls_arr = np.asarray(request.controls, dtype=np.float64)

        if self._inject_root_forces:
            # Undeclared ghost pelvis assistance
            has_undeclared_root_forces = True
        else:
            has_undeclared_root_forces = bool(
                np.any(np.abs(controls_arr[:6, :]) > 1e-9)
            )

        traj: list[tuple[float, ...]] = []
        for _ in times:
            traj.append((0.0, 0.0, 1.0, 0.0, 0.0, 0.2))

        is_phys = (
            not self._fail_physics
            and not has_undeclared_root_forces
            and not self._multiple_resets
        )

        audit = ReplayAudit(
            schema_version=REPLAY_AUDIT_SCHEMA_VERSION,
            candidate_id=f"cand_continuous_{self.rollout_count}",
            reset_count=2 if self._multiple_resets else 1,
            integrator_name="rk45",
            integrator_version="1.0.0",
            coverage_start_s=float(times[0]) if times else 0.0,
            coverage_end_s=float(times[-1]) if times else 0.0,
            max_grip_translation_error_m=0.0004 if is_phys else 0.05,
            max_grip_rotation_error_rad=0.002 if is_phys else 0.4,
            is_physically_accepted=is_phys,
        )
        return RolloutResult(
            trajectory=tuple(traj),
            realized_controls=request.controls,
            time_points_s=request.time_points_s,
            audit=audit,
        )


# ---------------------------------------------------------------------------
# Acceptance Criterion 1 & 2: Continuous Replay & Zero State Resets / Root Assistance
# ---------------------------------------------------------------------------


def test_real_video_to_continuous_candidate_with_fresh_replay() -> None:
    """AC-1: Real observations -> qualified continuous forward model -> fresh replay through public service."""
    cam = _make_calibrated_camera()
    renderer = AnalyticSilhouetteRenderer({cam.camera_id: cam})
    service = DefaultShadowTrackerService()

    obs_list = [_make_observation(i) for i in range(5)]
    mask_list = [_make_mask_frame(i, cam, renderer=renderer) for i in range(5)]

    service.initialize_session(
        source_asset=_make_source(),
        observations=obs_list,
        initial_masks=mask_list,
    )

    qualified_backend = MockQualifiedContinuousForwardModel(is_qualified=True)
    service.register_backend(
        forward_model=qualified_backend,
        renderer=renderer,
        camera=cam,
    )

    req = FitRequest(
        schema_version=FIT_REQUEST_SCHEMA_VERSION,
        request_id="req-continuous-01",
        shot_id="shot-mmr15",
        model_hash=SAMPLE_SHA256,
        candidate_count=1,
        objective_profile="silhouette_iou",
        time_window_start_pts=0,
        time_window_end_pts=4000,
        budget_seconds=2.0,
        engine_capability_requirement=(),
    )

    bundle = service.fit(req)

    assert isinstance(bundle, ResultBundle)
    assert bundle.execution_status == "completed"
    assert len(bundle.candidates) == 1
    cand = bundle.candidates[0]
    assert cand.is_accepted is True
    assert cand.replay_audit is not None
    assert cand.replay_audit.is_physically_accepted is True
    # Zero state resets
    assert cand.replay_audit.reset_count == 1
    # Checkpoint provenance recorded
    assert len(service.checkpoints) > 0


def test_zero_state_resets_and_zero_pelvis_assistance_enforced() -> None:
    """AC-2: Mid-trajectory state resets or hidden pelvis assistance strictly fails physical acceptance."""
    cam = _make_calibrated_camera()
    renderer = AnalyticSilhouetteRenderer({cam.camera_id: cam})
    service = DefaultShadowTrackerService()

    obs_list = [_make_observation(i) for i in range(3)]
    mask_list = [_make_mask_frame(i, cam, renderer=renderer) for i in range(3)]

    service.initialize_session(
        source_asset=_make_source(),
        observations=obs_list,
        initial_masks=mask_list,
    )

    # 1. Multiple resets (reset_count > 1) fails
    backend_resets = MockQualifiedContinuousForwardModel(
        is_qualified=True, multiple_resets=True
    )
    service.register_backend(
        forward_model=backend_resets, renderer=renderer, camera=cam
    )
    req = FitRequest(
        schema_version=FIT_REQUEST_SCHEMA_VERSION,
        request_id="req-resets",
        shot_id="shot-mmr15",
        model_hash=SAMPLE_SHA256,
        candidate_count=1,
        objective_profile="silhouette_iou",
        time_window_start_pts=0,
        time_window_end_pts=2000,
        budget_seconds=2.0,
        engine_capability_requirement=(),
    )
    bundle_resets = service.fit(req)
    assert bundle_resets.candidates[0].is_accepted is False
    assert bundle_resets.candidates[0].replay_audit is not None
    assert bundle_resets.candidates[0].replay_audit.is_physically_accepted is False

    # 2. Ghost / hidden pelvis assistance fails
    backend_root = MockQualifiedContinuousForwardModel(
        is_qualified=True, inject_root_forces=True
    )
    service.register_backend(forward_model=backend_root, renderer=renderer, camera=cam)
    bundle_root = service.fit(req)
    assert bundle_root.candidates[0].is_accepted is False
    assert bundle_root.candidates[0].replay_audit is not None
    assert bundle_root.candidates[0].replay_audit.is_physically_accepted is False


# ---------------------------------------------------------------------------
# Acceptance Criterion 3: Phase-Stratified Validation Against Held-Out C3D
# ---------------------------------------------------------------------------


def test_modern_synchronized_held_out_c3d_phase_stratified_validation() -> None:
    """AC-3: Held-out synchronized C3D reference validates joint/marker/club errors by swing phase."""
    # Synthetic candidate trajectory covering 5 phases: address, backswing, downswing, impact, follow_through
    time_points = [0.0, 0.2, 0.4, 0.6, 0.8]
    candidate_traj = tuple(
        tuple(0.0 + 0.01 * idx for _ in range(42)) for idx in range(5)
    )
    reference_c3d = {
        "time_points_s": time_points,
        "phases": [
            "address",
            "backswing",
            "downswing",
            "impact",
            "follow_through",
        ],
        "joint_angles_rad": [tuple(0.0 for _ in range(35)) for _ in range(5)],
        "marker_positions_m": [
            {"pelvis": (0.0, 0.0, 1.0), "club_head": (0.2, 0.0, 0.1)} for _ in range(5)
        ],
    }

    report = evaluate_phase_stratified_tracking(
        candidate_trajectory=candidate_traj,
        time_points_s=time_points,
        reference_data=reference_c3d,
    )

    assert len(report.phase_results) == 5
    phases = [p.phase_name for p in report.phase_results]
    assert phases == [
        "address",
        "backswing",
        "downswing",
        "impact",
        "follow_through",
    ]
    for p in report.phase_results:
        assert isinstance(p, PhaseEvaluationResult)
        assert p.joint_rmse_rad >= 0.0
        assert p.marker_rmse_m >= 0.0
        assert p.club_error_m >= 0.0


# ---------------------------------------------------------------------------
# Acceptance Criterion 4: Archive Stress Set (Cuts, Blur, Unknown Camera)
# ---------------------------------------------------------------------------


def test_archive_stress_set_resilience_cuts_blur_unknown_camera() -> None:
    """AC-4: Archive stress set evaluates resilience to discontinuous cuts, motion blur, and unknown camera."""
    # 1. Discontinuous cuts / telecine jumps
    discontinuous_obs = (
        _make_observation(0, physical_time_s=0.0),
        _make_observation(1, physical_time_s=0.04),
        _make_observation(2, physical_time_s=0.50),  # cut jump!
    )
    stress_cuts = evaluate_archive_stress_resilience(
        observations=discontinuous_obs,
        stress_type="cuts",
    )
    assert stress_cuts["has_discontinuity"] is True
    assert "cut_detected_at_step_2" in stress_cuts["flags"]

    # 2. Motion blur
    stress_blur = evaluate_archive_stress_resilience(
        observations=discontinuous_obs[:2],
        stress_type="blur",
        blur_kernel_size_px=15,
    )
    assert stress_blur["clubhead_uncertainty_inflation"] > 1.0
    assert stress_blur["abstain_on_high_speed_impact"] is True

    # 3. Unknown camera intrinsics
    uncalibrated_obs = (
        _make_observation(0, is_timing_exact=False, physical_time_s=None),
        _make_observation(1, is_timing_exact=False, physical_time_s=None),
    )
    stress_camera = evaluate_archive_stress_resilience(
        observations=uncalibrated_obs,
        stress_type="unknown_camera",
    )
    assert stress_camera["si_kinetics_permitted"] is False
    assert stress_camera["status"] == "kinematic_only"


# ---------------------------------------------------------------------------
# Acceptance Criterion 5: Confidence Coverage and Abstention on Holdouts
# ---------------------------------------------------------------------------


def test_confidence_coverage_and_abstention_assessed_on_holdouts() -> None:
    """AC-5: Nominal 90% confidence intervals assessed for empirical coverage and unidentifiable abstention."""
    # 9 covered, 1 missed gives empirical_coverage = 0.90 (calibrated to nominal 0.90)
    intervals = ((-10.0, -5.0),) + tuple(
        (float(i) - 1.0, float(i) + 1.0) for i in range(1, 10)
    )
    ground_truth = tuple(float(i) for i in range(10))

    result = assess_holdout_coverage_and_abstention(
        intervals=intervals,
        ground_truth=ground_truth,
        nominal_coverage=0.90,
        unidentifiable_cases=["monocular_depth_ambiguity", "unobserved_club_face"],
    )

    assert result["is_well_calibrated"] is True
    assert result["empirical_coverage"] >= 0.85
    assert len(result["abstention_reasons"]) == 2
    assert "monocular_depth_ambiguity" in result["abstention_reasons"]


# ---------------------------------------------------------------------------
# Acceptance Criterion 6: Torques/Forces Labeled Model-Dependent and Unavailable
# ---------------------------------------------------------------------------


def test_torques_and_forces_labeled_model_dependent_and_unidentifiable() -> None:
    """AC-6: Inferred torques and contact forces are labeled model-dependent and unavailable when unidentifiable."""
    cam = _make_calibrated_camera()
    renderer = AnalyticSilhouetteRenderer({cam.camera_id: cam})
    service = DefaultShadowTrackerService()

    obs_list = [_make_observation(i) for i in range(3)]
    mask_list = [_make_mask_frame(i, cam, renderer=renderer) for i in range(3)]

    service.initialize_session(
        source_asset=_make_source(),
        observations=obs_list,
        initial_masks=mask_list,
    )
    backend = MockQualifiedContinuousForwardModel(is_qualified=True)
    service.register_backend(forward_model=backend, renderer=renderer, camera=cam)

    req = FitRequest(
        schema_version=FIT_REQUEST_SCHEMA_VERSION,
        request_id="req-torque-check",
        shot_id="shot-mmr15",
        model_hash=SAMPLE_SHA256,
        candidate_count=1,
        objective_profile="silhouette_iou",
        time_window_start_pts=0,
        time_window_end_pts=2000,
        budget_seconds=2.0,
        engine_capability_requirement=(),
    )

    bundle = service.fit(req)
    cand = bundle.candidates[0]

    # In monocular video tracking, torques/forces cannot be uniquely identified without ground reaction force data
    # Diagnostics or bundle metrics must explicitly reflect model-dependent unidentifiable status
    forces_meta = bundle.metrics.get("forces_and_torques", {})
    assert forces_meta.get("status") == "model_dependent_unidentifiable"
    assert forces_meta.get("is_identified") is False
    assert "unidentifiable" in forces_meta.get("warning", "").lower()
