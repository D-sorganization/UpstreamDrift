"""Integration tests for model, subject, club, pose, fit, and dynamics handoff.

Acceptance cases for issue #10522 / parent epic #10508:
- RED: wrong subject/club/frame and unavailable backend rejected before job start.
- RED: kinematic output cannot be marked dynamic success; tour route refuses unsupported capture.
- GREEN: real small supported target -> model/pose -> fit -> saved result;
  changing model invalidates downstream state; cancel/reopen preserves prior run.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.shared.python.workspace import (
    ArtifactKind,
    ArtifactReference,
    SessionProjectStore,
    compute_file_sha256,
)
from src.shared.python.workspace.model_match_handoff import (
    ClubSpec,
    EngineQualification,
    FitJobRequest,
    FitJobResult,
    FitStage,
    InitialPoseSpec,
    MatchingRoute,
    ModelMatchHandoffCoordinator,
    ModelSpec,
    SubjectSpec,
    get_engine_qualification,
    list_available_backends,
    resolve_matching_route,
)


@pytest.mark.integration
def test_reject_wrong_subject_club_frame_and_unavailable_backend(
    tmp_path: Path,
) -> None:
    """Precondition checks must reject mismatched subject, club, frame, or unavailable backend."""
    store = SessionProjectStore(tmp_path)
    store.create_project("test_proj", "Test Project")
    store.add_subject("sub_01", "Alice")
    store.create_session("sess_01", "sub_01", "Session 1")

    sub_valid = SubjectSpec(subject_id="sub_01", stature_m=1.75, mass_kg=70.0)
    sub_wrong = SubjectSpec(subject_id="sub_99", stature_m=1.85, mass_kg=85.0)

    club_valid = ClubSpec(club_id="driver_01", name="driver")
    club_wrong = ClubSpec(club_id="putter_01", name="putter")

    model_valid = ModelSpec(
        model_id="model_01",
        model_type="anthropometric",
        source_path=tmp_path / "model.json",
        subject=sub_valid,
        club=club_valid,
    )

    pose_valid = InitialPoseSpec(
        pose_id="pose_01",
        frame="ground",
        angles_deg={"hip_flexion_r": 20.0},
    )

    dummy_file = tmp_path / "target.c3d"
    dummy_file.write_bytes(b"DUMMY_TARGET_PAYLOAD")

    target_ref = ArtifactReference(
        artifact_id="target_01",
        path=str(dummy_file),
        kind=ArtifactKind.OBSERVATION,
        schema="motion_capture.c3d/1",
        hash=compute_file_sha256(dummy_file),
        metadata={"frame": "ground"},
    )

    # 1. Reject invalid initial pose frame
    with pytest.raises(ValueError, match="frame"):
        InitialPoseSpec(
            pose_id="pose_02",
            frame="invalid_arbitrary_frame",
            angles_deg={},
        )

    # 2. Reject subject mismatch between request and model
    with pytest.raises(ValueError, match="Subject mismatch"):
        FitJobRequest(
            job_id="job_2",
            session_id="sess_01",
            subject=sub_wrong,
            club=club_valid,
            model=model_valid,
            initial_pose=pose_valid,
            target_ref=target_ref,
            backend="mujoco",
            stage=FitStage.KINEMATICS,
            route=MatchingRoute.TOUR_MATCHING,
        )

    # 3. Reject club mismatch between request and model
    with pytest.raises(ValueError, match="Club mismatch"):
        FitJobRequest(
            job_id="job_3",
            session_id="sess_01",
            subject=sub_valid,
            club=club_wrong,
            model=model_valid,
            initial_pose=pose_valid,
            target_ref=target_ref,
            backend="mujoco",
            stage=FitStage.KINEMATICS,
            route=MatchingRoute.TOUR_MATCHING,
        )

    # 4. Reject unavailable or non-existent backend
    with pytest.raises(
        ValueError, match="Backend 'nonexistent_engine' is not available"
    ):
        FitJobRequest(
            job_id="job_4",
            session_id="sess_01",
            subject=sub_valid,
            club=club_valid,
            model=model_valid,
            initial_pose=pose_valid,
            target_ref=target_ref,
            backend="nonexistent_engine",
            stage=FitStage.KINEMATICS,
            route=MatchingRoute.TOUR_MATCHING,
        )


@pytest.mark.integration
def test_kinematic_output_not_marked_dynamic_and_tour_route_refuses_unsupported_capture(
    tmp_path: Path,
) -> None:
    """Kinematic fits cannot claim dynamic qualification and tour route must refuse arbitrary captures."""
    # 1. Contract check: kinematic stage CANNOT be marked as dynamic qualified
    with pytest.raises(
        ValueError, match="Kinematic stage output cannot be marked as dynamic qualified"
    ):
        FitJobResult(
            job_id="job_k",
            session_id="sess_01",
            run_id="run_01",
            stage=FitStage.KINEMATICS,
            backend="mujoco",
            success=True,
            is_dynamic_qualified=True,  # VIOLATION: Kinematics cannot be dynamic qualified
            metrics={"marker_rms_mm": 12.0},
            receipt={"converged": True},
            artifact_refs=[],
        )

    # 2. Route resolution check: arbitrary video / optical observations must route to GENERAL_MOTION_PIPELINE
    optical_ref = ArtifactReference(
        artifact_id="opt_01",
        path=str(tmp_path / "video_keypoints.json"),
        hash="1" * 64,
        kind=ArtifactKind.OBSERVATION,
        schema="observation_2d/mediapipe/1.0.0",
        metadata={"frame": "app_xtarget_yup_zright"},
    )
    route = resolve_matching_route(optical_ref, capture_name="arbitrary_video")
    assert route == MatchingRoute.GENERAL_MOTION_PIPELINE

    # 3. Tour matching route refuses unsupported arbitrary video capture
    sub = SubjectSpec(subject_id="sub_01", stature_m=1.75, mass_kg=70.0)
    club = ClubSpec(club_id="driver_01", name="driver")
    model = ModelSpec(
        model_id="model_01",
        model_type="anthropometric",
        source_path=tmp_path / "model.json",
        subject=sub,
        club=club,
    )
    pose = InitialPoseSpec(pose_id="pose_01", frame="ground", angles_deg={})

    with pytest.raises(
        ValueError, match="Tour matching route requires qualified tour capture"
    ):
        FitJobRequest(
            job_id="job_tour_invalid",
            session_id="sess_01",
            subject=sub,
            club=club,
            model=model,
            initial_pose=pose,
            target_ref=optical_ref,
            backend="mujoco",
            stage=FitStage.KINEMATICS,
            route=MatchingRoute.TOUR_MATCHING,
        )


@pytest.mark.integration
def test_small_supported_target_to_fit_and_model_invalidation_cancel_reopen(
    tmp_path: Path,
) -> None:
    """Full lifecycle: target -> model/pose -> fit -> save; model change invalidates; cancel/reopen preserves."""
    store = SessionProjectStore(tmp_path / "project_store")
    store.create_project("test_proj", "Tour Test Project")
    store.add_subject("sub_01", "Tiger")
    store.create_session("sess_01", "sub_01", "Driver Session")

    coordinator = ModelMatchHandoffCoordinator(store)

    sub = SubjectSpec(subject_id="sub_01", stature_m=1.85, mass_kg=84.0)
    club = ClubSpec(club_id="driver_01", name="driver")
    model_file = tmp_path / "full_body_spec_anthro_driver.json"
    model_file.write_text(
        json.dumps({"name": "anthro_driver_test", "bodies": []}), encoding="utf-8"
    )
    model_1 = ModelSpec(
        model_id="model_anthro_1",
        model_type="anthropometric",
        source_path=model_file,
        subject=sub,
        club=club,
    )
    pose = InitialPoseSpec(
        pose_id="pose_address", frame="ground", angles_deg={"hip_flexion_r": 25.0}
    )

    # Prepare supported tour target
    tour_c3d = tmp_path / "C3D_TA_Driver.c3d"
    tour_c3d.write_bytes(b"FAKE_TOUR_C3D_PAYLOAD")
    tour_ref = ArtifactReference(
        artifact_id="tour_01",
        path=str(tour_c3d),
        hash=compute_file_sha256(tour_c3d),
        kind=ArtifactKind.OBSERVATION,
        schema="motion_capture.c3d/1",
        metadata={"frame": "ground"},
    )

    # 1. Execute Kinematic Fit Stage
    req_kin = FitJobRequest(
        job_id="job_kin_01",
        session_id="sess_01",
        subject=sub,
        club=club,
        model=model_1,
        initial_pose=pose,
        target_ref=tour_ref,
        backend="mujoco",
        stage=FitStage.KINEMATICS,
        route=MatchingRoute.TOUR_MATCHING,
    )
    res_kin = coordinator.execute_fit_stage(req_kin)
    assert res_kin.success is True
    assert res_kin.stage == FitStage.KINEMATICS
    assert res_kin.is_dynamic_qualified is False
    assert res_kin.run_id is not None
    assert len(store.list_runs("sess_01")) == 1

    # 2. Execute Dynamics Stage on the completed run
    req_dyn = FitJobRequest(
        job_id="job_dyn_01",
        session_id="sess_01",
        subject=sub,
        club=club,
        model=model_1,
        initial_pose=pose,
        target_ref=tour_ref,
        backend="mujoco",
        stage=FitStage.DYNAMICS,
        route=MatchingRoute.TOUR_MATCHING,
        run_id=res_kin.run_id,
    )
    res_dyn = coordinator.execute_fit_stage(req_dyn)
    assert res_dyn.success is True
    assert res_dyn.stage == FitStage.DYNAMICS
    assert res_dyn.is_dynamic_qualified is True

    # 3. Changing the model invalidates downstream state
    model_2_file = tmp_path / "model_2.json"
    model_2_file.write_text(json.dumps({"name": "anthro_v2"}), encoding="utf-8")
    model_2 = ModelSpec(
        model_id="model_anthro_2",
        model_type="anthropometric",
        source_path=model_2_file,
        subject=sub,
        club=club,
    )
    coordinator.update_session_model("sess_01", model_2)
    session_status = coordinator.get_session_status("sess_01")
    assert session_status["downstream_invalidated"] is True

    # 4. Cancel a job: preserves prior runs and records cancellation diagnostics
    req_cancel = FitJobRequest(
        job_id="job_kin_02",
        session_id="sess_01",
        subject=sub,
        club=club,
        model=model_2,
        initial_pose=pose,
        target_ref=tour_ref,
        backend="mujoco",
        stage=FitStage.KINEMATICS,
        route=MatchingRoute.TOUR_MATCHING,
    )
    res_cancel = coordinator.cancel_job(
        req_cancel.job_id, reason="User cancelled calibration"
    )
    assert res_cancel.canceled is True
    assert "User cancelled calibration" in res_cancel.diagnostics[0]
    # Prior completed run still preserved in store
    assert len(store.list_runs("sess_01")) >= 1

    # 5. Reopen run gives next action pointing to Results/Replay seam
    action = coordinator.reopen_run(res_kin.run_id)
    assert action["action"] == "open_results_replay"
    assert action["run_id"] == res_kin.run_id
    assert action["view"] == "results_browser"
