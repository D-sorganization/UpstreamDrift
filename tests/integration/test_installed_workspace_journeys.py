"""Integration acceptance tests for all enabled recommended task journeys across shipped surfaces (#10532).

Epic: #10508
Issue: #10532 ([ORG-23] Validate and Accept Every Enabled Recommended Task Journey Across Shipped Surfaces)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.integration

from src.shared.python.workspace.artifact_handoff import (
    ArtifactKind,
    ArtifactReference,
    compute_file_sha256,
)
from src.shared.python.workspace.capture_inspection_handoff import (
    CaptureInspectionHandoff,
    CropSpec,
    TrimSpec,
)
from src.shared.python.workspace.comparison_indicator_workspace import (
    BiomechanicalLoadChannels,
    ComparisonIndicatorWorkspaceCoordinator,
    ComparisonRunArtifact,
    CrossEngineComparisonAdapter,
    ModelFidelityLevel,
)
from src.shared.python.workspace.estimation_workspace import (
    EstimationObservationPayload,
    EstimationParameterPrior,
    EstimationRequest,
    EstimationWorkspaceCoordinator,
)
from src.shared.python.workspace.global_utilities import (
    GlobalWorkspaceUtilitiesCoordinator,
)
from src.shared.python.workspace.installed_journeys import (
    DependencyUnavailableError,
    InstalledWorkspaceJourneysCoordinator,
    InvalidArtifactSchemaError,
    JourneyExecutionResult,
    JourneyPhase,
    RecoveryStatus,
    UtilityNavigationTarget,
)
from src.shared.python.workspace.model_match_handoff import (
    ClubSpec,
    FitJobRequest,
    FitStage,
    InitialPoseSpec,
    MatchingRoute,
    ModelMatchHandoffCoordinator,
    ModelSpec,
    SubjectSpec,
)
from src.shared.python.workspace.optimization_training_workspace import (
    OptimizationJobConfig,
    OptimizationObjective,
    OptimizationTrainingWorkspaceCoordinator,
)
from src.shared.python.workspace.project_store import SessionProjectStore
from src.shared.python.workspace.shot_course_workspace import (
    BunkerFidelityTier,
    PuttingFixture,
    ShotCourseMode,
    ShotCourseWorkspaceCoordinator,
    SimulatorDeliveryRequest,
    TerrainConfig,
)


def _sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _make_file(path: Path, content: bytes) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path, _sha256_bytes(content)


def _setup_test_project_and_session(root: Path):
    store = SessionProjectStore(root)
    store.create_project("proj-e2e", "E2E Recommended Task Journeys")
    store.add_subject("subj-test", "Benchmark Golfer")
    session = store.create_session("sess-bench-01", "subj-test", "Benchmark Session")
    return store, session


# ============================================================================
# RED Cases: Fault Handling, Schema Validation, Cancellation & Recovery
# ============================================================================


def test_red_missing_dependency_fails_actionably_without_state_corruption(
    tmp_path: Path,
) -> None:
    """RED: Invoking a journey requiring an uninstalled external provider fails actionably without state corruption."""
    store, session = _setup_test_project_and_session(tmp_path / "proj")
    journeys = InstalledWorkspaceJourneysCoordinator(store=store)

    with pytest.raises(DependencyUnavailableError) as exc_info:
        journeys.execute_optical_import_to_inspection(
            project_id="proj-e2e",
            session_id=session.session_id,
            source_path=tmp_path / "nonexistent.mp4",
            provider="freemocap_missing_provider_bin",
        )

    assert "freemocap_missing_provider_bin" in str(exc_info.value)
    # Assert no corrupted artifacts or false-success entries were written to the session
    assert len(journeys.get_session_artifacts("sess-bench-01")) == 0


def test_red_corrupted_artifact_schema_rejected_without_output_pollution(
    tmp_path: Path,
) -> None:
    """RED: Input artifacts violating schema contracts are rejected fail-closed before processing."""
    store, session = _setup_test_project_and_session(tmp_path / "proj")
    journeys = InstalledWorkspaceJourneysCoordinator(store=store)

    bad_file, bad_sha = _make_file(
        tmp_path / "corrupt_data.json", b"{invalid-json: corrupted content"
    )

    with pytest.raises(InvalidArtifactSchemaError) as exc_info:
        journeys.validate_and_register_artifact(
            project_id="proj-e2e",
            session_id=session.session_id,
            artifact_kind=ArtifactKind.OBSERVATION,
            file_path=bad_file,
            schema_version="999.0.0",
        )

    assert (
        "corrupt" in str(exc_info.value).lower()
        or "schema" in str(exc_info.value).lower()
    )
    # Ensure nothing was committed to the store
    assert len(journeys.get_session_artifacts("sess-bench-01")) == 0


def test_red_cancellation_and_restart_recovery_preserves_source_media(
    tmp_path: Path,
) -> None:
    """RED: Cancelling an active journey preserves raw source artifacts intact and allows restart."""
    store, session = _setup_test_project_and_session(tmp_path / "proj")
    journeys = InstalledWorkspaceJourneysCoordinator(store=store)

    raw_video, raw_sha = _make_file(
        tmp_path / "source_take.mp4", b"original-unmodified-media-stream"
    )

    result = journeys.execute_optical_import_with_cancellation(
        project_id="proj-e2e",
        session_id=session.session_id,
        video_path=raw_video,
        cancel_requested=True,
    )

    assert result.status == RecoveryStatus.CANCELLED
    assert result.phase == JourneyPhase.ABORTED

    # Source media must remain completely untouched with identical hash
    assert raw_video.exists()
    assert compute_file_sha256(raw_video) == raw_sha

    # Verify session store does not claim completed import
    assert not any(
        a.kind == ArtifactKind.OBSERVATION
        for a in journeys.get_session_artifacts("sess-bench-01")
    )

    # Restart recovery: run without cancellation completes successfully
    recovered = journeys.execute_optical_import_with_cancellation(
        project_id="proj-e2e",
        session_id=session.session_id,
        video_path=raw_video,
        cancel_requested=False,
    )
    assert recovered.status == RecoveryStatus.SUCCESS


# ============================================================================
# GREEN Cases: The 7 Enabled Recommended Task Journeys
# ============================================================================


def test_journey_1_optical_import_to_inspection_and_save(tmp_path: Path) -> None:
    """Journey 1: video/optical import -> inspection -> save/register into project store."""
    store, session = _setup_test_project_and_session(tmp_path / "proj")
    journeys = InstalledWorkspaceJourneysCoordinator(store=store)

    video_path, video_sha = _make_file(
        tmp_path / "proj" / "views" / "cam_down_the_line.mp4",
        b"mp4-video-raw-payload-dtl",
    )

    res = journeys.execute_optical_import_to_inspection(
        project_id="proj-e2e",
        session_id=session.session_id,
        source_path=video_path,
        view_name="down_the_line",
        trim=TrimSpec(
            start_frame=10, end_frame=100, start_time_s=0.166, end_time_s=1.666
        ),
        crop=CropSpec(x=0, y=0, width=1920, height=1080),
        fps=60.0,
    )

    assert res.status == RecoveryStatus.SUCCESS
    assert res.artifact is not None
    assert res.artifact.kind == ArtifactKind.OBSERVATION
    assert res.artifact.metadata["trim"]["start_frame"] == 10

    # Verify session in store was updated with the artifact
    artifacts = journeys.get_session_artifacts("sess-bench-01")
    assert any(a.artifact_id == res.artifact.artifact_id for a in artifacts)


def test_journey_2_model_pose_to_supported_fit_and_export(tmp_path: Path) -> None:
    """Journey 2: model/pose -> supported fit -> replay/export."""
    store, session = _setup_test_project_and_session(tmp_path / "proj")
    journeys = InstalledWorkspaceJourneysCoordinator(store=store)

    obs_file, obs_sha = _make_file(
        tmp_path / "proj" / "obs_2d.json",
        json.dumps(
            {"frames": 60, "keypoints": ["head", "left_wrist", "right_wrist"]}
        ).encode(),
    )
    obs_ref = journeys.validate_and_register_artifact(
        project_id="proj-e2e",
        session_id=session.session_id,
        artifact_kind=ArtifactKind.OBSERVATION,
        file_path=obs_file,
    )

    sub = SubjectSpec(subject_id="subj-test", stature_m=1.80, mass_kg=75.0)
    club = ClubSpec(
        club_id="club-driver", name="driver", head_mass_kg=0.2, shaft_length_m=1.15
    )
    model_file, _ = _make_file(tmp_path / "model.urdf", b"<robot name='test'/>")
    model = ModelSpec(
        model_id="model-test",
        model_type="scapula",
        source_path=model_file,
        subject=sub,
        club=club,
    )
    initial_pose = InitialPoseSpec(
        pose_id="pose-address", frame="ground", angles_deg={}
    )

    fit_result = journeys.execute_model_pose_to_fit_and_export(
        project_id="proj-e2e",
        session_id=session.session_id,
        observation_ref=obs_ref,
        subject=sub,
        model=model,
        club=club,
        initial_pose=initial_pose,
        backend="mujoco",
    )

    assert fit_result.status == RecoveryStatus.SUCCESS
    assert fit_result.fit_stage == FitStage.KINEMATICS
    assert fit_result.trajectory_artifact is not None
    assert fit_result.trajectory_artifact.kind == ArtifactKind.TRAJECTORY

    # Verify trajectory artifact persisted in store
    artifacts = journeys.get_session_artifacts("sess-bench-01")
    assert any(
        a.artifact_id == fit_result.trajectory_artifact.artifact_id for a in artifacts
    )


def test_journey_3_shot_to_named_flight_comparison_and_reopen(tmp_path: Path) -> None:
    """Journey 3: shot -> named flight comparison -> reopen without path re-entry."""
    store, session = _setup_test_project_and_session(tmp_path / "proj")
    journeys = InstalledWorkspaceJourneysCoordinator(store=store)

    comp_result = journeys.execute_shot_to_flight_comparison(
        project_id="proj-e2e",
        session_id=session.session_id,
        launch_speed_mps=72.0,
        launch_angle_deg=11.5,
        launch_azimuth_deg=0.5,
        backspin_rpm=2400.0,
        reference_flight_name="tour_baseline_driver_2026",
    )

    assert comp_result.status == RecoveryStatus.SUCCESS
    assert comp_result.comparison_artifact is not None
    assert "metrics" in comp_result.comparison_artifact.metadata
    assert (
        comp_result.comparison_artifact.metadata["reference_flight"]
        == "tour_baseline_driver_2026"
    )

    # Reopen from store without path re-entry
    reopened = journeys.reopen_flight_comparison(
        session_id=session.session_id,
        comparison_artifact_id=comp_result.comparison_artifact.artifact_id,
    )
    assert reopened["reference_flight"] == "tour_baseline_driver_2026"
    assert "carry_distance_m" in reopened["metrics"]


def test_journey_4_optimization_training_to_result(tmp_path: Path) -> None:
    """Journey 4: optimization/training -> result."""
    store, session = _setup_test_project_and_session(tmp_path / "proj")
    journeys = InstalledWorkspaceJourneysCoordinator(store=store)

    config = OptimizationJobConfig(
        job_name="swing_energy_minimization",
        session_id=session.session_id,
        backend="scipy_lumped",
        objectives=[
            OptimizationObjective(name="minimize_energy", target=0.0, weight=1.0)
        ],
        golfer_model_id="default_human_anthropometrics",
        club_model_id="driver_standard",
    )

    opt_result = journeys.execute_optimization_training_to_result(config)

    assert opt_result.status == RecoveryStatus.SUCCESS
    assert opt_result.converged is True
    assert opt_result.result_artifact is not None
    assert opt_result.result_artifact.kind == ArtifactKind.RECEIPT
    assert opt_result.result_artifact.metadata["objective_value"] <= 1.0


def test_journey_5_estimation_to_comparison_and_indicators(tmp_path: Path) -> None:
    """Journey 5: bounded estimation -> cross-engine comparison & injury indicators."""
    store, session = _setup_test_project_and_session(tmp_path / "proj")
    journeys = InstalledWorkspaceJourneysCoordinator(store=store)

    times = np.linspace(0.0, 1.0, 5)
    true_scale = 1.08
    q_truth = times**2
    observations = (true_scale * q_truth)[:, None]

    obs = EstimationObservationPayload(
        trial_id="synthetic-trial-001",
        times=times,
        observations=observations,
        units="m",
    )
    prior = EstimationParameterPrior(
        name="upper_length_m",
        initial=1.0,
        kind="length",
        lower=0.5,
        upper=1.5,
    )
    est_request = EstimationRequest(
        task_id="est-req-001",
        task_kind="map_single_trial",
        observation=obs,
        parameter_specs=(prior,),
        max_iterations=80,
    )

    est_comp_result = journeys.execute_estimation_to_cross_engine_comparison(
        project_id="proj-e2e",
        session_id=session.session_id,
        estimation_request=est_request,
        baseline_engine="pinocchio",
        candidate_engine="mujoco",
    )

    assert est_comp_result.status == RecoveryStatus.SUCCESS
    assert est_comp_result.estimation_solved is True
    assert est_comp_result.indicator_result is not None
    assert est_comp_result.cross_engine_agreement is not None
    assert est_comp_result.cross_engine_agreement["pinocchio_vs_mujoco"] is True


def test_journey_6_global_utilities_navigation(tmp_path: Path) -> None:
    """Journey 6: global help/assistant navigation and sidekick dispatch."""
    store, session = _setup_test_project_and_session(tmp_path / "proj")
    journeys = InstalledWorkspaceJourneysCoordinator(store=store)

    nav_result = journeys.execute_global_utilities_navigation(
        action="open_sidekick_assistant",
        session_id=session.session_id,
        context_payload={"user_query": "Explain kinematics reconstruction step"},
    )

    assert nav_result.status == RecoveryStatus.SUCCESS
    assert nav_result.target == UtilityNavigationTarget.SIDEKICK
    assert nav_result.assistant_response is not None
    assert "kinematics" in nav_result.assistant_response.lower()
