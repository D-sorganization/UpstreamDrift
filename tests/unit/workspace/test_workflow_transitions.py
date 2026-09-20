"""Focused TDD acceptance tests for guided workflow transitions across workspaces.

Issue: #10518 ([ORG-09] Guided Workflow Transitions: Capture/Import -> Inspect Targets -> Model -> Fit -> Dynamics -> Compare -> Export)
Epic: #10508
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.shared.python.workspace import (
    ArtifactKind,
    ArtifactReference,
    SessionProjectStore,
    StepStatus,
    WorkflowCoordinator,
    WorkflowMode,
    WorkflowStepId,
)

pytestmark = pytest.mark.unit


def _sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _make_file(path: Path, content: bytes) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path, _sha256_bytes(content)


def _setup_store_and_project(root: Path):
    store = SessionProjectStore(root)
    store.create_project("proj-001", "Biomechanics Project")
    store.add_subject("subj-001", "Golfer 1")
    session = store.create_session("sess-001", "subj-001", "Session 1")
    return store, session


# ============================================================================
# RED Cases: Contract Enforcement & Stale Inputs
# ============================================================================


def test_reject_fabricated_completion_without_satisfying_outputs(
    tmp_path: Path,
) -> None:
    """RED: Step cannot advance to DONE without contract-satisfying outputs."""
    store, session = _setup_store_and_project(tmp_path / "proj")

    coordinator = WorkflowCoordinator(
        store=store,
        session_id=session.session_id,
        run_id="run-001",
        mode=WorkflowMode.FULL_BODY_3D,
    )

    # Attempting to advance CAPTURE_IMPORT with empty outputs fails
    with pytest.raises(ValueError, match="cannot advance step without valid outputs"):
        coordinator.advance_step(
            step_id=WorkflowStepId.CAPTURE_IMPORT,
            outputs=(),  # FABRICATED: empty outputs
        )

    projection = coordinator.get_projection()
    step_proj = projection.get_step(WorkflowStepId.CAPTURE_IMPORT)
    assert step_proj.status != StepStatus.DONE


def test_stale_input_hash_and_missing_output_prevent_advancement(
    tmp_path: Path,
) -> None:
    """RED: altered input hash on disk or deleted output file blocks advancement."""
    store, session = _setup_store_and_project(tmp_path / "proj")

    # Step 1 output
    cap_file, cap_hash = _make_file(
        tmp_path / "proj" / "capture.c3d", b"initial-c3d-bytes"
    )
    cap_ref = ArtifactReference(
        artifact_id="art-c3d",
        path="capture.c3d",
        hash=cap_hash,
        schema="motion_capture.c3d/1",
        kind=ArtifactKind.OBSERVATION,
    )

    coordinator = WorkflowCoordinator(
        store=store,
        session_id=session.session_id,
        run_id="run-001",
        mode=WorkflowMode.FULL_BODY_3D,
    )
    coordinator.advance_step(
        step_id=WorkflowStepId.CAPTURE_IMPORT,
        outputs=(cap_ref,),
    )

    # Now tamper with capture.c3d on disk (stale input hash)
    cap_file.write_bytes(b"altered-c3d-bytes")

    projection = coordinator.get_projection()
    inspect_step = projection.get_step(WorkflowStepId.INSPECT_TARGETS)

    # Inspect Targets must be BLOCKED because input hash is stale/altered
    assert inspect_step.status == StepStatus.BLOCKED
    assert (
        "hash mismatch" in inspect_step.reason.lower()
        or "stale" in inspect_step.reason.lower()
    )
    assert projection.can_advance is False


def test_missing_output_file_prevents_advancement(tmp_path: Path) -> None:
    """RED: If an output file is deleted on disk, step cannot remain DONE."""
    store, session = _setup_store_and_project(tmp_path / "proj")

    cap_file, cap_hash = _make_file(tmp_path / "proj" / "capture.c3d", b"c3d-data")
    cap_ref = ArtifactReference(
        artifact_id="art-c3d",
        path="capture.c3d",
        hash=cap_hash,
        schema="motion_capture.c3d/1",
        kind=ArtifactKind.OBSERVATION,
    )

    coordinator = WorkflowCoordinator(
        store=store,
        session_id=session.session_id,
        run_id="run-001",
        mode=WorkflowMode.FULL_BODY_3D,
    )
    coordinator.advance_step(
        step_id=WorkflowStepId.CAPTURE_IMPORT,
        outputs=(cap_ref,),
    )

    # Delete the output file
    cap_file.unlink()

    projection = coordinator.get_projection()
    step_proj = projection.get_step(WorkflowStepId.CAPTURE_IMPORT)
    assert step_proj.status == StepStatus.BLOCKED
    assert "missing" in step_proj.reason.lower()


# ============================================================================
# RED Cases: Single-View Coaching & Kinematic Pass Distinction
# ============================================================================


def test_single_view_coaching_never_requires_calibrated_3d_or_physics_engine(
    tmp_path: Path,
) -> None:
    """RED: Single-view coaching advances without calibrated 3-D or physics engines."""
    store, session = _setup_store_and_project(tmp_path / "proj")

    # Only single-camera coaching footage
    vid_file, vid_hash = _make_file(
        tmp_path / "proj" / "dtl_view.mp4", b"single-camera-video-stream"
    )
    vid_ref = ArtifactReference(
        artifact_id="art-video",
        path="dtl_view.mp4",
        hash=vid_hash,
        schema="motion_capture.c3d/1",  # observation schema
        kind=ArtifactKind.OBSERVATION,
    )

    # Empty engine availability (no MuJoCo, Drake, Simscape, or Pinocchio installed)
    coordinator = WorkflowCoordinator(
        store=store,
        session_id=session.session_id,
        run_id="run-coaching-1",
        mode=WorkflowMode.SINGLE_VIEW_COACHING,
        available_engines=frozenset(),  # NO PHYSICS ENGINES
    )

    coordinator.advance_step(
        step_id=WorkflowStepId.CAPTURE_IMPORT,
        outputs=(vid_ref,),
    )

    projection = coordinator.get_projection()
    # 3-D physics steps (DYNAMICS) are SKIPPED rather than BLOCKED/FAILED
    dyn_step = projection.get_step(WorkflowStepId.DYNAMICS)
    assert dyn_step.status == StepStatus.SKIPPED

    # Single view inspect targets is READY
    inspect_step = projection.get_step(WorkflowStepId.INSPECT_TARGETS)
    assert inspect_step.status == StepStatus.READY


def test_dynamics_cannot_inherit_kinematic_pass(tmp_path: Path) -> None:
    """RED: Dynamics requires real physics/dynamics output; kinematic pass rejected."""
    store, session = _setup_store_and_project(tmp_path / "proj")

    kin_file, kin_hash = _make_file(
        tmp_path / "proj" / "kinematics_only.json",
        b'{"q_pass": true, "physics": false}',
    )
    kin_ref = ArtifactReference(
        artifact_id="art-kin-pass",
        path="kinematics_only.json",
        hash=kin_hash,
        schema="pose_interchange/canonical/1",
        kind=ArtifactKind.STATIC_POSE,  # Only a pose / kinematic fit, not dynamic state or receipt
    )

    coordinator = WorkflowCoordinator(
        store=store,
        session_id=session.session_id,
        run_id="run-dyn-1",
        mode=WorkflowMode.FULL_BODY_3D,
    )

    # Attempting to satisfy DYNAMICS with a purely kinematic/pose artifact must be rejected
    with pytest.raises(ValueError, match="dynamics cannot inherit a kinematic pass"):
        coordinator.advance_step(
            step_id=WorkflowStepId.DYNAMICS,
            outputs=(kin_ref,),
        )


# ============================================================================
# GREEN Cases: Resume, Cancellation, Retry & Projection Parity
# ============================================================================


def test_resume_after_restart_and_projection_parity(tmp_path: Path) -> None:
    """GREEN: Reopening project restores workflow state and projection exactly."""
    store, session = _setup_store_and_project(tmp_path / "proj")

    cap_file, cap_hash = _make_file(tmp_path / "proj" / "capture.c3d", b"c3d-bytes")
    cap_ref = ArtifactReference(
        artifact_id="art-c3d",
        path="capture.c3d",
        hash=cap_hash,
        schema="motion_capture.c3d/1",
        kind=ArtifactKind.OBSERVATION,
    )

    coordinator = WorkflowCoordinator(
        store=store,
        session_id=session.session_id,
        run_id="run-001",
        mode=WorkflowMode.FULL_BODY_3D,
    )
    coordinator.advance_step(
        step_id=WorkflowStepId.CAPTURE_IMPORT,
        outputs=(cap_ref,),
    )

    # Reopen fresh coordinator pointed at same store/project
    fresh_store = SessionProjectStore(tmp_path / "proj")
    resumed_coordinator = WorkflowCoordinator.load(
        store=fresh_store,
        run_id="run-001",
    )

    projection = resumed_coordinator.get_projection()
    assert projection.current_step == WorkflowStepId.INSPECT_TARGETS
    cap_step = projection.get_step(WorkflowStepId.CAPTURE_IMPORT)
    assert cap_step.status == StepStatus.DONE
    assert cap_step.outputs[0].artifact_id == "art-c3d"

    # Verify projection dict representation matches Qt/React shared contract
    proj_dict = projection.to_dict()
    assert proj_dict["current_step"] == "inspect_targets"
    assert proj_dict["mode"] == "full_body_3d"
    assert len(proj_dict["steps"]) == 7
    assert proj_dict["steps"][0]["key"] == "capture_import"
    assert proj_dict["steps"][0]["status"] == "done"


def test_cancellation_and_retry_creates_new_attempt(tmp_path: Path) -> None:
    """GREEN: Cancellation preserves diagnostics; retry creates new attempt."""
    store, session = _setup_store_and_project(tmp_path / "proj")

    cap_file, cap_hash = _make_file(tmp_path / "proj" / "capture.c3d", b"c3d-bytes")
    cap_ref = ArtifactReference(
        artifact_id="art-c3d",
        path="capture.c3d",
        hash=cap_hash,
        schema="motion_capture.c3d/1",
        kind=ArtifactKind.OBSERVATION,
    )

    coordinator = WorkflowCoordinator(
        store=store,
        session_id=session.session_id,
        run_id="run-001",
        mode=WorkflowMode.FULL_BODY_3D,
    )
    coordinator.advance_step(
        step_id=WorkflowStepId.CAPTURE_IMPORT,
        outputs=(cap_ref,),
    )

    # Cancel FIT step
    coordinator.cancel_step(
        step_id=WorkflowStepId.FIT,
        reason="Solver convergence timeout requested by user",
    )

    proj_canceled = coordinator.get_projection()
    fit_step = proj_canceled.get_step(WorkflowStepId.FIT)
    assert fit_step.status == StepStatus.CANCELED
    assert "timeout" in fit_step.reason.lower()
    assert fit_step.attempt == 1

    # Retry FIT step creates attempt 2
    coordinator.retry_step(step_id=WorkflowStepId.FIT)

    proj_retried = coordinator.get_projection()
    fit_retried = proj_retried.get_step(WorkflowStepId.FIT)
    assert (
        fit_retried.status == StepStatus.READY
        or fit_retried.status == StepStatus.IN_PROGRESS
    )
    assert fit_retried.attempt == 2


def test_entry_from_imported_artifacts_skips_prior_stages(tmp_path: Path) -> None:
    """GREEN: Importing pre-existing simulation trace allows entry at COMPARE stage."""
    store, session = _setup_store_and_project(tmp_path / "proj")

    trace_file, trace_hash = _make_file(
        tmp_path / "proj" / "imported_trace.h5", b"trace-data-full"
    )
    receipt_file, receipt_hash = _make_file(
        tmp_path / "proj" / "receipt.json", b'{"status": "passed", "horizon": "G1"}'
    )

    trace_ref = ArtifactReference(
        artifact_id="art-trace",
        path="imported_trace.h5",
        hash=trace_hash,
        schema="simulation_backend.trace/2.1.0",
        kind=ArtifactKind.DYNAMIC_STATE,
    )
    receipt_ref = ArtifactReference(
        artifact_id="art-receipt",
        path="receipt.json",
        hash=receipt_hash,
        schema="pipeline.ground_support_receipt/1",
        kind=ArtifactKind.RECEIPT,
    )

    coordinator = WorkflowCoordinator.entry_from_artifacts(
        store=store,
        session_id=session.session_id,
        run_id="run-import-01",
        entry_step=WorkflowStepId.COMPARE,
        upstream_artifacts=(trace_ref, receipt_ref),
        mode=WorkflowMode.FULL_BODY_3D,
    )

    projection = coordinator.get_projection()
    assert projection.current_step == WorkflowStepId.COMPARE
    assert projection.get_step(WorkflowStepId.COMPARE).status == StepStatus.READY
    # Prior steps are marked SKIPPED or DONE as satisfied by imported artifacts
    assert projection.get_step(WorkflowStepId.DYNAMICS).status == StepStatus.DONE
