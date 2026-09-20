"""Focused TDD acceptance tests for artifact handoff and project context.

Issue: #10517 ([ORG-08] Unified Artifact and Project Context Handoff Between Workspaces)
Epic: #10508
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.shared.python.core.contracts.exceptions import StateError
from src.shared.python.workspace import (
    ArtifactKind,
    ArtifactReference,
    RunMetadata,
    SessionProjectStore,
    WorkspaceHandoff,
    convert_artifact,
    register_artifact_adapter,
)

pytestmark = pytest.mark.unit


def _sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _setup_project_and_session(root: Path):
    store = SessionProjectStore(root)
    store.create_project("proj-001", "Study Project")
    store.add_subject("subj-001", "Subject 1")
    store.add_subject("subj-002", "Subject 2")
    session1 = store.create_session("sess-001", "subj-001", "Capture Session 1")
    session2 = store.create_session("sess-002", "subj-002", "Capture Session 2")
    return store, session1, session2


def _make_dummy_file(path: Path, content: bytes) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path, _sha256_bytes(content)


# ============================================================================
# RED Cases: Rejection before writes
# ============================================================================


def test_reject_cross_session_subject_mismatch(tmp_path: Path) -> None:
    """RED: session belongs to subj-001, but handoff specifies subj-002 -> reject before write."""
    store, session1, _ = _setup_project_and_session(tmp_path / "proj")

    art_file, art_hash = _make_dummy_file(
        tmp_path / "proj" / "capture.c3d", b"c3d-content"
    )
    art_ref = ArtifactReference(
        artifact_id="art-c3d-1",
        path="capture.c3d",
        hash=art_hash,
        schema="motion_capture.c3d/1",
        kind=ArtifactKind.OBSERVATION,
    )

    handoff = WorkspaceHandoff(
        handoff_id="run-001",
        project_id="proj-001",
        session_id=session1.session_id,  # belongs to subj-001
        subject_id="subj-002",  # MISMATCH
        engine="mujoco",
        model_id="humanoid_g1",
        club={"loft_deg": 10.5, "mass_kg": 0.32},
        units={"length": "m", "mass": "kg", "time": "s"},
        frame="world",
        timebase={"rate_hz": 120.0, "t_start": 0.0},
        parameters={"solver": "rk4"},
        inputs=(art_ref,),
    )

    with pytest.raises(ValueError, match="cross-session subject mismatch"):
        store.register_run(handoff)

    # Verify no run was written to project
    loaded = store.load_project()
    assert "run-001" not in loaded.runs


def test_reject_unknown_frame(tmp_path: Path) -> None:
    """RED: unknown frame must be rejected before write."""
    store, session1, _ = _setup_project_and_session(tmp_path / "proj")

    art_file, art_hash = _make_dummy_file(
        tmp_path / "proj" / "capture.c3d", b"c3d-content"
    )
    art_ref = ArtifactReference(
        artifact_id="art-c3d-1",
        path="capture.c3d",
        hash=art_hash,
        schema="motion_capture.c3d/1",
        kind=ArtifactKind.OBSERVATION,
    )

    with pytest.raises(ValueError, match="unknown or unsupported frame"):
        handoff = WorkspaceHandoff(
            handoff_id="run-001",
            project_id="proj-001",
            session_id=session1.session_id,
            subject_id=session1.subject_id,
            engine="mujoco",
            model_id="humanoid_g1",
            club={"loft_deg": 10.5},
            units={"length": "m", "mass": "kg"},
            frame="unknown_arbitrary_frame_999",  # UNKNOWN FRAME
            timebase={"rate_hz": 120.0},
            parameters={},
            inputs=(art_ref,),
        )
        store.register_run(handoff)

    assert "run-001" not in store.load_project().runs


def test_reject_unknown_schema(tmp_path: Path) -> None:
    """RED: unknown artifact schema must be rejected before write."""
    store, session1, _ = _setup_project_and_session(tmp_path / "proj")

    art_file, art_hash = _make_dummy_file(
        tmp_path / "proj" / "data.bin", b"binary-data"
    )
    art_ref = ArtifactReference(
        artifact_id="art-bad-1",
        path="data.bin",
        hash=art_hash,
        schema="nonexistent.schema/999",  # UNKNOWN SCHEMA
        kind=ArtifactKind.OBSERVATION,
    )

    with pytest.raises(ValueError, match="unknown or unsupported schema"):
        handoff = WorkspaceHandoff(
            handoff_id="run-001",
            project_id="proj-001",
            session_id=session1.session_id,
            subject_id=session1.subject_id,
            engine="mujoco",
            model_id="humanoid_g1",
            club={},
            units={"length": "m"},
            frame="world",
            timebase={"rate_hz": 120.0},
            parameters={},
            inputs=(art_ref,),
        )
        store.register_run(handoff)

    assert "run-001" not in store.load_project().runs


def test_reject_missing_artifact(tmp_path: Path) -> None:
    """RED: missing artifact referenced in handoff must be rejected before write."""
    store, session1, _ = _setup_project_and_session(tmp_path / "proj")

    art_ref = ArtifactReference(
        artifact_id="art-missing-1",
        path="nonexistent/file.c3d",  # FILE DOES NOT EXIST
        hash="sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        schema="motion_capture.c3d/1",
        kind=ArtifactKind.OBSERVATION,
    )

    handoff = WorkspaceHandoff(
        handoff_id="run-001",
        project_id="proj-001",
        session_id=session1.session_id,
        subject_id=session1.subject_id,
        engine="mujoco",
        model_id="humanoid_g1",
        club={},
        units={"length": "m"},
        frame="world",
        timebase={"rate_hz": 120.0},
        parameters={},
        inputs=(art_ref,),
    )

    with pytest.raises((FileNotFoundError, ValueError), match="missing"):
        store.register_run(handoff)

    assert "run-001" not in store.load_project().runs


def test_reject_altered_hash(tmp_path: Path) -> None:
    """RED: altered file hash must be rejected before write."""
    store, session1, _ = _setup_project_and_session(tmp_path / "proj")

    art_file, _ = _make_dummy_file(tmp_path / "proj" / "capture.c3d", b"actual-content")
    altered_hash = (
        "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    )

    art_ref = ArtifactReference(
        artifact_id="art-altered-1",
        path="capture.c3d",
        hash=altered_hash,  # MISMATCH
        schema="motion_capture.c3d/1",
        kind=ArtifactKind.OBSERVATION,
    )

    handoff = WorkspaceHandoff(
        handoff_id="run-001",
        project_id="proj-001",
        session_id=session1.session_id,
        subject_id=session1.subject_id,
        engine="mujoco",
        model_id="humanoid_g1",
        club={},
        units={"length": "m"},
        frame="world",
        timebase={"rate_hz": 120.0},
        parameters={},
        inputs=(art_ref,),
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        store.register_run(handoff)

    assert "run-001" not in store.load_project().runs


# ============================================================================
# RED Cases: Interrupted save & Migration of older metadata
# ============================================================================


def test_interrupted_save_leaves_previous_project_readable(
    tmp_path: Path, monkeypatch
) -> None:
    """RED: interrupted save leaves previous project readable without corruption."""
    store, session1, _ = _setup_project_and_session(tmp_path / "proj")

    art_file, art_hash = _make_dummy_file(tmp_path / "proj" / "pose.json", b"pose-data")
    art_ref = ArtifactReference(
        artifact_id="art-pose-1",
        path="pose.json",
        hash=art_hash,
        schema="pose_interchange/canonical/1",
        kind=ArtifactKind.STATIC_POSE,
    )
    handoff = WorkspaceHandoff(
        handoff_id="run-001",
        project_id="proj-001",
        session_id=session1.session_id,
        subject_id=session1.subject_id,
        engine="pinocchio",
        model_id="humanoid_g1",
        club={"loft_deg": 10.5},
        units={"length": "m", "mass": "kg"},
        frame="world",
        timebase={"rate_hz": 120.0},
        parameters={},
        inputs=(art_ref,),
    )
    store.register_run(handoff)

    # Verify baseline state is saved
    assert "run-001" in store.load_project().runs

    # Simulate crash / interruption during atomic write of a new run
    from src.shared.python.workspace import project_store

    def _failing_atomic_write(*args, **kwargs):
        raise OSError("Disk write interrupted / power failure")

    monkeypatch.setattr(project_store, "_atomic_write_json", _failing_atomic_write)

    handoff2 = WorkspaceHandoff(
        handoff_id="run-002",
        project_id="proj-001",
        session_id=session1.session_id,
        subject_id=session1.subject_id,
        engine="pinocchio",
        model_id="humanoid_g1",
        club={"loft_deg": 10.5},
        units={"length": "m", "mass": "kg"},
        frame="world",
        timebase={"rate_hz": 120.0},
        parameters={},
        inputs=(art_ref,),
    )

    with pytest.raises(StateError, match="could not write project metadata"):
        store.register_run(handoff2)

    # Fresh store pointing to same directory must still read the uncorrupted previous project
    fresh_store = SessionProjectStore(tmp_path / "proj")
    loaded = fresh_store.load_project()
    assert "run-001" in loaded.runs
    assert "run-002" not in loaded.runs
    assert len(loaded.sessions) == 2


def test_loading_older_metadata_preserves_unknown_supported_fields(
    tmp_path: Path,
) -> None:
    """RED: loading older metadata preserves unknown supported fields via documented migration."""
    proj_dir = tmp_path / "legacy_proj"
    proj_dir.mkdir(parents=True, exist_ok=True)
    proj_file = proj_dir / "project.json"

    # Pre-existing project.json with older fields and unknown custom fields
    legacy_payload = {
        "project_id": "proj-legacy",
        "name": "Legacy Project",
        "root": str(proj_dir),
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "subjects": {
            "subj-001": {
                "subject_id": "subj-001",
                "display_name": "Subject 1",
                "metadata": {},
            }
        },
        "sessions": {
            "sess-001": {
                "session_id": "sess-001",
                "subject_id": "subj-001",
                "name": "S1",
                "created_at": "2026-01-01T00:00:00Z",
                "metadata": {},
            }
        },
        "datasets": {},
        # Unknown supported fields added by external tools or earlier formats:
        "custom_lab_note": "Calibrated in Munich",
        "legacy_workflow_version": 42,
        "external_provenance": {"origin": "lab_a", "sync_id": 999},
    }
    proj_file.write_text(json.dumps(legacy_payload, indent=2), encoding="utf-8")

    store = SessionProjectStore(proj_dir)
    loaded = store.load_project()

    # The store successfully loads, provides default runs/active_run_id, and preserves extra fields
    assert loaded.project_id == "proj-legacy"
    assert loaded.runs == {}
    assert loaded.active_run_id is None
    assert loaded.extra_fields["custom_lab_note"] == "Calibrated in Munich"
    assert loaded.extra_fields["legacy_workflow_version"] == 42
    assert loaded.extra_fields["external_provenance"] == {
        "origin": "lab_a",
        "sync_id": 999,
    }

    # Modifying the project retains the unknown fields upon saving
    store.add_subject("subj-002", "Subject 2")
    fresh_loaded = SessionProjectStore(proj_dir).load_project()
    assert fresh_loaded.extra_fields["custom_lab_note"] == "Calibrated in Munich"
    assert fresh_loaded.extra_fields["legacy_workflow_version"] == 42
    assert "subj-002" in fresh_loaded.subjects


# ============================================================================
# GREEN Cases: Round-trip & Context cloning
# ============================================================================


def test_save_reopen_round_trip_for_all_artifact_references(tmp_path: Path) -> None:
    """GREEN: save/reopen round-trip for capture/pose/run/flight/receipt references."""
    proj_root = tmp_path / "study"
    store, session1, _ = _setup_project_and_session(proj_root)

    # Create 4 distinct representative artifacts
    cap_file, cap_hash = _make_dummy_file(
        proj_root / "inputs" / "capture.c3d", b"c3d_raw_data_stream"
    )
    pose_file, pose_hash = _make_dummy_file(
        proj_root / "inputs" / "starting_pose.json", b'{"q": [0.0, 0.1]}'
    )
    run_file, run_hash = _make_dummy_file(
        proj_root / "outputs" / "simulation_trace.h5", b"hdf5_trace_data"
    )
    flight_file, flight_hash = _make_dummy_file(
        proj_root / "outputs" / "ball_flight.json",
        b'{"format": "swing_sim.ball_flight_trajectory/1"}',
    )
    receipt_file, receipt_hash = _make_dummy_file(
        proj_root / "outputs" / "receipt.json", b'{"status": "passed", "horizon": "G1"}'
    )

    cap_ref = ArtifactReference(
        artifact_id="art-capture",
        path="inputs/capture.c3d",
        hash=cap_hash,
        schema="motion_capture.c3d/1",
        kind=ArtifactKind.OBSERVATION,
        metadata={"camera_count": 8},
    )
    pose_ref = ArtifactReference(
        artifact_id="art-pose",
        path="inputs/starting_pose.json",
        hash=pose_hash,
        schema="pose_interchange/canonical/1",
        kind=ArtifactKind.STATIC_POSE,
        metadata={"convention": "canonical-core"},
    )
    run_ref = ArtifactReference(
        artifact_id="art-run",
        path="outputs/simulation_trace.h5",
        hash=run_hash,
        schema="simulation_backend.trace/2.1.0",
        kind=ArtifactKind.DYNAMIC_STATE,
        metadata={"backend": "mujoco"},
    )
    flight_ref = ArtifactReference(
        artifact_id="art-flight",
        path="outputs/ball_flight.json",
        hash=flight_hash,
        schema="swing_sim.ball_flight_trajectory/1",
        kind=ArtifactKind.TRAJECTORY,
        metadata={"model": "Penner"},
    )
    receipt_ref = ArtifactReference(
        artifact_id="art-receipt",
        path="outputs/receipt.json",
        hash=receipt_hash,
        schema="pipeline.ground_support_receipt/1",
        kind=ArtifactKind.RECEIPT,
        metadata={"gate": "G1"},
    )

    handoff = WorkspaceHandoff(
        handoff_id="run-001",
        project_id="proj-001",
        session_id=session1.session_id,
        subject_id=session1.subject_id,
        engine="mujoco",
        model_id="humanoid_g1",
        club={"club_id": "driver-01", "loft_deg": 10.5, "mass_kg": 0.32},
        units={"length": "m", "mass": "kg", "time": "s", "angle": "rad"},
        frame="world",
        timebase={"rate_hz": 120.0, "dt": 0.008333},
        parameters={"solver": "rk4", "iterations": 100},
        inputs=(cap_ref, pose_ref),
        outputs=(run_ref, flight_ref, receipt_ref),
        status="completed",
        qualification={"gate": "G1", "passed": True},
    )

    store.register_run(handoff)
    store.set_active_run("run-001")

    # Reopen via a fresh store
    fresh = SessionProjectStore(proj_root)
    loaded_project = fresh.load_project()
    active_run = fresh.get_active_run()

    assert loaded_project.active_run_id == "run-001"
    assert active_run is not None
    assert active_run.run_id == "run-001"
    assert active_run.engine == "mujoco"
    assert active_run.model_id == "humanoid_g1"
    assert active_run.club["club_id"] == "driver-01"
    assert active_run.units["length"] == "m"
    assert active_run.frame == "world"
    assert active_run.timebase["rate_hz"] == 120.0
    assert active_run.status == "completed"
    assert active_run.qualification == {"gate": "G1", "passed": True}

    # Verify all 5 artifact references round-tripped with native paths and hashes
    inputs_by_id = {art.artifact_id: art for art in active_run.inputs}
    outputs_by_id = {art.artifact_id: art for art in active_run.outputs}

    assert inputs_by_id["art-capture"].kind == ArtifactKind.OBSERVATION
    assert inputs_by_id["art-capture"].hash == cap_hash
    assert inputs_by_id["art-pose"].kind == ArtifactKind.STATIC_POSE
    assert inputs_by_id["art-pose"].hash == pose_hash

    assert outputs_by_id["art-run"].kind == ArtifactKind.DYNAMIC_STATE
    assert outputs_by_id["art-run"].hash == run_hash
    assert outputs_by_id["art-flight"].kind == ArtifactKind.TRAJECTORY
    assert outputs_by_id["art-flight"].hash == flight_hash
    assert outputs_by_id["art-receipt"].kind == ArtifactKind.RECEIPT
    assert outputs_by_id["art-receipt"].hash == receipt_hash


def test_copying_context_creates_new_run_id_not_overwritten_evidence(
    tmp_path: Path,
) -> None:
    """GREEN: copying context creates a new run ID without overwriting source evidence."""
    proj_root = tmp_path / "study"
    store, session1, _ = _setup_project_and_session(proj_root)

    cap_file, cap_hash = _make_dummy_file(
        proj_root / "inputs" / "capture.c3d", b"c3d_raw"
    )
    run_file, run_hash = _make_dummy_file(
        proj_root / "outputs" / "run1.h5", b"trace_run_1"
    )

    cap_ref = ArtifactReference(
        artifact_id="art-c3d",
        path="inputs/capture.c3d",
        hash=cap_hash,
        schema="motion_capture.c3d/1",
        kind=ArtifactKind.OBSERVATION,
    )
    run_ref = ArtifactReference(
        artifact_id="art-run1",
        path="outputs/run1.h5",
        hash=run_hash,
        schema="simulation_backend.trace/2.1.0",
        kind=ArtifactKind.DYNAMIC_STATE,
    )

    source_handoff = WorkspaceHandoff(
        handoff_id="run-baseline",
        project_id="proj-001",
        session_id=session1.session_id,
        subject_id=session1.subject_id,
        engine="mujoco",
        model_id="humanoid_g1",
        club={"loft_deg": 10.5},
        units={"length": "m"},
        frame="world",
        timebase={"rate_hz": 120.0},
        parameters={"stiffness": 500.0},
        inputs=(cap_ref,),
        outputs=(run_ref,),
        status="completed",
        qualification={"passed": True, "score": 98.5},
    )
    store.register_run(source_handoff)

    # Clone run context with modified parameters for a new experiment
    cloned_run = store.clone_run(
        source_run_id="run-baseline",
        new_run_id="run-variant-01",
        parameters={"stiffness": 750.0},
    )

    assert cloned_run.run_id == "run-variant-01"
    assert cloned_run.parameters["stiffness"] == 750.0
    # Inputs and club snapshot are preserved
    assert len(cloned_run.inputs) == 1
    assert cloned_run.inputs[0].artifact_id == "art-c3d"
    assert cloned_run.club == {"loft_deg": 10.5}

    # CRITICAL: Evidence from source run is NOT carried over as completed outputs of new run
    assert cloned_run.outputs == ()
    assert cloned_run.qualification is None
    assert cloned_run.status == "draft"

    # Source run evidence is untouched and intact in the store
    reloaded_source = store.load_run("run-baseline")
    assert reloaded_source.run_id == "run-baseline"
    assert len(reloaded_source.outputs) == 1
    assert reloaded_source.outputs[0].artifact_id == "art-run1"
    assert reloaded_source.qualification == {"passed": True, "score": 98.5}
    assert reloaded_source.status == "completed"


def test_explains_missing_files_without_inventing_substitutes(tmp_path: Path) -> None:
    """Explains missing files without inventing substitutes."""
    proj_root = tmp_path / "study"
    store, session1, _ = _setup_project_and_session(proj_root)

    art_file, art_hash = _make_dummy_file(proj_root / "temp.c3d", b"temp-content")
    art_ref = ArtifactReference(
        artifact_id="art-temp",
        path="temp.c3d",
        hash=art_hash,
        schema="motion_capture.c3d/1",
        kind=ArtifactKind.OBSERVATION,
    )
    handoff = WorkspaceHandoff(
        handoff_id="run-001",
        project_id="proj-001",
        session_id=session1.session_id,
        subject_id=session1.subject_id,
        engine="mujoco",
        model_id="humanoid_g1",
        club={},
        units={"length": "m"},
        frame="world",
        timebase={"rate_hz": 120.0},
        parameters={},
        inputs=(art_ref,),
    )
    store.register_run(handoff)

    # Now remove the file from disk
    art_file.unlink()

    # Checking artifacts explains the missing file explicitly without substituting a mock
    missing = store.check_run_artifacts("run-001")
    assert len(missing) == 1
    assert "art-temp" in missing[0]
    assert "missing" in missing[0].lower()
    assert "temp.c3d" in missing[0]


def test_artifact_conversion_requires_named_adapter_and_provenance(
    tmp_path: Path,
) -> None:
    """Artifact contracts distinct: conversion requires named adapter with provenance."""
    proj_root = tmp_path / "study"
    proj_root.mkdir(parents=True, exist_ok=True)

    src_file, src_hash = _make_dummy_file(
        proj_root / "source_capture.c3d", b"raw_marker_data"
    )
    src_art = ArtifactReference(
        artifact_id="art-src-1",
        path="source_capture.c3d",
        hash=src_hash,
        schema="motion_capture.c3d/1",
        kind=ArtifactKind.OBSERVATION,
    )

    # Conversion without a registered adapter fails
    with pytest.raises(ValueError, match="unknown or unregistered adapter"):
        convert_artifact(
            source=src_art,
            target_kind=ArtifactKind.STATIC_POSE,
            adapter_name="unregistered_adapter",
            output_path="out_pose.json",
            provenance={"engine": "mujoco", "created_at": "2026-09-19T00:00:00Z"},
            artifact_id="art-pose-1",
            schema="pose_interchange/canonical/1",
            root=proj_root,
        )

    # Register a named adapter
    def _mock_c3d_to_pose_adapter(
        src_p: Path, dst_p: Path, prov: dict[str, object]
    ) -> None:
        dst_p.write_text(json.dumps({"fitted_q": [0.0, 0.0]}), encoding="utf-8")

    register_artifact_adapter(
        name="ik_pose_fitter_v1",
        source_kind=ArtifactKind.OBSERVATION,
        target_kind=ArtifactKind.STATIC_POSE,
        adapter_fn=_mock_c3d_to_pose_adapter,
    )

    # Conversion without provenance fails
    with pytest.raises(ValueError, match="provenance must be non-empty"):
        convert_artifact(
            source=src_art,
            target_kind=ArtifactKind.STATIC_POSE,
            adapter_name="ik_pose_fitter_v1",
            output_path="out_pose.json",
            provenance={},  # EMPTY PROVENANCE REJECTED
            artifact_id="art-pose-1",
            schema="pose_interchange/canonical/1",
            root=proj_root,
        )

    # Conversion with named adapter and provenance succeeds
    converted = convert_artifact(
        source=src_art,
        target_kind=ArtifactKind.STATIC_POSE,
        adapter_name="ik_pose_fitter_v1",
        output_path="out_pose.json",
        provenance={
            "engine": "mujoco",
            "solver": "ik_horizon",
            "created_at": "2026-09-19T00:00:00Z",
        },
        artifact_id="art-pose-1",
        schema="pose_interchange/canonical/1",
        root=proj_root,
    )

    assert converted.kind == ArtifactKind.STATIC_POSE
    assert converted.artifact_id == "art-pose-1"
    assert (proj_root / "out_pose.json").exists()
    assert converted.metadata["conversion_provenance"]["adapter"] == "ik_pose_fitter_v1"
    assert (
        converted.metadata["conversion_provenance"]["source_artifact_id"] == "art-src-1"
    )
    assert (
        converted.metadata["conversion_provenance"]["source_kind"]
        == ArtifactKind.OBSERVATION
    )


def test_failed_or_canceled_run_cannot_claim_completed_qualification(
    tmp_path: Path,
) -> None:
    """Canceled/failed jobs never become completed results."""
    store, session1, _ = _setup_project_and_session(tmp_path / "proj")

    art_file, art_hash = _make_dummy_file(tmp_path / "proj" / "trace.h5", b"trace-data")
    art_ref = ArtifactReference(
        artifact_id="art-trace",
        path="trace.h5",
        hash=art_hash,
        schema="simulation_backend.trace/2.1.0",
        kind=ArtifactKind.DYNAMIC_STATE,
    )

    with pytest.raises(
        ValueError, match="cannot be marked with completed qualification"
    ):
        WorkspaceHandoff(
            handoff_id="run-001",
            project_id="proj-001",
            session_id=session1.session_id,
            subject_id=session1.subject_id,
            engine="mujoco",
            model_id="humanoid_g1",
            club={},
            units={"length": "m"},
            frame="world",
            timebase={"rate_hz": 120.0},
            parameters={},
            outputs=(art_ref,),
            status="failed",  # FAILED
            qualification={"passed": True},  # CANNOT BE QUALIFIED/PASSED
        )
