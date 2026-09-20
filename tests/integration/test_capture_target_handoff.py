"""Integration acceptance tests for connecting Capture Rig, Optical Import, Pose Inspection, and Model Calibration (#10519).

Epic: #10508
Issue: #10519 ([ORG-10] Connect Capture Rig, Optical Import, Pose Inspection, and Model Calibration Workspaces)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytestmark = pytest.mark.integration


from src.shared.python.workspace import (
    ArtifactKind,
    ArtifactReference,
    SessionProjectStore,
    StepStatus,
    WorkflowCoordinator,
    WorkflowMode,
    WorkflowStepId,
)
from src.shared.python.workspace.capture_inspection_handoff import (
    CaptureInspectionHandoff,
    CropSpec,
    EstimatorType,
    FreeMoCapJobAdapter,
    JobStatus,
    TrimSpec,
)

pytestmark = pytest.mark.integration


def _sha256_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _make_file(path: Path, content: bytes) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path, _sha256_bytes(content)


def _setup_project_and_session(root: Path):
    store = SessionProjectStore(root)
    store.create_project("proj-capture", "Capture Rig Integration Project")
    store.add_subject("subj-golfer", "Pro Golfer")
    session = store.create_session("sess-001", "subj-golfer", "Driver Take 1")
    return store, session


# ============================================================================
# RED Cases: Trim/Crop/Offset Survival & Masked C3D Missing Samples & Rejection
# ============================================================================


def test_trim_crop_time_offset_survive_handoff(tmp_path: Path) -> None:
    """RED: Trim, crop, and time offset parameters survive handoff to target inspection."""
    store, session = _setup_project_and_session(tmp_path / "proj")

    video_file, video_hash = _make_file(
        tmp_path / "proj" / "dtl_view.mp4", b"dummy-video-bytes-dtl"
    )

    trim = TrimSpec(start_frame=15, end_frame=90, start_time_s=0.25, end_time_s=1.50)
    crop = CropSpec(x=50, y=100, width=540, height=960)
    time_offset_s = 0.125

    handoff = CaptureInspectionHandoff(store=store, session_id=session.session_id)

    # Prepare handoff with explicit trim, crop, and offset
    prepared = handoff.prepare_video_inspection(
        video_path=video_file,
        view_name="dtl",
        trim=trim,
        crop=crop,
        time_offset_s=time_offset_s,
        fps=60.0,
    )

    assert prepared.trim == trim
    assert prepared.crop == crop
    assert prepared.time_offset_s == time_offset_s
    assert prepared.fps == 60.0

    # Verify time conversion incorporates time_offset
    # Frame 0 of trimmed segment corresponds to start_frame (15) -> 15/60 + 0.125 = 0.375s
    t0 = prepared.frame_to_time_s(0)
    assert abs(t0 - (15 / 60.0 + 0.125)) < 1e-6

    # Verify artifact reference metadata preserves trim/crop/offset
    art = prepared.to_artifact_reference()
    assert art.metadata["trim"]["start_frame"] == 15
    assert art.metadata["crop"]["width"] == 540
    assert art.metadata["time_offset_s"] == 0.125


def test_c3d_missing_samples_masked_and_incompatible_units_or_frame_fail(
    tmp_path: Path,
) -> None:
    """RED: C3D missing samples stay masked (NaN); incompatible units/frames are rejected;

    2-D pixel coordinates are never claimed to be 3-D metric coordinates.
    """
    store, session = _setup_project_and_session(tmp_path / "proj")
    handoff = CaptureInspectionHandoff(store=store, session_id=session.session_id)

    # 1. Verify C3D missing samples stay masked
    c3d_file, c3d_hash = _make_file(
        tmp_path / "proj" / "capture.c3d", b"fake-c3d-sample-content"
    )

    # Ingest synthetic mock marker data where marker 2 has missing samples (NaNs)
    marker_names = ("C7", "L_Shoulder", "R_Shoulder")
    # 3 frames, 3 markers, 3 coords
    coords = np.array(
        [
            [[0.1, 0.2, 1.5], [np.nan, np.nan, np.nan], [0.3, 0.2, 1.4]],
            [[0.12, 0.21, 1.5], [np.nan, np.nan, np.nan], [0.31, 0.21, 1.4]],
            [[0.14, 0.22, 1.5], [0.25, 0.22, 1.45], [0.32, 0.22, 1.4]],
        ],
        dtype=np.float64,
    )

    optical_target = handoff.import_optical_markers(
        name="markers_raw",
        marker_names=marker_names,
        coordinates_m=coords,
        fps=120.0,
        units="m",
        frame="canonical",
    )

    # Missing samples MUST remain NaN (masked), not filled with 0.0
    assert np.isnan(optical_target.coordinates[0, 1, 0])
    assert np.isnan(optical_target.coordinates[1, 1, 0])
    assert not np.isnan(optical_target.coordinates[2, 1, 0])
    assert optical_target.is_masked(frame_idx=0, marker_idx=1) is True

    # 2. Incompatible spatial units must fail explicitly
    with pytest.raises(ValueError, match="unsupported units|unknown.*units"):
        handoff.import_optical_markers(
            name="markers_bad_units",
            marker_names=marker_names,
            coordinates_m=coords,
            fps=120.0,
            units="furlongs",  # INCOMPATIBLE
            frame="canonical",
        )

    # 3. Incompatible reference frame must fail explicitly
    with pytest.raises(ValueError, match="unknown or unsupported frame"):
        handoff.import_optical_markers(
            name="markers_bad_frame",
            marker_names=marker_names,
            coordinates_m=coords,
            fps=120.0,
            units="m",
            frame="galactic_inertial_arbitrary",  # INCOMPATIBLE
        )

    # 4. Never pretend 2-D coordinates are metric 3-D
    with pytest.raises(ValueError, match="cannot treat 2-D coordinates as 3-D metric"):
        handoff.register_2d_pixels_as_3d_metric(
            name="fake_3d",
            pixel_coordinates=np.zeros((10, 5, 2)),
        )


# ============================================================================
# RED Cases: FreeMoCap Pre-Spawn Validation & Cancellation
# ============================================================================


def test_freemocap_missing_inputs_or_dependency_fails_before_process_spawn(
    tmp_path: Path,
) -> None:
    """RED: FreeMoCap adapter validates directories and dependencies before spawning."""
    adapter = FreeMoCapJobAdapter()

    # 1. Non-existent input directory fails before spawn
    missing_dir = tmp_path / "non_existent_input"
    out_dir = tmp_path / "output"
    with pytest.raises(ValueError, match="input directory does not exist"):
        adapter.start_job(input_dir=missing_dir, output_dir=out_dir)

    # 2. Empty input directory fails before spawn
    empty_dir = tmp_path / "empty_input"
    empty_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="contains no video files"):
        adapter.start_job(input_dir=empty_dir, output_dir=out_dir)

    # 3. Missing python interpreter / freemocap env fails before spawn
    vid_file, _ = _make_file(empty_dir / "view1.mp4", b"video-content")
    with pytest.raises(FileNotFoundError, match="freemocap executable not found"):
        adapter.start_job(
            input_dir=empty_dir,
            output_dir=out_dir,
            freemocap_env_python=tmp_path / "non_existent_venv" / "bin" / "python",
        )


def test_freemocap_cancellation_leaves_source_untouched(tmp_path: Path) -> None:
    """RED: Job cancellation preserves source videos and keeps HMR2/AGPL isolation."""
    input_dir = tmp_path / "session_videos"
    v1, h1 = _make_file(input_dir / "cam_0.mp4", b"raw-cam-0-footage")
    v2, h2 = _make_file(input_dir / "cam_1.mp4", b"raw-cam-1-footage")
    out_dir = tmp_path / "output_freemocap"

    adapter = FreeMoCapJobAdapter()
    job = adapter.start_job(
        input_dir=input_dir,
        output_dir=out_dir,
        dry_run=True,  # Test scaffold
    )

    assert job.status in (JobStatus.RUNNING, JobStatus.COMPLETED)

    # Cancel job
    job.cancel()
    assert job.status == JobStatus.CANCELED

    # Source videos must be completely untouched (hashes match exactly)
    assert _sha256_bytes(v1.read_bytes()) == h1
    assert _sha256_bytes(v2.read_bytes()) == h2

    # Verify license isolation: FreeMoCap is not imported into main process
    import sys

    assert "freemocap" not in sys.modules


# ============================================================================
# GREEN Cases: Media Ingest -> Pose Inspection -> Registered Store Target
# ============================================================================


def test_deterministic_imported_media_to_inspection_to_registered_target(
    tmp_path: Path,
) -> None:
    """GREEN: Deterministic imported media -> MediaPipe & OpenPose choices -> store registration."""
    store, session = _setup_project_and_session(tmp_path / "proj")

    vid_path, vid_hash = _make_file(
        tmp_path / "proj" / "take1_dtl.mp4", b"synthetic-camera-stream"
    )

    handoff = CaptureInspectionHandoff(store=store, session_id=session.session_id)

    # Ingest with MediaPipe
    mp_obs = handoff.create_2d_observation_set(
        video_path=vid_path,
        view_name="dtl",
        estimator=EstimatorType.MEDIAPIPE,
        keypoints_px={
            0: {"nose": (320.0, 180.0), "left_shoulder": (350.0, 240.0)},
            1: {"nose": (322.0, 182.0), "left_shoulder": (352.0, 242.0)},
        },
        confidence={
            0: {"nose": 0.95, "left_shoulder": 0.90},
            1: {"nose": 0.94, "left_shoulder": 0.88},
        },
        fps=30.0,
        time_offset_s=0.0,
    )

    # Ingest separate observation set with OpenPose
    op_obs = handoff.create_2d_observation_set(
        video_path=vid_path,
        view_name="dtl",
        estimator=EstimatorType.OPENPOSE_DNN,
        keypoints_px={
            0: {"neck": (335.0, 210.0), "mid_hip": (330.0, 450.0)},
            1: {"neck": (337.0, 212.0), "mid_hip": (332.0, 452.0)},
        },
        confidence={
            0: {"neck": 0.85, "mid_hip": 0.82},
            1: {"neck": 0.84, "mid_hip": 0.80},
        },
        fps=30.0,
        time_offset_s=0.0,
    )

    # Separate observation sets maintained
    assert mp_obs.estimator == EstimatorType.MEDIAPIPE
    assert op_obs.estimator == EstimatorType.OPENPOSE_DNN
    assert "nose" in mp_obs.keypoint_names
    assert "neck" in op_obs.keypoint_names

    # Register into project store automatically without manual path re-entry
    target_art = handoff.register_target_in_store(
        run_id="run-capture-001",
        observations=(mp_obs, op_obs),
        annotations={"swing_phase": "downswing", "address_frame": 10},
        calibration={"view": "dtl", "lens": "standard_50mm"},
        club_metadata={"club_type": "driver", "length_in": 45.5, "loft_deg": 9.5},
    )

    assert target_art.artifact_id.startswith("art-target-")
    assert target_art.kind == ArtifactKind.OBSERVATION

    # Store verification
    project = store.load_project()
    assert "run-capture-001" in project.runs
    recorded_run = project.runs["run-capture-001"]
    assert recorded_run.club["club_type"] == "driver"
    assert recorded_run.metadata["annotations"]["swing_phase"] == "downswing"
    assert recorded_run.metadata["calibration"]["lens"] == "standard_50mm"


def test_single_view_coaching_works_without_engine_gpu_camera(
    tmp_path: Path,
) -> None:
    """GREEN: Single-view coaching advances without engine, GPU, or camera hardware."""
    store, session = _setup_project_and_session(tmp_path / "proj")

    vid_path, vid_hash = _make_file(
        tmp_path / "proj" / "single_phone_swing.mp4", b"single-view-video"
    )

    handoff = CaptureInspectionHandoff(store=store, session_id=session.session_id)

    # Initialize single-view coaching workflow coordinator (empty physics engines)
    coordinator = WorkflowCoordinator(
        store=store,
        session_id=session.session_id,
        run_id="run-coaching-single-view",
        mode=WorkflowMode.SINGLE_VIEW_COACHING,
        available_engines=frozenset(),  # NO PHYSICS ENGINES REQUIRED
    )

    # Step 1: Capture / Import
    vid_ref = ArtifactReference(
        artifact_id="art-coaching-vid",
        path="single_phone_swing.mp4",
        hash=vid_hash,
        schema="motion_capture.c3d/1",
        kind=ArtifactKind.OBSERVATION,
    )
    coordinator.advance_step(WorkflowStepId.CAPTURE_IMPORT, outputs=(vid_ref,))

    # Step 2: Open in Inspect Targets -> extract 2-D observation
    obs = handoff.create_2d_observation_set(
        video_path=vid_path,
        view_name="face_on",
        estimator=EstimatorType.MEDIAPIPE,
        keypoints_px={0: {"nose": (200.0, 150.0)}},
        confidence={0: {"nose": 0.99}},
        fps=30.0,
    )
    obs_file = tmp_path / "proj" / "observations_face_on.json"
    obs_file.write_text(json.dumps(obs.to_dict()), encoding="utf-8")
    obs_ref = ArtifactReference(
        artifact_id="art-obs-face-on",
        path="observations_face_on.json",
        hash=_sha256_bytes(obs_file.read_bytes()),
        schema="pipeline.ground_support_receipt/1",
        kind=ArtifactKind.OBSERVATION,
    )
    coordinator.advance_step(WorkflowStepId.INSPECT_TARGETS, outputs=(obs_ref,))

    projection = coordinator.get_projection()
    # 3D dynamics is SKIPPED
    assert projection.get_step(WorkflowStepId.DYNAMICS).status == StepStatus.SKIPPED
    assert projection.get_step(WorkflowStepId.INSPECT_TARGETS).status == StepStatus.DONE
