"""Tests for Pose2Sim multi-camera ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.contracts import (
    Calibration,
    Keypoint,
    KeypointFrame,
    KeypointSequence,
)
from src.shared.python.motion_pipeline.sources.pose2sim_adapter import (
    Pose2SimAdapter,
    Pose2SimDetector,
    _aggregate_confidence,
    _find_keypoint,
    load_pose2sim_observations,
)

pytestmark = pytest.mark.unit

# Anchored to this file: the autouse _prevent_repo_root_io fixture (#7935)
# chdirs every test to tmp_path, so a CWD-relative fixture path never resolves.
FIXTURE_ROOT = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "data"
    / "motion_pipeline"
    / "pose2sim"
    / "sample_session"
)


def test_pose2sim_ingests_multicamera_fixture() -> None:
    observations = load_pose2sim_observations(FIXTURE_ROOT)

    assert observations.detector == "mediapipe"
    assert set(observations.camera_observations) == {"cam0", "cam1"}
    assert observations.triangulated is not None
    assert observations.triangulated.num_frames == 2
    assert observations.calibration.id == "pose2sim-sample"


def test_pose2sim_preserves_camera_confidence() -> None:
    observations = load_pose2sim_observations(FIXTURE_ROOT)

    cam0 = observations.camera_observations["cam0"]
    first_keypoint = cam0.frames[0].keypoints[0]
    assert first_keypoint.name == "nose"
    assert first_keypoint.confidence == pytest.approx(0.95)

    triangulated = observations.triangulated
    assert triangulated is not None
    assert triangulated.frames[0].keypoints[0].confidence == pytest.approx(0.925)


def test_pose2sim_default_detector_is_permissive_mediapipe() -> None:
    adapter = Pose2SimAdapter()

    assert adapter.detector == "mediapipe"


def test_pose2sim_openpose_is_explicit_opt_in() -> None:
    adapter = Pose2SimAdapter(detector=Pose2SimDetector.OPENPOSE)

    assert adapter.detector == "openpose"


def test_pose2sim_requires_multiple_camera_streams(tmp_path: Path) -> None:
    session = tmp_path / "single_camera"
    detections = session / "detections"
    detections.mkdir(parents=True)
    (session / "calibration.json").write_text(
        json.dumps(
            {
                "id": "single-camera",
                "source_fps": 30.0,
                "cameras": {
                    "cam0": {
                        "intrinsics": {"fx": 1000, "fy": 1000, "cx": 320, "cy": 240}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (detections / "cam0_mediapipe.json").write_text(
        json.dumps(
            {
                "schema": "MediaPipe_33",
                "fps": 30.0,
                "frames": [
                    {
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "landmarks": [{"x": 0.5, "y": 0.4, "visibility": 1.0}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least two camera"):
        load_pose2sim_observations(session)


def test_aggregate_confidence_empty_observations_returns_zero() -> None:
    """Missing evidence must return zero confidence, never 1.0 (Issue #9552)."""
    assert _aggregate_confidence({}, 0, "left_wrist", 0) == 0.0


def test_aggregate_confidence_rejects_missing_marker() -> None:
    """If cameras observe other markers but not the requested one, confidence is 0.0."""
    calib = Calibration(
        id="calib",
        source_fps=30.0,
        cameras={
            "cam0": {
                "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 320.0, "cy": 240.0}
            },
            "cam1": {
                "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 320.0, "cy": 240.0}
            },
        },
    )
    seq0 = KeypointSequence(
        id="seq0",
        calibration=calib,
        frames=[
            KeypointFrame(
                timestamp=0.0,
                keypoints=[Keypoint(name="nose", x=0.5, y=0.5, confidence=0.9)],
                schema_name="MediaPipe_33",
            )
        ],
    )
    seq1 = KeypointSequence(
        id="seq1",
        calibration=calib,
        frames=[
            KeypointFrame(
                timestamp=0.0,
                keypoints=[Keypoint(name="nose", x=0.5, y=0.5, confidence=0.85)],
                schema_name="MediaPipe_33",
            )
        ],
    )
    cameras = {"cam0": seq0, "cam1": seq1}
    assert _aggregate_confidence(cameras, 0, "left_wrist", 15, timestamp=0.0) == 0.0


def test_find_keypoint_rejects_positional_fallback_on_unmatched_name() -> None:
    """Positional fallback across incompatible schemas or names is rejected (Issue #9552)."""
    frame = KeypointFrame(
        timestamp=0.0,
        keypoints=[
            Keypoint(name="right_ankle", x=0.2, y=0.8, confidence=0.95),
            Keypoint(name="left_knee", x=0.3, y=0.7, confidence=0.88),
        ],
        schema_name="custom",
    )
    # Asking for "nose" at index 0 should NOT return "right_ankle"
    match = _find_keypoint(frame, "nose", 0)
    assert match is None


def test_find_keypoint_matches_reordered_keypoints_by_name() -> None:
    """Reordered keypoints are resolved by name, not by static index."""
    nose_kp = Keypoint(name="nose", x=0.5, y=0.4, confidence=0.92)
    frame = KeypointFrame(
        timestamp=0.0,
        keypoints=[
            Keypoint(name="left_shoulder", x=0.4, y=0.5, confidence=0.85),
            nose_kp,
        ],
        schema_name="MediaPipe_33",
    )
    # Query for "nose" should find nose_kp even if index passed is 0
    match = _find_keypoint(frame, "nose", 0)
    assert match is not None
    assert match.name == "nose"
    assert match.confidence == 0.92


def test_aggregate_confidence_respects_timestamp_tolerance() -> None:
    """Timestamp matching uses bounded tolerance; dropped frames do not poll wrong indices."""
    calib = Calibration(
        id="calib",
        source_fps=30.0,
        cameras={
            "cam0": {
                "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 320.0, "cy": 240.0}
            },
            "cam1": {
                "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 320.0, "cy": 240.0}
            },
        },
    )
    # cam0 has frames at t=0.0 and t=0.1 (dropped frame at t=0.033)
    seq0 = KeypointSequence(
        id="seq0",
        calibration=calib,
        frames=[
            KeypointFrame(
                timestamp=0.0,
                keypoints=[Keypoint(name="nose", x=0.5, y=0.5, confidence=0.9)],
                schema_name="MediaPipe_33",
            ),
            KeypointFrame(
                timestamp=0.1,
                keypoints=[Keypoint(name="nose", x=0.5, y=0.5, confidence=0.9)],
                schema_name="MediaPipe_33",
            ),
        ],
    )
    seq1 = KeypointSequence(
        id="seq1",
        calibration=calib,
        frames=[
            KeypointFrame(
                timestamp=0.033333,
                keypoints=[Keypoint(name="nose", x=0.5, y=0.5, confidence=0.9)],
                schema_name="MediaPipe_33",
            ),
        ],
    )
    cameras = {"cam0": seq0, "cam1": seq1}
    # At timestamp 0.033333, cam0 has no matching frame within tolerance (e.g. 0.016s)
    # Only 1 camera (cam1) has observation, which is insufficient for 3D reconstruction (< 2 views)
    conf = _aggregate_confidence(
        cameras,
        1,
        "nose",
        0,
        timestamp=0.033333,
        tolerance_s=0.016,
    )
    assert conf == 0.0


def test_pose2sim_rejects_duplicate_camera_stream_ids(tmp_path: Path) -> None:
    """Duplicate camera streams must raise ValueError instead of silently overwriting."""
    session = tmp_path / "dup_cameras"
    detections = session / "detections"
    detections.mkdir(parents=True)
    (session / "calibration.json").write_text(
        json.dumps(
            {
                "id": "dup-check",
                "source_fps": 30.0,
                "cameras": {
                    "cam0": {
                        "intrinsics": {"fx": 1000, "fy": 1000, "cx": 320, "cy": 240}
                    },
                    "cam1": {
                        "intrinsics": {"fx": 1000, "fy": 1000, "cx": 320, "cy": 240}
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    valid_payload = json.dumps(
        {
            "schema": "MediaPipe_33",
            "fps": 30.0,
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp": 0.0,
                    "landmarks": [
                        {"name": "nose", "x": 0.5, "y": 0.4, "visibility": 0.9}
                    ],
                }
            ],
        }
    )
    (detections / "cam0_mediapipe.json").write_text(valid_payload, encoding="utf-8")
    (detections / "cam1_mediapipe.json").write_text(valid_payload, encoding="utf-8")
    # Duplicate stream for cam0
    nested = detections / "nested"
    nested.mkdir()
    (nested / "cam0_mediapipe.json").write_text(valid_payload, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate camera"):
        load_pose2sim_observations(session)


def test_reconstruction_quality_preserves_per_view_confidences_and_view_count() -> None:
    """Reconstruction metadata preserves per-view detector confidence and view count."""
    observations = load_pose2sim_observations(FIXTURE_ROOT)

    quality_nose = observations.get_keypoint_quality(0, "nose")
    assert quality_nose is not None
    assert quality_nose.quality == "valid"
    assert quality_nose.reason is None
    assert quality_nose.contributing_views == 2
    assert quality_nose.view_confidences == {
        "cam0": pytest.approx(0.95),
        "cam1": pytest.approx(0.90),
    }
    assert quality_nose.reconstruction_confidence == pytest.approx(0.925)

    # Check that triangulated sequence metadata contains reconstruction quality
    triangulated = observations.triangulated
    assert triangulated is not None
    assert "reconstruction_quality" in triangulated.metadata


def test_pose2sim_to_canonical_observations() -> None:
    """Pose2Sim observations convert to canonical observations schema (#9422)."""
    observations = load_pose2sim_observations(FIXTURE_ROOT)
    canonical = observations.to_canonical_observations()

    assert canonical.id == f"pose2sim-{observations.calibration.id}"
    assert canonical.num_frames == 2
    assert canonical.marker_set_name == "Pose2Sim-OpenSim"
    assert "nose" in canonical.frames[0].markers
