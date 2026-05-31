"""Tests for Pose2Sim multi-camera ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources.pose2sim_adapter import (
    Pose2SimAdapter,
    Pose2SimDetector,
    load_pose2sim_observations,
)


FIXTURE_ROOT = Path("tests/data/motion_pipeline/pose2sim/sample_session")


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
