"""Tests for canonical markerless observation records."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.pose_estimation.observations import (
    CANONICAL_OBSERVATIONS_SCHEMA_VERSION,
    TRACE_META_OBSERVATIONS_JSON,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    CanonicalObservations,
    DetectorLayout,
    KeypointObservation,
)
from src.shared.python.simulation_backends.trace_io import read_trace, write_trace

pytestmark = pytest.mark.unit


def _layout() -> DetectorLayout:
    return DetectorLayout(name="coco-3", keypoint_names=("pelvis", "neck", "head"))


def _camera(camera_id: str) -> CameraCalibration:
    return CameraCalibration(
        camera_id=camera_id,
        intrinsics=CameraIntrinsics(
            matrix=np.array(
                [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]]
            ),
            distortion=np.array([0.01, -0.02, 0.0, 0.0, 0.0]),
        ),
        extrinsics=CameraExtrinsics(
            rotation_world_from_camera=np.eye(3),
            translation_world_from_camera_m=np.array([1.0, 0.0, 1.5]),
        ),
        image_size_px=(1280, 720),
    )


def _frame(camera_id: str, time_s: float = 0.0) -> KeypointObservation:
    return KeypointObservation(
        camera_id=camera_id,
        time_s=time_s,
        keypoints_px=np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
        confidence=np.array([0.95, 0.5, 0.0]),
    )


def _observations() -> CanonicalObservations:
    return CanonicalObservations(
        detector_layout=_layout(),
        cameras=(_camera("cam-a"), _camera("cam-b")),
        frames=(_frame("cam-a"), _frame("cam-b")),
        keypoints_3d_m=np.array([[[0.0, 0.0, 1.0], [0.1, 0.0, 1.2], [0.2, 0.0, 1.4]]]),
        keypoints_3d_confidence=np.array([[0.9, 0.6, 0.1]]),
        provenance={"source": "unit-test", "trial_id": "swing-001"},
    )


def test_canonical_observations_round_trip_preserves_confidence_and_calibration(
    tmp_path: Path,
) -> None:
    obs = _observations()
    path = tmp_path / "observations.json"

    obs.to_path(path)
    restored = CanonicalObservations.from_path(path)

    assert restored.schema_version == CANONICAL_OBSERVATIONS_SCHEMA_VERSION
    assert restored.detector_layout.keypoint_names == ("pelvis", "neck", "head")
    np.testing.assert_allclose(restored.frames[0].confidence, [0.95, 0.5, 0.0])
    np.testing.assert_allclose(
        restored.camera("cam-a").intrinsics.matrix,
        [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
    )
    np.testing.assert_allclose(restored.keypoints_3d_confidence, [[0.9, 0.6, 0.1]])
    assert restored.provenance["trial_id"] == "swing-001"


def test_sample_multi_camera_fixture_loads_and_validates() -> None:
    fixture = (
        Path(__file__).parents[2]
        / "fixtures"
        / "pose_estimation"
        / "canonical_observations_multi_camera.json"
    )

    obs = CanonicalObservations.from_path(fixture)

    assert len(obs.cameras) == 2
    assert len(obs.frames_for_camera("front")) == 2
    assert obs.detector_layout.name == "coco-3"


def test_keypoint_count_must_match_detector_layout() -> None:
    with pytest.raises(ValueError, match="keypoint count"):
        CanonicalObservations(
            detector_layout=_layout(),
            cameras=(_camera("cam-a"),),
            frames=(
                KeypointObservation(
                    camera_id="cam-a",
                    time_s=0.0,
                    keypoints_px=np.array([[10.0, 20.0], [30.0, 40.0]]),
                    confidence=np.array([0.5, 0.5]),
                ),
            ),
        )


def test_confidence_must_be_in_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        KeypointObservation(
            camera_id="cam-a",
            time_s=0.0,
            keypoints_px=np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
            confidence=np.array([0.2, 1.2, 0.5]),
        )


def test_camera_matrices_are_validated() -> None:
    with pytest.raises(ValueError, match="intrinsics.matrix"):
        CameraIntrinsics(matrix=np.eye(4))

    with pytest.raises(ValueError, match="rotation_world_from_camera"):
        CameraExtrinsics(
            rotation_world_from_camera=np.ones((3, 3)),
            translation_world_from_camera_m=np.zeros(3),
        )


def test_payload_rejects_unknown_camera_reference() -> None:
    with pytest.raises(ValueError, match="unknown camera_id"):
        CanonicalObservations(
            detector_layout=_layout(),
            cameras=(_camera("cam-a"),),
            frames=(_frame("missing-camera"),),
        )


def test_to_dict_is_plain_json_serializable() -> None:
    encoded = json.dumps(_observations().to_dict())
    decoded = json.loads(encoded)

    restored = CanonicalObservations.from_dict(decoded)

    np.testing.assert_allclose(
        restored.frames[1].keypoints_px, _frame("cam-b").keypoints_px
    )


def test_cc4_trace_writer_round_trip_preserves_observations(tmp_path: Path) -> None:
    obs = _observations()
    trace_path = tmp_path / "observations_trace.h5"

    write_trace(obs.to_trace(), trace_path)
    loaded_trace = read_trace(trace_path)
    restored = CanonicalObservations.from_trace(loaded_trace)

    assert TRACE_META_OBSERVATIONS_JSON in loaded_trace.meta
    np.testing.assert_allclose(loaded_trace.markers, obs.keypoints_3d_m)
    np.testing.assert_allclose(restored.frames[0].confidence, obs.frames[0].confidence)
    np.testing.assert_allclose(
        restored.camera("cam-b").extrinsics.translation_world_from_camera_m,
        obs.camera("cam-b").extrinsics.translation_world_from_camera_m,
    )
