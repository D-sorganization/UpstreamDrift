"""Source-bound, provisional native marker attachment calibration."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.engines.feedback_comparison import DriveMode
from src.engines.feedback_native_execution import NativeAdapterBinding
from src.engines.feedback_marker_calibration import (
    NativeMarkerCalibrationRequest,
    calibrate_static_marker_attachments,
    capture_observation_sha256,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit


def _binding() -> NativeAdapterBinding:
    return NativeAdapterBinding(
        "mujoco/fixture",
        "default",
        DriveMode.TORQUE,
        "mujoco-native-fixture",
        "fixture-variant",
        "mujoco-native-replay",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        ("joint-torque",),
    )


def _capture_and_poses() -> tuple[
    TourCapture, tuple[dict[str, tuple[np.ndarray, np.ndarray]], ...]
]:
    times = np.array([0.0, 0.01, 0.02])
    labels = ("marker-a", "marker-b")
    frames = ("/bodyset/arm", "/bodyset/hand")
    offsets = (np.array([0.03, -0.02, 0.11]), np.array([-0.04, 0.01, 0.07]))
    poses = tuple(
        {
            frames[0]: (
                np.eye(3),
                np.array([float(i), 0.0, 0.2 * i]),
            ),
            frames[1]: (
                np.eye(3),
                np.array([float(i), 0.3, 0.1 * i]),
            ),
        }
        for i in range(len(times))
    )
    points = np.empty((len(times), len(labels), 3))
    for frame_index, frame_poses in enumerate(poses):
        for marker_index, frame_id in enumerate(frames):
            rotation, translation = frame_poses[frame_id]
            points[frame_index, marker_index] = (
                rotation @ offsets[marker_index] + translation
            )
    capture = TourCapture(
        times,
        labels,
        points,
        np.ones((len(times), len(labels)), dtype=bool),
        "f" * 64,
    )
    return capture, poses


def _calibrate(capture: TourCapture, poses):
    return calibrate_static_marker_attachments(
        NativeMarkerCalibrationRequest(
            capture,
            {"marker-a": "/bodyset/arm", "marker-b": "/bodyset/hand"},
            poses,
            capture.time_s,
            _binding(),
            "mujoco",
            "mujoco-native-fk-fixture",
            "1" * 64,
            "capture-world",
            "capture-relative",
        )
    )


def _validate(artifact, capture, poses, binding=None):
    artifact.validate(
        NativeMarkerCalibrationRequest(
            capture,
            {"marker-a": "/bodyset/arm", "marker-b": "/bodyset/hand"},
            poses,
            capture.time_s,
            binding or _binding(),
            "mujoco",
            "mujoco-native-fk-fixture",
            "1" * 64,
            "capture-world",
            "capture-relative",
        )
    )


def test_static_calibration_emits_ordered_provisional_source_bound_artifact() -> None:
    capture, poses = _capture_and_poses()

    artifact = _calibrate(capture, poses)

    assert artifact.schema_version == "native-marker-calibration/1.0.0"
    assert artifact.calibration_status == "provisional_estimated"
    assert artifact.qualification == "unqualified"
    assert artifact.holdout_status == "not_evaluated"
    assert artifact.physiology_status == "unqualified"
    assert tuple(item.label for item in artifact.attachments) == capture.labels
    np.testing.assert_allclose(
        [item.local_position_m for item in artifact.attachments],
        [[0.03, -0.02, 0.11], [-0.04, 0.01, 0.07]],
        atol=1e-14,
    )
    assert artifact.capture_observations_sha256
    assert artifact.pose_trajectory_sha256

    marker_map = artifact.to_native_marker_map(
        _binding(), output_timebase_id="simulation-relative"
    )
    assert marker_map.calibration_artifact_sha256 == artifact.sha256
    marker_map.validate(_binding(), "simulation-relative")
    assert marker_map.sha256
    with pytest.raises(ValueError, match="timebase differs"):
        marker_map.validate(_binding(), "capture-relative")


def test_artifact_rejects_capture_mutation_even_when_source_id_is_reused() -> None:
    capture, poses = _capture_and_poses()
    artifact = _calibrate(capture, poses)
    changed_points = capture.points_m.copy()
    changed_points[1, 0, 0] += 1e-5
    changed = TourCapture(
        capture.time_s,
        capture.labels,
        changed_points,
        capture.valid,
        capture.source_sha256,
    )

    with pytest.raises(ValueError, match="capture observations differ"):
        _validate(artifact, changed, poses)


def test_artifact_rejects_provider_and_pose_clock_changes() -> None:
    capture, poses = _capture_and_poses()
    artifact = _calibrate(capture, poses)

    with pytest.raises(ValueError, match="binding identity differs"):
        _validate(
            artifact,
            capture,
            poses,
            replace(_binding(), native_execution_provider_id="other-provider"),
        )
    with pytest.raises(ValueError, match="pose clock differs"):
        calibrate_static_marker_attachments(
            NativeMarkerCalibrationRequest(
                capture,
                {"marker-a": "/bodyset/arm", "marker-b": "/bodyset/hand"},
                poses,
                np.array([0.0, 0.011, 0.02]),
                _binding(),
                "mujoco",
                "mujoco-native-fk-fixture",
                "1" * 64,
                "capture-world",
                "capture-relative",
            )
        )


def test_artifact_rejects_pose_history_mutation() -> None:
    capture, poses = _capture_and_poses()
    artifact = _calibrate(capture, poses)
    changed_pose = list(poses)
    changed_pose[1] = dict(changed_pose[1])
    rotation, translation = changed_pose[1]["/bodyset/arm"]
    changed_pose[1]["/bodyset/arm"] = (
        rotation,
        translation + np.array([1e-6, 0.0, 0.0]),
    )

    with pytest.raises(ValueError, match="pose trajectory differs"):
        _validate(artifact, capture, changed_pose)


def test_artifact_rejects_changed_native_frame_mapping() -> None:
    capture, poses = _capture_and_poses()
    artifact = _calibrate(capture, poses)
    request = NativeMarkerCalibrationRequest(
        capture,
        {"marker-a": "/bodyset/hand", "marker-b": "/bodyset/arm"},
        poses,
        capture.time_s,
        _binding(),
        "mujoco",
        "mujoco-native-fk-fixture",
        "1" * 64,
        "capture-world",
        "capture-relative",
    )

    with pytest.raises(ValueError, match="native frame mapping differs"):
        artifact.validate(request)


def test_artifact_rejects_changed_native_engine_identity() -> None:
    capture, poses = _capture_and_poses()
    artifact = _calibrate(capture, poses)
    request = NativeMarkerCalibrationRequest(
        capture,
        {"marker-a": "/bodyset/arm", "marker-b": "/bodyset/hand"},
        poses,
        capture.time_s,
        _binding(),
        "opensim",
        "mujoco-native-fk-fixture",
        "1" * 64,
        "capture-world",
        "capture-relative",
    )

    with pytest.raises(ValueError, match="native engine identity differs"):
        artifact.validate(request)


@pytest.mark.parametrize("mutate_after_calibration", [False, True])
def test_calibration_revalidates_aliased_capture_clock(
    mutate_after_calibration: bool,
) -> None:
    owner_time = np.array([0.0, 0.01, 0.02])
    capture, poses = _capture_and_poses()
    aliased_capture = TourCapture(
        owner_time.view(),
        capture.labels,
        capture.points_m,
        capture.valid,
        capture.source_sha256,
    )
    request = NativeMarkerCalibrationRequest(
        aliased_capture,
        {"marker-a": "/bodyset/arm", "marker-b": "/bodyset/hand"},
        poses,
        aliased_capture.time_s,
        _binding(),
        "mujoco",
        "mujoco-native-fk-fixture",
        "1" * 64,
        "capture-world",
        "capture-relative",
    )
    artifact = (
        calibrate_static_marker_attachments(request)
        if mutate_after_calibration
        else None
    )
    owner_time[1] = 0.03

    with pytest.raises(
        ValueError, match="Capture time must start at zero and increase strictly"
    ):
        if mutate_after_calibration:
            assert artifact is not None
            artifact.validate(request)
        else:
            calibrate_static_marker_attachments(request)


def test_artifact_recomputes_offsets_instead_of_trusting_receipt_fields() -> None:
    capture, poses = _capture_and_poses()
    artifact = _calibrate(capture, poses)
    changed = replace(
        artifact,
        attachments=(
            replace(
                artifact.attachments[0],
                local_position_m=(0.03001, -0.02, 0.11),
            ),
            artifact.attachments[1],
        ),
    )

    with pytest.raises(ValueError, match="offsets differ from frozen inputs"):
        _validate(changed, capture, poses)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pose_provider_id", "other-provider", "pose provider identity differs"),
        ("pose_provider_sha256", "2" * 64, "pose provider identity differs"),
        ("capture_frame_id", "other-world", "capture coordinates differ"),
        ("capture_timebase_id", "other-clock", "capture coordinates differ"),
    ],
)
def test_revalidation_rejects_changed_calibration_coordinate_identity(
    field: str, value: str, message: str
) -> None:
    capture, poses = _capture_and_poses()
    artifact = _calibrate(capture, poses)
    expected = {
        "pose_provider_id": "mujoco-native-fk-fixture",
        "pose_provider_sha256": "1" * 64,
        "capture_frame_id": "capture-world",
        "capture_timebase_id": "capture-relative",
    }
    if field in expected:
        expected[field] = value

    with pytest.raises(ValueError, match=message):
        artifact.validate(
            NativeMarkerCalibrationRequest(
                capture,
                {"marker-a": "/bodyset/arm", "marker-b": "/bodyset/hand"},
                poses,
                capture.time_s,
                _binding(),
                "mujoco",
                expected["pose_provider_id"],
                expected["pose_provider_sha256"],
                expected["capture_frame_id"],
                expected["capture_timebase_id"],
            )
        )


def test_artifact_serialization_omits_observation_arrays() -> None:
    capture, poses = _capture_and_poses()
    artifact = _calibrate(capture, poses)

    serialized = str(artifact.as_dict())

    assert capture_observation_sha256(capture) in serialized
    assert "points_m" not in serialized
    assert "time_s" not in serialized
    assert capture.source_sha256 in serialized


def test_artifact_map_uses_exact_ordered_native_frames() -> None:
    capture, poses = _capture_and_poses()
    artifact = _calibrate(capture, poses)

    attachments = artifact.to_native_marker_map(
        _binding(), output_timebase_id="simulation-relative"
    ).attachments

    assert tuple(item.frame_id for item in attachments) == (
        "/bodyset/arm",
        "/bodyset/hand",
    )


def test_calibrated_map_rejects_malformed_artifact_digest() -> None:
    capture, poses = _capture_and_poses()
    marker_map = _calibrate(capture, poses).to_native_marker_map(
        _binding(), output_timebase_id="simulation-relative"
    )

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(marker_map, calibration_artifact_sha256="stale").validate(
            _binding(), "simulation-relative"
        )
