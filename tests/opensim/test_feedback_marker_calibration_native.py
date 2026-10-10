"""Actual OpenSim FK round-trip for provisional marker calibration."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.engines.feedback_comparison import DriveMode
from src.engines.feedback_marker_calibration import (
    NativeMarkerCalibrationRequest,
    calibrate_static_marker_attachments,
)
from src.engines.feedback_native_execution import NativeAdapterBinding
from src.engines.physics_engines.opensim.python.tour_matching.native_marker_geometry import (
    NativeMarkerGeometry,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit


@pytest.fixture
def native_pin(tmp_path: Path) -> Path:
    osim = pytest.importorskip("opensim")
    model = osim.Model()
    body = osim.Body("segment", 1.0, osim.Vec3(0), osim.Inertia(0.02))
    model.addBody(body)
    joint = osim.PinJoint(
        "pin",
        model.getGround(),
        osim.Vec3(1, 0, 0),
        osim.Vec3(0),
        body,
        osim.Vec3(0),
        osim.Vec3(0),
    )
    joint.updCoordinate().setName("angle")
    joint.updCoordinate().setRangeMin(-1.5)
    joint.updCoordinate().setRangeMax(1.5)
    model.addJoint(joint)
    model.finalizeConnections()
    path = tmp_path / "pin.osim"
    model.printToXML(str(path))
    return path


def test_calibration_recovers_offsets_from_native_frame_poses(
    native_pin: Path,
) -> None:
    provider = NativeMarkerGeometry(native_pin, ("/jointset/pin/angle",))
    labels = ("marker-a", "marker-b")
    frame_id = "/bodyset/segment"
    frame_ids = dict.fromkeys(labels, frame_id)
    known_offsets = (
        np.array([0.08, -0.03, 0.04]),
        np.array([-0.02, 0.06, 0.11]),
    )
    times = np.array([0.0, 0.01, 0.02])
    coordinates = (np.array([0.1]), np.array([0.4]), np.array([-0.2]))
    pose_bindings = dict.fromkeys(labels, (frame_id, (0.0, 0.0, 0.0)))
    poses = tuple(provider.frame_poses(pose_bindings, q) for q in coordinates)
    points = np.empty((len(times), len(labels), 3))
    for sample, pose_set in enumerate(poses):
        rotation, translation = pose_set[frame_id]
        for marker, offset in enumerate(known_offsets):
            points[sample, marker] = rotation @ offset + translation
    capture = TourCapture(
        times,
        labels,
        points,
        np.ones((len(times), len(labels)), dtype=bool),
        hashlib.sha256(b"synthetic native pose calibration fixture").hexdigest(),
    )
    binding = NativeAdapterBinding(
        "opensim/fixture",
        "default",
        DriveMode.TORQUE,
        "opensim-calibration-fixture",
        "default",
        "opensim-native-replay-fixture",
        "a" * 64,
        provider.source_sha256,
        provider.loaded_sha256,
        "b" * 64,
        "c" * 64,
        ("fixture-torque",),
    )

    artifact = calibrate_static_marker_attachments(
        NativeMarkerCalibrationRequest(
            capture,
            frame_ids,
            poses,
            times,
            binding,
            "opensim",
            "opensim-native-marker-geometry",
            provider.provider_sha256,
            "native-ground",
            "capture-relative",
        )
    )

    np.testing.assert_allclose(
        [item.local_position_m for item in artifact.attachments],
        known_offsets,
        atol=1e-12,
    )
    artifact.validate(
        NativeMarkerCalibrationRequest(
            capture,
            frame_ids,
            poses,
            times,
            binding,
            "opensim",
            "opensim-native-marker-geometry",
            provider.provider_sha256,
            "native-ground",
            "capture-relative",
        )
    )
    assert artifact.source_model_sha256 == provider.source_sha256
    assert artifact.loaded_native_model_sha256 == provider.loaded_sha256
    assert artifact.pose_provider_sha256 == provider.provider_sha256
    assert artifact.calibration_fit_rms_m < 1e-12
