"""Bind actual camera, geometry and clock evidence without claiming accuracy."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reference.evidence import CameraSnapshot, ViewClock
from src.motion_capture.reference.registration import ReferenceRegistration
from src.motion_capture.reference.scene import session_camera, session_clock
from tests.motion_capture.test_reference_registration import two_camera_rig
from tests.motion_capture.test_reference_timing_qualification import motion

pytestmark = pytest.mark.unit


def test_binding_detects_changed_camera_and_geometry_and_ignores_notes() -> None:
    asset = motion((0.0, 0.1))
    camera = CameraSnapshot.from_calibration(
        two_camera_rig()[0].to_calibration(), provenance="Fixture"
    )
    clock = ViewClock(view=camera.camera_id)
    reg = ReferenceRegistration(
        reference_id=asset.id, calibration_id="legacy", is_calibrated=True
    )
    bound = reg.bound(asset, camera, clock)
    assert not bound.is_calibrated
    bound.validate_binding(asset.changed(notes="Instructor note"), camera, clock)
    changed = asset.changed(points_m=(((9, 0, 0),), ((1, 0, 0),)))
    with pytest.raises(ValueError, match="geometry"):
        bound.validate_binding(changed, camera, clock)
    other = CameraSnapshot.model_validate(
        camera.model_dump() | {"distortion": [0.1, 0, 0, 0, 0]}
    )
    with pytest.raises(ValueError, match="camera"):
        bound.validate_binding(asset, other, clock)
    with pytest.raises(ValueError, match="clock"):
        bound.validate_binding(
            asset,
            camera,
            ViewClock(view=camera.camera_id, offset_ns=1000, source="manual"),
        )
    restored = ReferenceRegistration.model_validate_json(bound.model_dump_json())
    assert restored == bound


def test_session_preserves_measured_lens_and_rejects_conflicting_dimensions(
    tmp_path: Path,
) -> None:
    record = two_camera_rig()[0].to_calibration()
    directory = tmp_path / "reconstruct"
    directory.mkdir()
    (directory / "reconstruction.json").write_text(
        json.dumps({"cameras": [record.to_dict()]}), encoding="utf-8"
    )
    assert (
        "no lens evidence" in session_camera(tmp_path, "", record.camera_id).provenance
    )
    lens = {
        "camera_id": record.camera_id,
        "matrix": record.intrinsics.matrix.tolist(),
        "distortion": [0.1, 0, 0, 0, 0],
        "image_size_px": record.image_size_px,
        "rms_px": 0.2,
        "frames_used": 30,
        "frames_without_board": 2,
        "board": {},
    }
    path = tmp_path / "intrinsics.json"
    path.write_text(json.dumps([lens]), encoding="utf-8")
    snapshot = session_camera(tmp_path, "", record.camera_id)
    np.testing.assert_allclose(
        snapshot.record().intrinsics.distortion, lens["distortion"]
    )
    lens["image_size_px"] = [320, 240]
    path.write_text(json.dumps([lens]), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        session_camera(tmp_path, "", record.camera_id)


def test_clock_keeps_nominal_assumption_distinct_from_recorded_offset() -> None:
    nominal = session_clock({}, "a")
    assert nominal.source == "nominal-frame-rate-unverified"
    assert nominal.player_time(2) == 2
    timing = {
        "method": "strobe",
        "views": [
            {
                "view": "a",
                "status": "available",
                "offset_ns": 1_000_000_000,
                "uncertainty_ns": 1000,
            }
        ],
    }
    clock = session_clock(timing, "a")
    assert clock.source == "strobe"
    assert clock.player_time(2) == 1


def test_reconstruction_distortion_survives_shared_camera_selection(
    tmp_path: Path,
) -> None:
    record = two_camera_rig()[0].to_calibration().to_dict()
    record["intrinsics"]["distortion"] = [0.12, 0, 0, 0, 0]
    directory = tmp_path / "reconstruct"
    directory.mkdir()
    (directory / "reconstruction.json").write_text(
        json.dumps({"cameras": [record]}), encoding="utf-8"
    )
    snapshot = session_camera(tmp_path, "", record["camera_id"])
    assert snapshot.distortion == (0.12, 0, 0, 0, 0)


def test_two_camera_pixels_match_independent_pinhole_projection() -> None:
    from src.motion_capture.reference.registration import project_reference_to_camera

    points = np.array([[[0.2, 1.1, 0.3], [0.4, 1.4, 0.2]]])
    for camera in two_camera_rig():
        expected, front = camera.project(points[0])
        actual, visible = project_reference_to_camera(
            points, np.ones((1, 2), dtype=bool), camera.to_calibration()
        )
        np.testing.assert_allclose(actual[0], expected, atol=1e-8)
        assert front.all() and visible.all()
