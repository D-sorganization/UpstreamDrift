"""Pinhole projection, look-at, and the bridge to ADR-0041 calibration records."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct.cameras import (
    PinholeCamera,
    intrinsics_from_fov,
    look_at,
)

pytestmark = pytest.mark.unit


def _camera(position=(0.0, 1.0, 3.0), target=(0.0, 1.0, 0.0)) -> PinholeCamera:
    return PinholeCamera(
        camera_id="cam",
        matrix=intrinsics_from_fov(1280, 720, 60.0),
        rotation_world_from_camera=look_at(np.array(position), np.array(target)),
        translation_world_from_camera_m=np.array(position),
        image_size_px=(1280, 720),
    )


def test_look_at_is_a_rotation_with_forward_toward_target() -> None:
    r = look_at(np.array([0.0, 1.0, 3.0]), np.array([0.0, 1.0, 0.0]))
    assert np.allclose(r.T @ r, np.eye(3)) and np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(r[:, 2], [0.0, 0.0, -1.0])  # camera +z points at target
    assert r[1, 1] < 0  # camera +y (down) maps to world -y
    with pytest.raises(Exception, match="parallel"):
        look_at(np.zeros(3), np.array([0.0, 1.0, 0.0]))


def test_projection_puts_the_target_at_the_principal_point() -> None:
    cam = _camera()
    px, in_front = cam.project(np.array([[0.0, 1.0, 0.0]]))
    assert in_front.all() and np.allclose(px[0], [640.0, 360.0])
    behind, mask = cam.project(np.array([[0.0, 1.0, 5.0]]))
    assert not mask[0] and np.isnan(behind[0]).all()
    assert cam.in_image(
        np.array([[10.0, 10.0], [-1.0, 5.0], [np.nan, np.nan]])
    ).tolist() == [
        True,
        False,
        False,
    ]


def test_world_axes_project_in_the_expected_image_directions() -> None:
    cam = _camera()
    px, _ = cam.project(np.array([[0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [0.0, 1.5, 0.0]]))
    # camera on +z looking toward -z: world +x is image-right (right = fwd x up)
    assert px[1, 0] > px[0, 0]
    assert px[2, 1] < px[0, 1]  # +y world (up) is image-up


def test_calibration_record_round_trip() -> None:
    cam = _camera()
    record = cam.to_calibration()
    assert record.camera_id == "cam" and record.image_size_px == (1280, 720)
    back = PinholeCamera.from_calibration(record)
    assert np.allclose(back.rotation_world_from_camera, cam.rotation_world_from_camera)
    assert np.allclose(back.position_m, cam.position_m)


def test_contracts_reject_bad_rotation_and_fov() -> None:
    with pytest.raises(Exception, match="orthonormal"):
        PinholeCamera("c", np.eye(3), np.ones((3, 3)), np.zeros(3), (10, 10))
    with pytest.raises(Exception, match="fov"):
        intrinsics_from_fov(10, 10, 180.0)
