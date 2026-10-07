"""Golf camera view presets and per-engine adapters (NV-1, #11674)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.golf_view_presets import (
    VIEW_ORDER,
    ViewPreset,
    drake_meshcat_camera_pose,
    get_view_preset,
    meshcat_camera,
    mujoco_camera_params,
    mujoco_fixed_camera,
    simbody_camera_transform,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

LOOKAT = (1.0, 0.0, 0.9)
# Spec frame: Z up, golfer faces -X, right-handed target line is -Y.
EXPECTED_DIRECTION = {
    "face_on": (1.0, 0.0),  # camera in front of the golfer looking along +X
    "down_the_line": (0.0, -1.0),  # camera behind on the target line, looks -Y
    "overhead": (0.0, 0.0),  # horizontal part vanishes (looking down)
    "oblique": (-1.0, 1.0),  # rear, target side: looks -X, +Y
}


def _horizontal_unit(vec: np.ndarray) -> np.ndarray:
    h = np.asarray(vec[:2], dtype=float)
    n = np.linalg.norm(h)
    return h / n if n > 1e-9 else h


def test_view_order_is_the_four_issue_views() -> None:
    assert VIEW_ORDER == ("face_on", "down_the_line", "overhead", "oblique")


@pytest.mark.parametrize("name", VIEW_ORDER)
def test_preset_view_direction_matches_spec_frame(name: str) -> None:
    d = get_view_preset(name).view_direction()
    assert np.linalg.norm(d) == pytest.approx(1.0)
    expected = np.asarray(EXPECTED_DIRECTION[name], dtype=float)
    if name != "overhead":
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(_horizontal_unit(d), expected, atol=1e-9)
    assert d[2] < 0.0, "every golf view looks slightly or fully downward"


def test_issue_table_angles() -> None:
    table = {
        "face_on": (0.0, -6.0),
        "down_the_line": (-90.0, -8.0),
        "overhead": (0.0, -89.0),
        "oblique": (135.0, -14.0),
    }
    for name, (az, el) in table.items():
        p = get_view_preset(name)
        assert (p.azimuth_deg, p.elevation_deg) == (az, el)


def test_face_on_target_is_image_right() -> None:
    p = get_view_preset("face_on")
    np.testing.assert_allclose(p.image_right(), (0.0, -1.0, 0.0), atol=1e-9)


def test_camera_position_is_behind_lookat_along_view() -> None:
    p = get_view_preset("down_the_line")
    pos = p.camera_position(LOOKAT, 3.0)
    np.testing.assert_allclose(
        pos - np.asarray(LOOKAT), -3.0 * p.view_direction(), atol=1e-12
    )
    assert pos[1] > 0.0, "down-the-line camera sits on the +Y (behind) side"


def test_unknown_preset_rejected() -> None:
    with pytest.raises(ValueError, match="unknown view preset"):
        get_view_preset("sideways")


def test_preset_validation() -> None:
    with pytest.raises(ValueError):
        ViewPreset("x", "X", 0.0, -120.0, 3.0)
    with pytest.raises(ValueError):
        ViewPreset("x", "X", 0.0, -10.0, 0.0)
    with pytest.raises(ValueError):
        get_view_preset("face_on").camera_position((0.0, 0.0), 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        get_view_preset("face_on").camera_position(LOOKAT, -1.0)


@pytest.mark.parametrize("name", VIEW_ORDER)
def test_mujoco_params_reproduce_view_direction(name: str) -> None:
    p = get_view_preset(name)
    cam = mujoco_camera_params(name, LOOKAT, 3.2)
    a, e = np.radians(cam.azimuth), np.radians(cam.elevation)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    np.testing.assert_allclose(fwd, p.view_direction(), atol=1e-9)
    assert cam.distance == 3.2
    assert cam.lookat == LOOKAT


@pytest.mark.parametrize("name", VIEW_ORDER)
def test_mujoco_fixed_camera_axes(name: str) -> None:
    p = get_view_preset(name)
    cam = mujoco_fixed_camera(name, LOOKAT, 3.0)
    right, up = np.asarray(cam.xyaxes[:3]), np.asarray(cam.xyaxes[3:])
    # camera z axis (backwards) = right x up; viewing direction is its negative
    view = -np.cross(right, up)
    np.testing.assert_allclose(view, p.view_direction(), atol=1e-9)
    assert up[2] > 0.0
    np.testing.assert_allclose(
        np.asarray(cam.position) + 3.0 * view, LOOKAT, atol=1e-9
    )


@pytest.mark.parametrize("name", VIEW_ORDER)
def test_drake_pose_is_world_frame_position_and_target(name: str) -> None:
    p = get_view_preset(name)
    pos, target = drake_meshcat_camera_pose(name, LOOKAT, 3.0)
    np.testing.assert_allclose(target, LOOKAT)
    np.testing.assert_allclose(
        (np.asarray(target) - np.asarray(pos)) / 3.0, p.view_direction(), atol=1e-9
    )


@pytest.mark.parametrize("name", VIEW_ORDER)
def test_meshcat_applies_rx_minus_90_scene_transform(name: str) -> None:
    p = get_view_preset(name)
    cam = meshcat_camera(name, LOOKAT, 3.0)
    rx = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=float)  # Rx(-90 deg)
    np.testing.assert_allclose(rx @ np.asarray(cam.position_world), cam.position_three)
    np.testing.assert_allclose(rx @ np.asarray(cam.target_world), cam.target_three)
    # node offset is the rotated camera-to-target vector
    off = np.asarray(cam.node_offset_three)
    np.testing.assert_allclose(off, rx @ (-3.0 * p.view_direction()), atol=1e-9)
    # Three.js is Y-up: an overhead camera must sit high on Y
    if name == "overhead":
        assert cam.position_three[1] > cam.target_three[1] + 2.9


@pytest.mark.parametrize("name", VIEW_ORDER)
def test_simbody_transform_looks_along_view_direction(name: str) -> None:
    p = get_view_preset(name)
    rot, pos = simbody_camera_transform(name, LOOKAT, 3.0)
    rot = np.asarray(rot)
    np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(rot) == pytest.approx(1.0)
    # simbody camera frame: -Z is the viewing direction, +Y is image up
    np.testing.assert_allclose(-rot[:, 2], p.view_direction(), atol=1e-9)
    assert rot[2, 1] > 0.0, "image up has a positive world-Z component"
    np.testing.assert_allclose(
        np.asarray(pos) + 3.0 * p.view_direction(), LOOKAT, atol=1e-9
    )
