"""Projected framing of the golf view presets (NV-9, #11697)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from src.shared.python.golf_view_presets import (
    DEFAULT_FRAME_MARGIN,
    VIEW_ORDER,
    VIEWER_FOV_Y_RAD,
    bounding_box_fill_fraction,
    drake_meshcat_camera_pose,
    fit_distance_m,
    get_view_preset,
    golfer_bounding_box,
    meshcat_camera,
    mujoco_camera_params,
    projected_extent,
    simbody_camera_transform,
)

pytestmark = pytest.mark.unit

LOOKAT = (1.0, 0.0, 0.9)
ASPECT = 1280 / 720
#: three.js PerspectiveCamera default, which MeshCat used before NV-9.
THREE_JS_DEFAULT_FOV_Y_RAD = math.radians(75.0)


def _golfer_box() -> np.ndarray:
    """Corners of a standing golfer with a club: 0.8 x 1.4 x 1.85 m on the ground."""
    lo = np.array([LOOKAT[0] - 0.4, LOOKAT[1] - 0.7, 0.0])
    hi = np.array([LOOKAT[0] + 0.4, LOOKAT[1] + 0.7, 1.85])
    return np.array(list(itertools.product(*zip(lo, hi, strict=True))))


def test_shared_fov_is_the_opensim_viewer_fov() -> None:
    assert pytest.approx(0.7) == VIEWER_FOV_Y_RAD


def test_extent_of_the_lookat_is_zero_and_edge_point_is_one() -> None:
    preset = get_view_preset("face_on")
    distance = 3.2
    half_height = distance * math.tan(VIEWER_FOV_Y_RAD / 2.0)
    edge = np.asarray(LOOKAT) + half_height * preset.image_up()

    assert projected_extent(preset, [LOOKAT], LOOKAT, distance) == pytest.approx(0.0)
    assert projected_extent(preset, [edge], LOOKAT, distance) == pytest.approx(1.0)


def test_extent_uses_the_aspect_ratio_for_horizontal_points() -> None:
    preset = get_view_preset("face_on")
    half_width = 3.2 * math.tan(VIEWER_FOV_Y_RAD / 2.0) * ASPECT
    edge = np.asarray(LOOKAT) + half_width * preset.image_right()

    extent = projected_extent(preset, [edge], LOOKAT, 3.2, aspect=ASPECT)

    assert extent == pytest.approx(1.0)


def test_three_js_default_fov_leaves_the_golfer_small() -> None:
    """Reproduces the NV-9 defect: the 75 deg MeshCat default shrinks the golfer."""
    preset = get_view_preset("face_on")
    box = _golfer_box()

    wide = projected_extent(
        preset, box, LOOKAT, 3.2, fov_y_rad=THREE_JS_DEFAULT_FOV_Y_RAD, aspect=ASPECT
    )
    shared = projected_extent(preset, box, LOOKAT, 3.2, aspect=ASPECT)

    assert wide < 0.5
    assert shared > 1.8 * wide


@pytest.mark.parametrize("view", VIEW_ORDER)
def test_fitted_distance_frames_the_golfer_within_the_margin(view: str) -> None:
    preset = get_view_preset(view)
    box = _golfer_box()

    distance = fit_distance_m(preset, box, LOOKAT, aspect=ASPECT)
    extent = projected_extent(preset, box, LOOKAT, distance, aspect=ASPECT)

    assert pytest.approx(0.15) == DEFAULT_FRAME_MARGIN
    assert 1.0 - DEFAULT_FRAME_MARGIN - 1e-6 <= extent <= 1.0
    assert extent == pytest.approx(1.0 - DEFAULT_FRAME_MARGIN, abs=1e-4)


def test_fit_rejects_bad_inputs() -> None:
    preset = get_view_preset("face_on")
    box = _golfer_box()
    with pytest.raises(ValueError, match="margin"):
        fit_distance_m(preset, box, LOOKAT, margin=1.0)
    with pytest.raises(ValueError, match="points_m"):
        fit_distance_m(preset, np.zeros((0, 3)), LOOKAT)
    with pytest.raises(ValueError, match="fov_y_rad"):
        projected_extent(preset, box, LOOKAT, 3.2, fov_y_rad=0.0)
    with pytest.raises(ValueError, match="aspect"):
        projected_extent(preset, box, LOOKAT, 3.2, aspect=-1.0)


def test_extent_rejects_points_behind_the_camera() -> None:
    preset = get_view_preset("face_on")
    behind = np.asarray(LOOKAT) - 5.0 * preset.view_direction()
    with pytest.raises(ValueError, match="in front of the camera"):
        projected_extent(preset, [behind], LOOKAT, 3.2)


def test_golfer_bounding_box_matches_standard_dimensions() -> None:
    box = golfer_bounding_box(LOOKAT)
    assert box.shape == (8, 3)
    np.testing.assert_allclose(box.min(axis=0), (0.6, -0.7, 0.0))
    np.testing.assert_allclose(box.max(axis=0), (1.4, 0.7, 1.85))


@pytest.mark.parametrize("view", VIEW_ORDER)
def test_each_camera_preset_fits_golfer_bounding_box_within_margin(view: str) -> None:
    preset = get_view_preset(view)
    fill = bounding_box_fill_fraction(preset, aspect=ASPECT)
    margin = 1.0 - fill
    assert 0.85 <= fill <= 1.0
    assert margin <= DEFAULT_FRAME_MARGIN


@pytest.mark.parametrize("view", VIEW_ORDER)
def test_every_backend_fits_golfer_bounding_box_within_margin(view: str) -> None:
    # Drake MeshCat
    eye, target = drake_meshcat_camera_pose(view, LOOKAT)
    dist_drake = float(np.linalg.norm(np.asarray(eye) - np.asarray(target)))
    fill_drake = bounding_box_fill_fraction(view, distance_m=dist_drake, aspect=ASPECT)
    assert 0.85 <= fill_drake <= 1.0
    assert (1.0 - fill_drake) <= DEFAULT_FRAME_MARGIN

    # Pinocchio MeshCat
    cam = meshcat_camera(view, LOOKAT)
    dist_pin = float(
        np.linalg.norm(np.asarray(cam.position_world) - np.asarray(cam.target_world))
    )
    fill_pin = bounding_box_fill_fraction(view, distance_m=dist_pin, aspect=ASPECT)
    assert 0.85 <= fill_pin <= 1.0
    assert (1.0 - fill_pin) <= DEFAULT_FRAME_MARGIN

    # OpenSim Simbody
    _rows, pos = simbody_camera_transform(view, LOOKAT)
    dist_osim = float(np.linalg.norm(np.asarray(pos) - np.asarray(LOOKAT)))
    fill_osim = bounding_box_fill_fraction(view, distance_m=dist_osim, aspect=ASPECT)
    assert 0.85 <= fill_osim <= 1.0
    assert (1.0 - fill_osim) <= DEFAULT_FRAME_MARGIN

    # MyoSuite MJRenderer
    m_cam = mujoco_camera_params(view, LOOKAT)
    fill_myo = bounding_box_fill_fraction(
        view, distance_m=m_cam.distance, aspect=ASPECT
    )
    assert 0.85 <= fill_myo <= 1.0
    assert (1.0 - fill_myo) <= DEFAULT_FRAME_MARGIN
