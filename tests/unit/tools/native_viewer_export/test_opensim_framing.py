"""OpenSim face-on framing keeps both feet and the clubhead in view (#12308).

The simbody window is 4:3 and the clip is 16:9, so the legacy centre-crop cut
the vertical field of view and put the feet below the image edge. The framing
now crops about the camera's principal point, sets the simbody field of view so
the visible crop matches the shared viewer lens, and fits the camera distance
to the trajectory's foot, head and clubhead points.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.golf_view_presets import VIEWER_FOV_Y_RAD, get_view_preset
from src.tools.native_viewer_export.backends import _opensim_framing as framing
from src.tools.native_viewer_export.overlay2d import pinhole_for_view
from tests.helpers.opensim_support import require_opensim

pytestmark = pytest.mark.unit
SPECS = Path(__file__).resolve().parents[4] / "docs/development/full_body_models"
LOOK = (1.0, 0.0, 0.9)
SIZE = (1280, 720)
WINDOW_HW = (600, 800)  # the simbody visualizer window under xvfb
STRIP_PX = 40
MARGIN_PX = 0.05  # required empty border, as a fraction of each image side


def _inside_with_margin(px: np.ndarray, size: tuple[int, int]) -> bool:
    w, h = size
    lo = np.array([MARGIN_PX * w, MARGIN_PX * h])
    hi = np.array([(1 - MARGIN_PX) * w, (1 - MARGIN_PX) * h])
    return bool(np.isfinite(px).all() and (px >= lo).all() and (px <= hi).all())


def _output_px(points, view, look, distance, size=SIZE):
    cam = pinhole_for_view(view, look, distance, VIEWER_FOV_Y_RAD, size)
    px, front = cam.project(np.asarray(points, dtype=float))
    assert front.all()
    return px


def test_crop_is_centred_on_the_principal_point_and_clear_of_the_strip() -> None:
    y0, y1, x0, x1 = framing.principal_crop(WINDOW_HW, STRIP_PX, SIZE)
    assert (y0 + y1) / 2 == WINDOW_HW[0] / 2
    assert (x0 + x1) / 2 == WINDOW_HW[1] / 2
    assert y1 <= WINDOW_HW[0] - STRIP_PX
    assert abs((x1 - x0) / (y1 - y0) - SIZE[0] / SIZE[1]) < 0.01


def test_simbody_fov_makes_the_visible_crop_match_the_viewer_lens() -> None:
    crop = framing.principal_crop(WINDOW_HW, STRIP_PX, SIZE)
    ch = crop[1] - crop[0]
    fov = framing.simbody_fov_y_rad(VIEWER_FOV_Y_RAD, WINDOW_HW[0], ch)
    focal = (WINDOW_HW[0] / 2) / math.tan(fov / 2)
    assert math.isclose(2 * math.atan((ch / 2) / focal), VIEWER_FOV_Y_RAD, rel_tol=1e-9)
    assert fov > VIEWER_FOV_Y_RAD  # the window is taller than the crop


def test_legacy_centre_crop_pushed_the_feet_off_the_image() -> None:
    """Regression record of the cause: 0.7 rad on the window, then fit_to_size."""
    h_w, w_w = WINDOW_HW
    focal = (h_w / 2) / math.tan(VIEWER_FOV_Y_RAD / 2)
    usable = h_w - STRIP_PX  # window[:-strip] then centre-crop to 16:9
    crop_h = round(w_w / (SIZE[0] / SIZE[1]))
    y0 = (usable - crop_h) // 2
    preset = get_view_preset("face_on")
    depth = preset.default_distance_m
    foot_row = h_w / 2 + focal * (LOOK[2] - 0.0) / depth  # world z = 0
    assert foot_row > y0 + crop_h  # below the cropped image


def test_fitted_distance_never_zooms_in_and_keeps_points_in_frame() -> None:
    preset = get_view_preset("face_on")
    pts = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 2.6], [0.5, 0.0, 1.0]])
    dist = framing.fitted_distance_m(
        "face_on", pts, LOOK, None, VIEWER_FOV_Y_RAD, SIZE[0] / SIZE[1]
    )
    assert dist >= preset.default_distance_m
    assert _inside_with_margin(_output_px(pts, "face_on", LOOK, dist), SIZE)
    # an explicit distance is honoured untouched
    assert (
        framing.fitted_distance_m(
            "face_on", pts, LOOK, 2.0, VIEWER_FOV_Y_RAD, SIZE[0] / SIZE[1]
        )
        == 2.0
    )


def test_fitted_lookat_centres_the_points_without_changing_depth() -> None:
    preset = get_view_preset("face_on")
    pts = np.array([[0.2, 0.0, 0.0], [1.8, 0.0, 2.6]])
    look = np.array(framing.fitted_lookat_m("face_on", pts, LOOK))
    up = pts @ preset.image_up()
    assert math.isclose(float(look @ preset.image_up()), up.mean(), abs_tol=1e-9)
    right = pts @ preset.image_right()
    assert math.isclose(float(look @ preset.image_right()), right.mean(), abs_tol=1e-9)
    depth = preset.view_direction()
    assert math.isclose(
        float(look @ depth), float(np.array(LOOK) @ depth), abs_tol=1e-9
    )


def test_fitted_distance_validates_points() -> None:
    with pytest.raises(ValueError, match="points"):
        framing.fitted_distance_m(
            "face_on", np.zeros((0, 3)), LOOK, None, VIEWER_FOV_Y_RAD, 16 / 9
        )


def test_framing_body_names_pick_feet_head_and_clubhead() -> None:
    names = ["Hip", "Head", "Clubhead", "Grip", "calcn_l", "toes_r", "talus_l"]
    assert framing.framing_body_names(names) == [
        "Head",
        "Clubhead",
        "calcn_l",
        "toes_r",
    ]


def test_opensim_clip_keeps_feet_head_and_clubhead_in_the_face_on_image() -> None:
    osim = require_opensim()
    from src.tools.native_viewer_export.backends import opensim_worker as worker

    raw = (SPECS / "full_body_spec_anthro_driver.json").read_bytes()
    model, ground_h = worker.build_model(osim, raw)
    state = model.initSystem()
    names = [c.getName() for c in model.getCoordinateSet()]
    q = np.zeros((12, len(names)))
    for k in range(q.shape[0]):  # a swept arm/torso motion about the address pose
        phase = math.sin(math.pi * k / (q.shape[0] - 1))
        for j, name in enumerate(names):
            if "Torso" in name or "Spine" in name or "Shoulder" in name:
                q[k, j] = 0.5 * phase
    points = worker.collect_framing_points(model, state, names, q, ground_h)
    assert points.ndim == 2 and points.shape[1] == 3 and len(points) >= q.shape[0]
    look = framing.fitted_lookat_m("face_on", points, LOOK)
    dist = framing.fitted_distance_m(
        "face_on", points, look, None, VIEWER_FOV_Y_RAD, SIZE[0] / SIZE[1]
    )
    assert _inside_with_margin(_output_px(points, "face_on", look, dist), SIZE)
