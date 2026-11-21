"""Raised, pulled-back down-the-line flight view (#15150).

From behind the golfer at eye height a 207 m carry collapses into a few pixels
at the horizon; the raised view frames the golfer in the foreground and the
whole flight above a readable stretch of ground.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.golf_view_presets import (
    VIEW_ORDER,
    ViewPreset,
    get_view_preset,
    mujoco_camera_params,
    tracked_lookats,
)
from src.tools.native_viewer_export import flight_tracer as tracer
from src.tools.native_viewer_export.overlay2d import pinhole_for_view

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

HIGH, FLAT = "down_the_line_flight_high", "down_the_line_flight"
LOOKAT, SIZE = (1.0, 0.0, 0.9), (1280, 720)
FOV = math.radians(tracer.load_tracer_settings().fov_y_deg)
HUD_TOP_PX, CARRY_M, APEX_M, RIGHT_M = 470, 207.0, 18.0, 15.0


def _project(view: str, points: np.ndarray) -> np.ndarray:
    camera = pinhole_for_view(view, LOOKAT, None, FOV, SIZE)
    pixels, _ = tracer.project(camera, points, 0.05)
    return pixels


def _flight() -> np.ndarray:
    """Target line is world -Y; the ball ends RIGHT_M to the side."""
    s = np.linspace(0.0, 1.0, 200)
    return np.column_stack(
        [1.0 + RIGHT_M * s, -CARRY_M * s, 0.03 + 4.0 * APEX_M * s * (1.0 - s)]
    )


def test_high_view_is_registered_beside_the_unchanged_flight_view() -> None:
    high, flat = get_view_preset(HIGH), get_view_preset(FLAT)
    assert HIGH not in VIEW_ORDER  # the 2x2 grid stays four views
    assert high.tracks is None and high.azimuth_deg == flat.azimuth_deg
    assert high.elevation_deg < flat.elevation_deg  # looks further down
    assert (flat.elevation_deg, flat.default_distance_m) == (-3.0, 4.6)
    assert flat.lookat_offset_m == (0.0, 0.0, 0.0)


def test_camera_is_raised_and_behind_the_golfer() -> None:
    cam = np.asarray(get_view_preset(HIGH).camera_position(LOOKAT, 15.72))
    assert cam[1] > 0.0, "behind the golfer (target is -Y)"
    assert cam[2] >= 2.9, "raised well above the eye-height view"
    params = mujoco_camera_params(HIGH, LOOKAT)
    assert params.lookat[1] < 0.0 and params.lookat[1] != LOOKAT[1]


def test_lookat_offset_moves_the_look_at_point_only_for_that_preset() -> None:
    high, flat = get_view_preset(HIGH), get_view_preset(FLAT)
    (got,) = tracked_lookats(high, LOOKAT, [None])
    np.testing.assert_allclose(got, np.add(LOOKAT, high.lookat_offset_m))
    assert tracked_lookats(flat, LOOKAT, [None]) == [LOOKAT]
    centre = _project(HIGH, np.array([got]))[0]
    np.testing.assert_allclose(centre, (640.0, 360.0), atol=1e-6)


@pytest.mark.parametrize("bad", [(0.0, 1.0), (0.0, float("nan"), 0.0), "abc"])
def test_lookat_offset_is_validated(bad: object) -> None:
    with pytest.raises(ValueError, match="lookat_offset_m"):
        ViewPreset("x", "X", -90.0, -10.0, 5.0, lookat_offset_m=bad)  # type: ignore[arg-type]


def test_golfer_is_recognisable_and_clear_of_the_hud() -> None:
    feet, head = _project(HIGH, np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.8]]))
    assert feet[1] - head[1] >= 150.0
    assert feet[1] <= HUD_TOP_PX


def test_whole_flight_is_framed_and_the_landing_is_below_the_horizon() -> None:
    pixels = _project(HIGH, _flight())
    assert (pixels >= 0).all() and (pixels < SIZE).all(), "flight inside the frame"
    assert pixels[:, 1].min() >= 30.0, "apex clear of the top edge"
    preset = get_view_preset(HIGH)
    horizon_v = SIZE[1] / 2 - (SIZE[1] / 2 / math.tan(FOV / 2)) * math.tan(
        math.radians(-preset.elevation_deg)
    )
    land_v = pixels[-1, 1]
    assert land_v - horizon_v >= 10.0, "visible ground beyond the landing point"
    tee_v = pixels[0, 1]
    assert tee_v - land_v >= 240.0, "the ground track spans real screen height"
    # the arc is open: it rises to the apex, then comes down at a different place
    apex = int(np.argmin(pixels[:, 1]))
    assert 0 < apex < len(pixels) - 1
    assert abs(pixels[-1, 0] - pixels[0, 0]) >= 30.0
    assert (np.diff(pixels[:apex, 1]) < 0).all()
    assert (np.diff(pixels[apex:, 1]) > 0).all()
