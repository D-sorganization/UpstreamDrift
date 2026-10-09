"""Hands close-up camera preset tracking the grip midpoint (GCV-10, #11716)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.golf_view_presets import (
    VIEW_ORDER,
    get_view_preset,
    mujoco_camera_params,
    tracked_lookats,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

STATIC = (1.0, 0.0, 0.9)


def test_preset_is_a_close_tracked_view_outside_the_grid():
    p = get_view_preset("hands_closeup")
    assert p.tracks == "grip_midpoint"
    assert "hands_closeup" not in VIEW_ORDER  # the 2x2 grid stays four views
    assert p.default_distance_m < 1.5
    assert get_view_preset("face_on").tracks is None
    assert p.view_direction()[2] < 0.0


def test_target_equals_the_grip_midpoint_over_time():
    mid = [(0.1 * k, 0.02 * k, 1.0 - 0.01 * k) for k in range(6)]
    looks = tracked_lookats(get_view_preset("hands_closeup"), STATIC, mid)
    assert looks == [tuple(m) for m in mid]
    for k, look in enumerate(looks):
        cam = mujoco_camera_params("hands_closeup", look, None)
        assert cam.lookat == pytest.approx(mid[k])


def test_missing_focus_holds_the_last_known_point_never_the_origin():
    mid = [None, (0.5, 0.0, 1.0), None, (float("nan"), 0.0, 1.0), (0.6, 0.1, 1.1)]
    looks = tracked_lookats(get_view_preset("hands_closeup"), STATIC, mid)
    assert looks[0] == STATIC  # nothing known yet: the static look-at
    assert looks[1] == (0.5, 0.0, 1.0)
    assert looks[2] == (0.5, 0.0, 1.0) and looks[3] == (0.5, 0.0, 1.0)
    assert looks[4] == (0.6, 0.1, 1.1)
    assert not any(np.allclose(look, 0.0) for look in looks)


def test_untracked_presets_keep_the_static_lookat():
    looks = tracked_lookats(get_view_preset("face_on"), STATIC, [(9, 9, 9)] * 3)
    assert looks == [STATIC] * 3


def test_preconditions():
    with pytest.raises(ValueError):
        tracked_lookats(get_view_preset("hands_closeup"), (1.0, 0.0), [])
    with pytest.raises(ValueError):
        get_view_preset("nope")
