"""Bar and hand geometry reductions of the lift pack audit."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.lifting.pack_audit import geometry

pytestmark = pytest.mark.unit


def _pose(hand_l=(0.0, 0.22, 1.0), hand_r=(0.0, -0.22, 1.0)) -> dict:
    return {
        "barbell_shaft": (0.0, 0.0, 1.0),
        "barbell_left_sleeve": (0.0, 0.8775, 1.0),
        "barbell_right_sleeve": (0.0, -0.8775, 1.0),
        "hand_l": hand_l,
        "hand_r": hand_r,
        "foot_l": (0.0, 0.09, 0.08),
        "foot_r": (0.0, -0.09, 0.08),
    }


def test_bar_frame_axis_points_to_the_left_sleeve() -> None:
    centre, axis = geometry.bar_frame(_pose())
    assert np.allclose(centre, [0, 0, 1])
    assert np.allclose(axis, [0, 1, 0])


def test_hands_on_the_bar_have_zero_axis_distance() -> None:
    metrics = geometry.hand_bar_metrics(_pose())
    assert metrics["l"]["axis_distance_m"] == pytest.approx(0.0)
    assert metrics["l"]["lateral_m"] == pytest.approx(0.22)
    assert metrics["r"]["lateral_m"] == pytest.approx(-0.22)


def test_a_hand_off_the_bar_is_measured() -> None:
    metrics = geometry.hand_bar_metrics(_pose(hand_r=(0.1, -0.22, 0.7)))
    assert metrics["r"]["axis_distance_m"] == pytest.approx(np.hypot(0.1, 0.3))


def test_summary_is_independent_of_pelvis_translation() -> None:
    base = geometry.pose_summary(_pose())
    shifted = {k: tuple(np.add(v, (2.0, -1.0, 0.5))) for k, v in _pose().items()}
    moved = geometry.pose_summary(shifted)
    assert np.allclose(base["bar_centre_rel_feet_m"], moved["bar_centre_rel_feet_m"])
    assert base["grip_width_m"] == pytest.approx(moved["grip_width_m"])


def test_missing_body_and_degenerate_bar_rejected() -> None:
    pose = _pose()
    del pose["hand_l"]
    with pytest.raises(ValueError, match="hand_l"):
        geometry.hand_bar_metrics(pose)
    pose = _pose()
    pose["barbell_left_sleeve"] = pose["barbell_right_sleeve"]
    with pytest.raises(ValueError, match="coincide"):
        geometry.bar_frame(pose)


def test_max_abs_difference() -> None:
    a, b = _pose(), _pose(hand_l=(0.0, 0.22, 1.05))
    assert geometry.max_abs_difference(a, b, ["hand_l", "hand_r"]) == pytest.approx(
        0.05
    )
    with pytest.raises(ValueError):
        geometry.max_abs_difference(a, b, [])
