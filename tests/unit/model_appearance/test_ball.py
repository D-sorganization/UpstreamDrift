"""Ball placement at address: one shared function (OSV-3 #11729, GCV-13 #11719)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.model_appearance.ball import (
    BALL_RADIUS_M,
    ball_position_at_address,
)

pytestmark = pytest.mark.unit


def test_radius_is_regulation_ball() -> None:
    assert BALL_RADIUS_M == pytest.approx(0.021335)


def test_ball_rests_on_ground_and_touches_face_plane() -> None:
    face = np.array([0.0, 0.0, 0.03])
    normal = np.array([1.0, 0.0, 0.0])
    ball = ball_position_at_address(face, normal, ground_height_m=0.0)
    assert ball[2] == pytest.approx(BALL_RADIUS_M)
    assert ball[0] == pytest.approx(BALL_RADIUS_M)


def test_normal_is_normalised_and_ground_height_respected() -> None:
    ball = ball_position_at_address(
        [1.0, 2.0, 0.55], [0.0, 3.0, 0.0], ground_height_m=0.5
    )
    assert ball == pytest.approx([1.0, 2.0 + BALL_RADIUS_M, 0.5 + BALL_RADIUS_M])


def test_lofted_normal_keeps_horizontal_offset_only() -> None:
    n = np.array([np.cos(np.radians(10)), 0.0, np.sin(np.radians(10))])
    ball = ball_position_at_address([0.0, 0.0, 0.02], n, ground_height_m=0.0)
    assert ball[0] == pytest.approx(BALL_RADIUS_M * np.cos(np.radians(10)))
    assert ball[2] == pytest.approx(BALL_RADIUS_M)


@pytest.mark.parametrize(
    "face, normal",
    [
        ([0, 0], [1, 0, 0]),
        ([0, 0, 0], [0, 0, 0]),
        ([0, 0, np.nan], [1, 0, 0]),
    ],
)
def test_rejects_bad_inputs(face, normal) -> None:
    with pytest.raises(ValueError):
        ball_position_at_address(face, normal)


def test_rejects_nonpositive_radius() -> None:
    with pytest.raises(ValueError):
        ball_position_at_address([0, 0, 0], [1, 0, 0], radius_m=0.0)
