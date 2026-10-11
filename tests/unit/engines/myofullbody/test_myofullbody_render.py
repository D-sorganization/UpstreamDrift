"""Colour mapping, frame selection and camera tests for the renderer (#11646)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.myofullbody import render

pytestmark = pytest.mark.unit


def test_activation_colour_runs_blue_to_red() -> None:
    rgba = render.activation_rgba(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(rgba[0], [0.10, 0.30, 1.00, 1.0])
    np.testing.assert_allclose(rgba[2], [1.00, 0.10, 0.10, 1.0])
    np.testing.assert_allclose(rgba[1, :3], 0.5 * (render.COLD_RGB + render.HOT_RGB))
    assert rgba[0, 2] > rgba[0, 0] and rgba[2, 0] > rgba[2, 2]


def test_activation_colour_is_monotone_in_red() -> None:
    a = np.linspace(0.0, 1.0, 11)
    red = render.activation_rgba(a)[:, 0]
    assert (np.diff(red) > 0.0).all()


def test_activation_colour_contract() -> None:
    with pytest.raises(ValueError):
        render.activation_rgba(np.array([1.2]))
    with pytest.raises(ValueError):
        render.activation_rgba(np.array([np.nan]))
    with pytest.raises(ValueError):
        render.activation_rgba(np.array([0.5]), alpha=2.0)
    assert render.activation_rgba(np.array([1.0 + 1e-9]))[0, 0] == pytest.approx(1.0)


def test_quarter_speed_stretches_the_swing_four_times() -> None:
    t = np.arange(0.0, 1.0 + 1e-9, 0.005)  # 1 s swing sampled at 200 Hz
    shown = render.select_frames(t, fps=50.0, slowdown=0.25)
    assert shown.size == 200 + 1  # 1 s / 0.25 = 4 s of video at 50 fps
    assert shown[0] == 0 and shown[-1] == t.size - 1
    assert (np.diff(shown) >= 0).all()
    np.testing.assert_array_equal(np.diff(shown), 1)  # 50 fps x 0.25 = one 5 ms sample


def test_frame_selection_holds_nearest_sample_when_video_is_denser() -> None:
    t = np.array([0.0, 0.1, 0.2])
    shown = render.select_frames(t, fps=100.0, slowdown=0.25)
    assert shown.min() == 0 and shown.max() == 2
    assert np.all(np.diff(shown) >= 0)


def test_frame_selection_contract() -> None:
    with pytest.raises(ValueError):
        render.select_frames(np.array([0.0]))
    with pytest.raises(ValueError):
        render.select_frames(np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        render.select_frames(np.array([0.0, 1.0]), slowdown=0.0)


def test_cameras_cover_four_views_with_expected_geometry() -> None:
    cams = render.view_cameras(
        np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.zeros(3)
    )
    assert tuple(cams) == render.VIEW_NAMES
    assert cams["overhead"]["elevation"] < -80.0
    assert cams["face_on"]["azimuth"] == pytest.approx(90.0)  # looks toward +y
    assert cams["down_the_line"]["azimuth"] == pytest.approx(0.0)  # looks toward +x
    assert cams["oblique"]["azimuth"] == pytest.approx(45.0)
    with pytest.raises(ValueError):
        render.view_cameras(
            np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.zeros(3)
        )


def test_circular_mean_handles_the_wrap() -> None:
    assert render.circular_mean_deg(170.0, -170.0) == pytest.approx(
        180.0, abs=1e-9
    ) or (abs(render.circular_mean_deg(170.0, -170.0)) == pytest.approx(180.0))


def test_club_segment_starts_at_the_grip_and_follows_the_lead_forearm() -> None:
    elbow = np.array([0.0, 0.0, 1.0])
    lead = np.array([0.0, 0.0, 0.5])  # forearm points straight down
    trail = np.array([0.0, 0.1, 0.5])
    grip, head = render.club_segment(elbow, lead, trail, 1.1)
    np.testing.assert_allclose(grip, [0.0, 0.05, 0.5])
    np.testing.assert_allclose(head, [0.0, 0.05, -0.6])
    assert np.linalg.norm(head - grip) == pytest.approx(1.1)


def test_club_segment_contract() -> None:
    v = np.zeros(3)
    with pytest.raises(ValueError):
        render.club_segment(v, v, v, 1.0)  # degenerate forearm
    with pytest.raises(ValueError):
        render.club_segment(v, np.ones(3), v, 0.0)
