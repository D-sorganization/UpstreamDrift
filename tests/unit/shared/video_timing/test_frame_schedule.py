"""Time-based frame schedule (GCV-14, #11720)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.video_timing.frame_schedule import (
    FrameSchedule,
    quaternion_groups_from_model,
    select_frames,
    slerp,
    speed_suffix,
    stride_for_speed,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

T = np.arange(0.0, 0.3 + 1e-12, 0.001)  # 0.3 s swing sampled at 1 kHz


def test_sample_times_are_monotone_and_cover_the_swing() -> None:
    times = FrameSchedule(T).sample_times_s
    assert (np.diff(times) > 0).all()
    assert times[0] == T[0]
    assert T[-1] - times[-1] < 1.0 / 60.0  # last frame within one video frame


def test_speed_one_spacing_is_one_over_fps_of_real_time() -> None:
    times = FrameSchedule(T, fps=60.0, speed=1.0).sample_times_s
    np.testing.assert_allclose(np.diff(times), 1.0 / 60.0)


def test_half_speed_doubles_the_frame_count() -> None:
    full = FrameSchedule(T, fps=60.0, speed=1.0).n_frames
    half = FrameSchedule(T, fps=60.0, speed=0.5).n_frames
    assert abs(half - 2 * full) <= 1
    np.testing.assert_allclose(
        np.diff(FrameSchedule(T, speed=0.5).sample_times_s), 0.5 / 60.0
    )


def test_window_clips_to_the_requested_interval() -> None:
    sch = FrameSchedule(T, fps=60.0, speed=0.1, window=(0.25, 0.35))
    assert sch.sample_times_s[0] == pytest.approx(0.25)
    assert sch.sample_times_s[-1] <= T[-1] + 1e-12
    assert sch.n_frames == 31  # 0.05 s of swing at 0.1x and 60 fps


def test_window_must_overlap_the_swing() -> None:
    with pytest.raises(ValueError, match="window"):
        FrameSchedule(T, window=(1.0, 2.0))
    with pytest.raises(ValueError, match="window"):
        FrameSchedule(T, window=(0.2, 0.1))


@pytest.mark.parametrize(
    "kwargs",
    [{"fps": 0.0}, {"fps": -1.0}, {"speed": 0.0}, {"speed": float("nan")}],
)
def test_schedule_contract(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        FrameSchedule(T, **kwargs)


def test_times_contract() -> None:
    with pytest.raises(ValueError):
        FrameSchedule(np.array([0.0]))
    with pytest.raises(ValueError):
        FrameSchedule(np.array([0.0, 0.0, 1.0]))


def test_interpolation_is_exact_at_sample_times() -> None:
    states = np.random.default_rng(0).normal(size=(T.size, 3))
    sch = FrameSchedule(T, fps=1000.0, speed=1.0)  # frames land on samples
    np.testing.assert_allclose(sch.interpolate(states), states, atol=1e-9)


def test_interpolation_is_linear_between_samples() -> None:
    t = np.array([0.0, 0.1])
    states = np.array([[0.0, 10.0], [1.0, 20.0]])
    out = FrameSchedule(t, fps=50.0, speed=1.0).interpolate(states)
    np.testing.assert_allclose(out[:, 0], np.arange(out.shape[0]) * 0.02 / 0.1)
    np.testing.assert_allclose(out[:, 1], 10.0 + 10.0 * out[:, 0])


def test_quaternion_slerp_midpoint_and_unit_norm() -> None:
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # 90 deg about z
    mid = slerp(a, b, 0.5)
    assert np.linalg.norm(mid) == pytest.approx(1.0)
    assert 2 * np.degrees(np.arccos(mid[0])) == pytest.approx(45.0)
    np.testing.assert_allclose(slerp(a, b, 0.0), a)
    np.testing.assert_allclose(slerp(a, b, 1.0), b)


def test_slerp_takes_the_short_path_for_antipodal_signs() -> None:
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = -np.array([np.cos(0.1), 0.0, 0.0, np.sin(0.1)])
    mid = slerp(a, b, 0.5)
    assert abs(mid[0]) > 0.99  # not swung around the long way


def test_quaternion_groups_are_slerped_not_lerped() -> None:
    t = np.array([0.0, 1.0])
    half_turn = np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about z
    states = np.array([[1.0, 0.0, 0.0, 0.0, 5.0], np.append(half_turn, 7.0)])
    sch = FrameSchedule(t, fps=2.0, speed=1.0)  # frames at 0, 0.5, 1.0
    out = sch.interpolate(states, quaternion_groups=((0, 1, 2, 3),))
    quat = out[1, :4]
    assert np.linalg.norm(quat) == pytest.approx(1.0)
    assert quat[0] == pytest.approx(np.cos(np.pi / 4))  # 90 deg, lerp would shrink
    assert out[1, 4] == pytest.approx(6.0)


def test_interpolate_rejects_mismatched_states() -> None:
    with pytest.raises(ValueError, match="states"):
        FrameSchedule(T).interpolate(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="quaternion"):
        FrameSchedule(T).interpolate(
            np.zeros((T.size, 2)), quaternion_groups=((0, 1, 2, 3),)
        )


def test_nearest_indices_match_legacy_select_frames() -> None:
    t = np.arange(0.0, 1.0 + 1e-9, 0.005)
    legacy = select_frames(t, fps=50.0, slowdown=0.25)
    assert legacy.size == 201
    np.testing.assert_array_equal(
        FrameSchedule(t, 50.0, 0.25).nearest_indices(), legacy
    )
    with pytest.raises(ValueError):
        select_frames(np.array([0.0, 1.0]), slowdown=0.0)


class _Model:
    njnt = 3
    jnt_type = np.array([0, 1, 3])  # free, ball, hinge
    jnt_qposadr = np.array([0, 7, 11])


def test_quaternion_groups_from_model_metadata() -> None:
    assert quaternion_groups_from_model(_Model()) == ((3, 4, 5, 6), (7, 8, 9, 10))


def test_stride_for_speed_and_suffixes() -> None:
    assert stride_for_speed(0.001, 60.0, 1.0) == 17  # 16.7 ms per frame
    assert stride_for_speed(0.001, 60.0, 0.5) == 8
    assert stride_for_speed(0.1, 60.0, 0.5) == 1  # never below one sample
    with pytest.raises(ValueError):
        stride_for_speed(0.0)
    assert [speed_suffix(v) for v in (1, 0.5, 0.25, 0.1)] == [
        "_1x",
        "_0p5x",
        "_0p25x",
        "_0p1x",
    ]
