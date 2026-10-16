"""Wall-clock playback timing for the results panel (#12948, epic #12878)."""

from __future__ import annotations

import numpy as np
import pytest

from src.tools.model_explorer.playback_clock import SPEEDS, PlaybackClock

TIME = np.linspace(0.0, 2.0, 21)  # 10 Hz samples over 2 s


@pytest.mark.unit
def test_speeds_offered_are_quarter_half_and_real_time() -> None:
    assert SPEEDS == (0.25, 0.5, 1.0)


@pytest.mark.unit
def test_frame_follows_wall_clock_at_real_time() -> None:
    clock = PlaybackClock(TIME)
    clock.start(0, now_s=100.0)
    assert clock.frame_at(100.0) == 0
    assert clock.frame_at(100.5) == 5
    assert clock.frame_at(101.0) == 10


@pytest.mark.unit
def test_slow_render_skips_frames_instead_of_slowing_down() -> None:
    clock = PlaybackClock(TIME)
    clock.start(0, now_s=0.0)
    # Only two ticks happened in 1.5 s: the frame is still the right one.
    assert clock.frame_at(0.3) == 3
    assert clock.frame_at(1.5) == 15


@pytest.mark.unit
@pytest.mark.parametrize(("speed", "frame"), [(0.25, 3), (0.5, 6), (1.0, 12)])
def test_speed_scales_the_media_rate(speed: float, frame: int) -> None:
    clock = PlaybackClock(TIME, speed=speed)
    clock.start(0, now_s=0.0)
    assert clock.frame_at(1.2) == frame


@pytest.mark.unit
def test_changing_speed_keeps_the_current_position() -> None:
    clock = PlaybackClock(TIME)
    clock.start(0, now_s=0.0)
    assert clock.frame_at(0.4) == 4
    clock.set_speed(0.5, now_s=0.4)
    assert clock.frame_at(0.4) == 4
    assert clock.frame_at(1.4) == 9  # +1 s wall = +0.5 s media = +5 frames


@pytest.mark.unit
def test_playback_wraps_at_the_end_of_the_run() -> None:
    clock = PlaybackClock(TIME)
    clock.start(0, now_s=0.0)
    assert clock.frame_at(2.3) == 3


@pytest.mark.unit
def test_start_from_a_scrubbed_frame() -> None:
    clock = PlaybackClock(TIME)
    clock.start(12, now_s=5.0)
    assert clock.frame_at(5.0) == 12
    assert clock.frame_at(5.2) == 14


@pytest.mark.unit
def test_non_uniform_times_pick_the_nearest_sample() -> None:
    clock = PlaybackClock(np.array([0.0, 0.1, 0.5, 1.0]))
    clock.start(0, now_s=0.0)
    assert clock.frame_at(0.2) == 1
    assert clock.frame_at(0.4) == 2


@pytest.mark.unit
@pytest.mark.parametrize("speed", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_speed(speed: float) -> None:
    with pytest.raises(ValueError, match="speed"):
        PlaybackClock(TIME, speed=speed)
    clock = PlaybackClock(TIME)
    with pytest.raises(ValueError, match="speed"):
        clock.set_speed(speed, now_s=0.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad",
    [np.array([0.0]), np.array([0.0, 0.0]), np.array([1.0, 0.5]), np.zeros((2, 2))],
)
def test_rejects_invalid_time_axis(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="time"):
        PlaybackClock(bad)


@pytest.mark.unit
def test_start_rejects_a_frame_outside_the_run() -> None:
    clock = PlaybackClock(TIME)
    with pytest.raises(ValueError, match="frame"):
        clock.start(21, now_s=0.0)
