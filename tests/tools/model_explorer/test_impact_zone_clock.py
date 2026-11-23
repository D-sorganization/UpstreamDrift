"""Impact-zone playback clock: slow motion around contact (#15524)."""

from __future__ import annotations

import numpy as np
import pytest

from src.tools.model_explorer.impact_zone_clock import ImpactZoneClock
from src.tools.model_explorer.playback_clock import PlaybackClock

pytestmark = pytest.mark.unit

TIME = np.linspace(0.0, 2.0, 201)  # 100 Hz over 2 s
IMPACT_S = 1.0
HALF_S = 0.1  # zone is 0.9 s .. 1.1 s of media time
FACTOR = 0.1  # ten times slower inside the zone


def _clock(**kw: float) -> ImpactZoneClock:
    args = {"impact_s": IMPACT_S, "half_width_s": HALF_S, "slow_factor": FACTOR}
    return ImpactZoneClock(TIME, **{**args, **kw})


def test_outside_the_zone_it_plays_like_the_plain_clock() -> None:
    zone, plain = _clock(), PlaybackClock(TIME)
    zone.start(0, now_s=0.0)
    plain.start(0, now_s=0.0)
    for wall in (0.0, 0.25, 0.5, 0.89):
        assert zone.frame_at(wall) == plain.frame_at(wall)


def test_inside_the_zone_media_time_advances_at_the_slow_factor() -> None:
    clock = _clock()
    clock.start(0, now_s=0.0)
    # 0.9 s of lead-in is 0.9 s of wall time; then 0.5 s wall = 0.05 s media.
    assert clock.media_time_at(0.9) == pytest.approx(0.9)
    assert clock.media_time_at(1.4) == pytest.approx(0.9 + 0.05)


def test_after_the_zone_normal_speed_resumes() -> None:
    clock = _clock()
    clock.start(0, now_s=0.0)
    zone_wall = HALF_S * 2 / FACTOR  # 2.0 s of wall time inside the zone
    assert clock.media_time_at(0.9 + zone_wall) == pytest.approx(1.1)
    assert clock.media_time_at(0.9 + zone_wall + 0.4) == pytest.approx(1.5)


def test_loop_restarts_and_speed_scales_the_whole_clock() -> None:
    clock = _clock()
    clock.start(0, now_s=0.0)
    loop_wall = 0.9 + 2.0 + 0.9
    assert clock.loop_wall_s == pytest.approx(loop_wall)
    assert clock.media_time_at(loop_wall + 0.5) == pytest.approx(0.5)
    clock.set_speed(0.5, now_s=0.0)
    assert clock.media_time_at(1.0) == pytest.approx(0.5)  # 0.5x of real time
    assert clock.loop_wall_s == pytest.approx(2 * loop_wall)


def test_start_inside_the_zone_and_set_speed_keep_the_current_frame() -> None:
    clock = _clock()
    clock.start(100, now_s=5.0)  # exactly at contact
    assert clock.frame_at(5.0) == 100
    assert clock.media_time_at(5.5) == pytest.approx(1.05)
    clock.set_speed(0.25, now_s=5.5)
    assert clock.media_time_at(5.5) == pytest.approx(1.05)


def test_zone_is_clamped_to_the_run_and_factor_one_is_plain_playback() -> None:
    edge = _clock(impact_s=0.02)  # zone would start before the run
    edge.start(0, now_s=0.0)
    assert edge.media_time_at(0.12 / FACTOR) == pytest.approx(0.12)
    flat = _clock(slow_factor=1.0)
    plain = PlaybackClock(TIME)
    flat.start(0, now_s=0.0)
    plain.start(0, now_s=0.0)
    assert flat.frame_at(1.37) == plain.frame_at(1.37)


@pytest.mark.parametrize(
    "bad",
    [
        {"impact_s": float("nan")},
        {"impact_s": 5.0},  # outside the run
        {"half_width_s": 0.0},
        {"slow_factor": 0.0},
        {"slow_factor": 1.5},
    ],
)
def test_bad_zone_is_rejected(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        _clock(**bad)
