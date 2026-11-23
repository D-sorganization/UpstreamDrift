"""Impact-zone playback switch: toggling slow motion never jumps (#15524)."""

from __future__ import annotations

import numpy as np
import pytest

from src.tools.model_explorer.impact_zone_playback import ImpactZonePlayback
from src.tools.model_explorer.playback_clock import PlaybackClock

pytestmark = pytest.mark.unit

TIME = np.linspace(0.0, 2.0, 201)  # 100 Hz over 2 s
WINDOW = (1.0, 0.1)  # impact_s, half_width_s: zone is 0.9 s .. 1.1 s
FACTOR = 0.1


def _playback(window: tuple[float, float] | None = WINDOW) -> ImpactZonePlayback:
    return ImpactZonePlayback(TIME, window, slow_factor=FACTOR)


def test_without_a_window_it_is_unavailable_and_plays_plainly() -> None:
    pb, plain = _playback(None), PlaybackClock(TIME)
    assert pb.available is False
    assert pb.enabled is False
    pb.start(0, now_s=0.0)
    plain.start(0, now_s=0.0)
    for wall in (0.0, 0.7, 1.4, 2.3):
        assert pb.frame_at(wall) == plain.frame_at(wall)


def test_enabling_without_a_window_fails_closed() -> None:
    pb = _playback(None)
    with pytest.raises(ValueError, match="no impact window"):
        pb.set_enabled(True, now_s=0.0)
    assert pb.enabled is False


def test_disabling_when_unavailable_is_a_harmless_no_op() -> None:
    pb = _playback(None)
    pb.set_enabled(False, now_s=0.0)
    assert pb.enabled is False


def test_enabling_mid_run_keeps_the_current_frame_then_slows_down() -> None:
    pb, plain = _playback(), PlaybackClock(TIME)
    pb.start(0, now_s=0.0)
    plain.start(0, now_s=0.0)
    assert pb.frame_at(0.85) == 85
    pb.set_enabled(True, now_s=0.85)
    assert pb.enabled is True
    assert pb.frame_at(0.85) == 85  # no jump at the switch
    # 0.15 s later the plain clock is at 1.0 s; the slow clock spent 0.05 s of
    # wall time before the zone and 0.1 s inside it (0.01 s of media).
    assert plain.frame_at(1.0) == 100
    assert pb.frame_at(1.0) == 91


def test_disabling_inside_the_zone_keeps_the_frame_then_catches_up() -> None:
    pb = _playback()
    pb.start(0, now_s=0.0)
    pb.set_enabled(True, now_s=0.0)
    frame = pb.frame_at(1.5)
    assert 90 <= frame <= 110  # still inside the slow zone
    pb.set_enabled(False, now_s=1.5)
    assert pb.enabled is False
    assert pb.frame_at(1.5) == frame
    assert pb.frame_at(1.5 + 0.2) == frame + 20  # plain rate again


def test_speed_survives_toggling_in_both_directions() -> None:
    pb = _playback()
    pb.start(0, now_s=0.0)
    pb.set_speed(0.5, now_s=0.0)
    pb.set_enabled(True, now_s=0.2)
    assert pb.speed == 0.5
    pb.set_speed(0.25, now_s=0.4)
    pb.set_enabled(False, now_s=0.6)
    assert pb.speed == 0.25


def test_enabling_twice_does_not_rebase_the_clock() -> None:
    pb = _playback()
    pb.start(0, now_s=0.0)
    pb.set_enabled(True, now_s=0.0)
    before = pb.frame_at(1.2)
    pb.set_enabled(True, now_s=1.2)
    assert pb.frame_at(1.2) == before
    reference = _playback()
    reference.start(0, now_s=0.0)
    reference.set_enabled(True, now_s=0.0)
    assert pb.frame_at(1.5) == reference.frame_at(1.5)


def test_loop_wall_s_grows_only_while_slow_motion_is_on() -> None:
    pb = _playback()
    plain_loop = pb.loop_wall_s
    assert plain_loop == pytest.approx(2.0)
    pb.set_enabled(True, now_s=0.0)
    # 1.8 s outside the zone + 0.2 s / 0.1 inside it.
    assert pb.loop_wall_s == pytest.approx(3.8)
    pb.set_enabled(False, now_s=0.0)
    assert pb.loop_wall_s == pytest.approx(plain_loop)


def test_start_rebases_the_active_clock() -> None:
    pb = _playback()
    pb.set_enabled(True, now_s=0.0)
    pb.start(50, now_s=10.0)
    assert pb.frame_at(10.0) == 50
    pb.set_enabled(False, now_s=10.0)
    assert pb.frame_at(10.0) == 50


@pytest.mark.parametrize(
    "window",
    [(3.0, 0.1), (-0.5, 0.1), (1.0, 0.0), (1.0, -0.1), (float("nan"), 0.1)],
)
def test_a_window_outside_the_run_or_degenerate_is_rejected_up_front(
    window: tuple[float, float],
) -> None:
    with pytest.raises(ValueError):
        _playback(window)


@pytest.mark.parametrize("factor", [0.0, -0.5, 1.5, float("inf")])
def test_invalid_slow_factor_is_rejected(factor: float) -> None:
    with pytest.raises(ValueError, match="slow_factor"):
        ImpactZonePlayback(TIME, WINDOW, slow_factor=factor)


def test_bad_time_axis_is_rejected_even_without_a_window() -> None:
    with pytest.raises(ValueError, match="time axis"):
        ImpactZonePlayback(np.array([0.0]), None)


def test_enabling_on_the_terminal_frame_does_not_wrap_to_the_start() -> None:
    pb = _playback()
    pb.start(0, now_s=0.0)
    assert pb.frame_at(1.999) == TIME.size - 1
    pb.set_enabled(True, now_s=1.999)
    assert pb.frame_at(1.999) == TIME.size - 1  # no 200 -> 0 jump


def test_disabling_on_the_terminal_frame_does_not_wrap_to_the_start() -> None:
    pb = _playback()
    pb.set_enabled(True, now_s=0.0)
    pb.start(0, now_s=0.0)
    wall = pb.loop_wall_s - 0.001
    assert pb.frame_at(wall) == TIME.size - 1
    pb.set_enabled(False, now_s=wall)
    assert pb.frame_at(wall) == TIME.size - 1  # no 200 -> 0 jump


@pytest.mark.parametrize("slow_factor", [0.0, -0.5, 1.5, float("nan"), float("inf")])
@pytest.mark.parametrize("window", [None, WINDOW])
def test_invalid_slow_factor_is_rejected_with_or_without_a_window(
    slow_factor: float, window: tuple[float, float] | None
) -> None:
    with pytest.raises(ValueError, match="slow_factor"):
        ImpactZonePlayback(TIME, window, slow_factor=slow_factor)
