# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for AnimationControlMixin playback on barbell tabs."""

from __future__ import annotations

import pytest

from movement_optimizer.gui.main_window import MainWindow

from .conftest import make_test_result


@pytest.fixture
def window_with_result(qapp):
    win = MainWindow()
    win._resolve_exercise_params(0)
    win._set_exercise_result(0, make_test_result())
    win.tabs.setCurrentIndex(0)  # a barbell tab -> mixin handles playback
    yield win
    win._stop_anim()
    win.close()


def test_step_forward_back_rewind_jump(window_with_result) -> None:
    win = window_with_result
    win._step_fwd()
    assert win.exercise_states[0].anim_frame == 1
    win._step_back()
    assert win.exercise_states[0].anim_frame == 0
    win._jump_to_end()
    assert win.exercise_states[0].anim_frame == len(win.exercise_states[0].result.t) - 1
    win._rewind()
    assert win.exercise_states[0].anim_frame == 0


def test_toggle_play_starts_and_stops(window_with_result) -> None:
    win = window_with_result
    win._toggle_play()  # start
    assert win.is_playing
    win._toggle_play()  # stop
    assert not win.is_playing


def test_anim_step_advances_frame(window_with_result) -> None:
    win = window_with_result
    win.is_playing = True
    win._anim_step()
    assert win.exercise_states[0].anim_frame == 1
    win._stop_anim()


def test_anim_step_noop_when_not_playing(window_with_result) -> None:
    win = window_with_result
    win.is_playing = False
    win._anim_step()
    assert win.exercise_states[0].anim_frame == 0


def test_on_speed_updates_controls(window_with_result) -> None:
    window_with_result._on_speed(2.0)  # smoke; updates the speed label


def test_playback_helpers_noop_without_result(qapp) -> None:
    win = MainWindow()
    win.tabs.setCurrentIndex(0)
    win.exercise_states[0].result = None
    # All of these must early-return without raising when no result is loaded.
    win._toggle_play()
    win._step_fwd()
    win._step_back()
    win._rewind()
    win._jump_to_end()
    assert not win.is_playing
    win.close()
