# Copyright (c) 2026 D-Sorganization. All rights reserved.
# mypy: disable-error-code="misc,has-type"
# Mixin pattern: methods annotate self as MainWindow to access its attributes,
# but mypy cannot verify this pattern without the concrete class in scope.
"""Animation playback helpers extracted from MainWindow.

Provides play/pause toggle, step forward/back, rewind, and frame
advance as a mixin class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import BodyModel

if TYPE_CHECKING:
    from .main_window import MainWindow


def _require_body_for_animation(body: BodyModel | None) -> BodyModel:
    if body is None:
        raise ValueError("DbC Blocked: animation frame requires a body model.")
    return body


def _active_exercise_index(window: MainWindow) -> int | None:
    """Return the active barbell exercise index, or ``None`` for analysis tabs."""
    idx = window.tabs.currentIndex()
    if 0 <= idx < len(window.EXERCISE_CONFIGS):
        return idx
    return None


class AnimationControlMixin:
    """Mixin providing animation playback for MainWindow."""

    is_playing: bool

    def _toggle_play(self: MainWindow) -> None:  # type: ignore[override]
        idx = _active_exercise_index(self)
        if idx is None:
            self._stop_anim()
            return
        r, _fi, _body, _dyn = self._snapshot_idx_state(idx)
        if r is None:
            return
        if self.is_playing:
            self._stop_anim()
        else:
            self.is_playing = True
            self.controls.set_playing(True)
            self._anim_step()

    def _stop_anim(self: MainWindow) -> None:  # type: ignore[override]
        self.is_playing = False
        self.anim_timer.stop()
        self.controls.set_playing(False)

    def _anim_step(self: MainWindow) -> None:  # type: ignore[override]
        if not self.is_playing:
            return
        idx = _active_exercise_index(self)
        if idx is None:
            self._stop_anim()
            return
        r, fi, body, dyn = self._snapshot_idx_state(idx)
        if r is None:
            self._stop_anim()
            return

        _, etype = self.EXERCISE_CONFIGS[idx]
        body = _require_body_for_animation(body)
        self.exercise_tabs[idx].draw_anim_frame(
            fi,
            r,
            dyn,
            body,
            etype,
        )

        n = len(r.t)
        next_frame = (fi + 1) % n
        self._set_anim_frame(idx, next_frame)
        speed = self.controls.speed_multiplier()
        self.controls.set_playback_status(fi + 1, n, speed)
        delay = max(15, int(40 / max(0.1, speed)))
        if next_frame == 0:
            delay = 700
        self.anim_timer.start(delay)

    def _step_fwd(self: MainWindow) -> None:  # type: ignore[override]
        idx = _active_exercise_index(self)
        if idx is None:
            self._stop_anim()
            return
        r, fi, body, dyn = self._snapshot_idx_state(idx)
        if r is None:
            return
        self._stop_anim()
        n = len(r.t)
        new_frame = (fi + 1) % n
        self._set_anim_frame(idx, new_frame)
        _, etype = self.EXERCISE_CONFIGS[idx]
        body = _require_body_for_animation(body)
        self.exercise_tabs[idx].draw_anim_frame(
            new_frame,
            r,
            dyn,
            body,
            etype,
        )
        self.controls.set_playback_status(
            new_frame + 1,
            n,
            self.controls.speed_multiplier(),
        )

    def _step_back(self: MainWindow) -> None:  # type: ignore[override]
        idx = _active_exercise_index(self)
        if idx is None:
            self._stop_anim()
            return
        r, fi, body, dyn = self._snapshot_idx_state(idx)
        if r is None:
            return
        self._stop_anim()
        n = len(r.t)
        new_frame = (fi - 1) % n
        self._set_anim_frame(idx, new_frame)
        _, etype = self.EXERCISE_CONFIGS[idx]
        body = _require_body_for_animation(body)
        self.exercise_tabs[idx].draw_anim_frame(
            new_frame,
            r,
            dyn,
            body,
            etype,
        )

    def _rewind(self: MainWindow) -> None:  # type: ignore[override]
        idx = _active_exercise_index(self)
        if idx is None:
            self._stop_anim()
            return
        r, _fi, body, dyn = self._snapshot_idx_state(idx)
        if r is None:
            return
        self._stop_anim()
        self._set_anim_frame(idx, 0)
        _, etype = self.EXERCISE_CONFIGS[idx]
        body = _require_body_for_animation(body)
        self.exercise_tabs[idx].draw_anim_frame(
            0,
            r,
            dyn,
            body,
            etype,
        )

    def _jump_to_end(self: MainWindow) -> None:  # type: ignore[override]
        idx = _active_exercise_index(self)
        if idx is None:
            self._stop_anim()
            return
        r, _fi, body, dyn = self._snapshot_idx_state(idx)
        if r is None:
            return
        self._stop_anim()
        n = len(r.t)
        last_frame = n - 1
        self._set_anim_frame(idx, last_frame)
        _, etype = self.EXERCISE_CONFIGS[idx]
        body = _require_body_for_animation(body)
        self.exercise_tabs[idx].draw_anim_frame(
            last_frame,
            r,
            dyn,
            body,
            etype,
        )
        self.controls.set_playback_status(
            last_frame + 1,
            n,
            self.controls.speed_multiplier(),
        )

    def _on_speed(self: MainWindow, speed: float) -> None:  # type: ignore[override]
        self.controls.set_speed_multiplier_text(speed)
