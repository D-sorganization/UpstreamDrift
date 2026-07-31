# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for OptimizationMixin controller logic (signal handlers, no real threads)."""

from __future__ import annotations

import dataclasses

import pytest

from movement_optimizer.errors import OptimizationError
from movement_optimizer.gui.main_window import MainWindow
from movement_optimizer.trajectory.result import ProgressReport

from .conftest import make_test_result


@pytest.fixture
def window(qapp, monkeypatch):
    # Neutralise the modal error dialog so _on_err does not block.
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.critical", lambda *a, **k: None)
    win = MainWindow()
    yield win
    win.close()


def test_resolve_exercise_params_populates_shared_state(window) -> None:
    body, dyn, etype, bar, _dur, _smoothness = window._resolve_exercise_params(0)
    assert window.exercise_states[0].dynamics is dyn
    assert window.exercise_states[0].body is body
    assert etype == "squat"
    assert bar > 0


def test_resolve_exercise_params_enforces_min_duration(window) -> None:
    # full_squat (idx 1) has a 3.0s minimum duration floor.
    window.sidebar.dur_slider.set_value(1.0)
    _body, _dyn, etype, _bar, dur, _smoothness = window._resolve_exercise_params(1)
    assert etype == "full_squat"
    assert dur >= 3.0


def test_snapshot_and_set_anim_frame(window) -> None:
    window._set_anim_frame(0, 5)
    _result, frame, _body, _dyn = window._snapshot_idx_state(0)
    assert frame == 5
    assert window._seg_mults().keys() == {"lower_leg", "upper_leg", "torso"}


def test_progress_cb_emits_and_updates(window) -> None:
    report = ProgressReport(
        iteration=250, cost=5.0, best_cost=4.0, improvement_pct=2.0, elapsed_s=1.0
    )
    cb = window._make_progress_cb()
    cb(report)  # emits _sig_progress -> _update_progress -> sidebar
    window._update_progress(report)
    assert "Converging" in window.sidebar.prog_label.text()


def test_on_cancelled_resets_state(window) -> None:
    window._opt_running = True
    window._on_cancelled()
    assert not window._opt_running
    assert "cancelled" in window.status_label.text().lower()


def test_on_err_with_structured_and_plain_errors(window) -> None:
    window._opt_running = True
    window._on_err(
        OptimizationError("boom", error_code="OPT_X", suggestion="try again")
    )
    assert "OPT_X" in window.status_label.text()
    window._on_err("plain failure")
    assert "plain failure" in window.status_label.text()
    assert not window._opt_running


def test_update_result_summary_both_branches(window) -> None:
    result = make_test_result()
    window._update_result_summary("Squat", result, exercise_type="squat")
    assert "Balance" in window.sidebar.result_label.text()
    window._update_result_summary("Bench Press", result, exercise_type="bench_press")
    assert "Shoulder" in window.sidebar.result_label.text()


def test_on_done_success_and_warning_paths(window) -> None:
    window._resolve_exercise_params(0)
    body = window.exercise_states[0].body
    window._opt_running = True
    window._on_done(0, make_test_result(), body, 60.0, None)
    assert not window._opt_running  # finished (no chain)

    failed = dataclasses.replace(make_test_result(), success=False)
    window._opt_running = True
    window._on_done(0, failed, body, 60.0, None)
    assert not window._opt_running


def test_finish_or_chain_advances_then_chain(window, monkeypatch) -> None:
    calls: list[tuple[int, list[int] | None]] = []
    monkeypatch.setattr(
        window, "_run_exercise", lambda idx, rest=None: calls.append((idx, rest))
    )
    window._finish_or_chain([1, 2], "msg")
    assert calls == [(1, [2])]


def test_reset_clears_cache_and_status(window) -> None:
    window._reset()
    assert "Defaults restored" in window.status_label.text()
