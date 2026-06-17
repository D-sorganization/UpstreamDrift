# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for the OptimizationMixin worker body and its error branches."""

from __future__ import annotations

import numpy as np
import pytest

from movement_optimizer.gui.main_window import MainWindow
from movement_optimizer.trajectory import CancelledError

from .conftest import make_test_result


@pytest.fixture
def window(qapp, monkeypatch):
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.critical", lambda *a, **k: None)
    win = MainWindow()
    yield win
    win.close()


def _signal_recorder(window):
    events: dict[str, object] = {}
    window._sig_done.connect(lambda *a: events.__setitem__("done", a))
    window._sig_cancelled.connect(lambda: events.__setitem__("cancelled", True))
    window._sig_error.connect(lambda e: events.__setitem__("error", e))
    return events


def test_worker_cache_hit_emits_done(window, monkeypatch) -> None:
    events = _signal_recorder(window)
    cached = make_test_result()
    monkeypatch.setattr(window._cache, "get", lambda *a, **k: cached)

    window._opt_worker(0, None)

    assert "done" in events
    assert window.exercise_states[0].result is cached


def test_worker_success_runs_real_optimizer(window) -> None:
    events = _signal_recorder(window)
    window.sidebar.dur_slider.set_value(0.6)  # short rollout -> fast optimize

    window._opt_worker(0, None)

    assert "done" in events
    assert window.exercise_states[0].result is not None


@pytest.mark.parametrize(
    ("exc", "key"),
    [
        (CancelledError(), "cancelled"),
        (NotImplementedError("nope"), "error"),
        (np.linalg.LinAlgError("singular"), "error"),
        (ValueError("bad"), "error"),
        (RuntimeError("boom"), "error"),
    ],
)
def test_worker_error_branches_emit_signals(window, monkeypatch, exc, key) -> None:
    events = _signal_recorder(window)
    monkeypatch.setattr(window._cache, "get", lambda *a, **k: None)

    def _raise(*_a, **_k):
        raise exc

    monkeypatch.setattr(window, "_run_optimizer", _raise)

    window._opt_worker(0, None)

    assert key in events
