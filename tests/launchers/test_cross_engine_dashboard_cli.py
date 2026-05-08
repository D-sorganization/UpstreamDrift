"""CLI / headless coverage for ``src.launchers.cross_engine_dashboard``.

The dashboard's GUI window is built lazily via ``_create_dashboard_window_class``
and requires matplotlib + Qt — exercising the full window tree is out of
scope here.  This module instead targets the pure-logic surface:

* ``_StubEngine`` reset / step / get_state behaviour
* ``_try_build_real_engine`` and ``_build_engine`` fallback to the stub
* ``_build_arg_parser`` defaults / overrides
* ``_run_headless`` end-to-end with the stub engine
* ``main`` dispatch into headless mode and into the no-engines error branch
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.launchers import cross_engine_dashboard as ced


def test_stub_engine_reset_and_step() -> None:
    eng = ced._StubEngine("s", n_dof=3)
    eng.set_control(np.array([1.0, 1.0, 1.0]))
    eng.step(0.01)
    q, qd = eng.get_state()
    assert q.shape == (3,)
    assert qd.shape == (3,)
    eng.reset()
    q2, qd2 = eng.get_state()
    np.testing.assert_array_equal(q2, np.zeros(3))
    np.testing.assert_array_equal(qd2, np.zeros(3))


def test_stub_engine_step_uses_default_dt() -> None:
    eng = ced._StubEngine("s")
    eng.set_control(np.array([0.5, 0.5]))
    # dt = None should fall back to a sensible default (no exception).
    eng.step()
    q, _ = eng.get_state()
    assert q.shape == (2,)


def test_try_build_real_engine_returns_none_on_unknown() -> None:
    assert ced._try_build_real_engine("never_an_engine_name") is None


def test_build_engine_returns_stub_for_unknown() -> None:
    eng = ced._build_engine("pendulum_stub")
    # Whatever it is, calling step / reset / get_state must work.
    eng.reset()
    eng.set_control(np.zeros(2))
    eng.step(0.01)
    q, qd = eng.get_state()
    assert q.shape == qd.shape


def test_build_arg_parser_defaults() -> None:
    parser = ced._build_arg_parser()
    args = parser.parse_args([])
    assert args.no_gui is False
    assert args.engines == "pendulum_stub"
    assert args.n_trials == 10
    assert args.amplitude == pytest.approx(0.1)
    assert args.t_end == pytest.approx(1.5)
    assert args.dt == pytest.approx(0.01)


def test_build_arg_parser_overrides() -> None:
    parser = ced._build_arg_parser()
    args = parser.parse_args(
        [
            "--no-gui",
            "--engines",
            "pendulum_stub,foo",
            "--n-trials",
            "3",
            "--amplitude",
            "0.5",
            "--t-end",
            "0.2",
            "--dt",
            "0.05",
        ]
    )
    assert args.no_gui is True
    assert "pendulum_stub" in args.engines
    assert args.n_trials == 3
    assert args.t_end == pytest.approx(0.2)


def test_run_headless_returns_cv_summary() -> None:
    config = ced.CrossEngineSimConfig(
        t_end=0.05, dt=0.01, noise_amplitude=0.0, n_trials=2
    )
    summary = ced._run_headless(["pendulum_stub"], config)
    assert "cv_total_energy_final" in summary
    assert "cv_end_effector_speed_final" in summary
    assert "cv_peak_end_effector_speed" in summary


def test_run_headless_raises_for_empty_engine_list() -> None:
    config = ced.CrossEngineSimConfig(
        t_end=0.05, dt=0.01, noise_amplitude=0.0, n_trials=1
    )
    with pytest.raises(ValueError):
        ced._run_headless([], config)


def test_main_no_engine_names_exits() -> None:
    with pytest.raises(SystemExit):
        ced.main(["--no-gui", "--engines", " , ,"])


def test_main_headless_mode_runs_without_gui() -> None:
    # Drive ``main`` through the headless path with a single tiny trial.
    ced.main(
        [
            "--no-gui",
            "--engines",
            "pendulum_stub",
            "--n-trials",
            "1",
            "--t-end",
            "0.02",
            "--dt",
            "0.01",
        ]
    )


def test_main_falls_back_to_headless_when_pyqt_missing() -> None:
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def boom(name, *a, **k):
        if name.startswith("PyQt6"):
            raise ImportError("Qt unavailable in this test")
        return real_import(name, *a, **k)

    with patch("builtins.__import__", side_effect=boom):
        ced.main(
            [
                "--engines",
                "pendulum_stub",
                "--n-trials",
                "1",
                "--t-end",
                "0.02",
                "--dt",
                "0.01",
            ]
        )


def test_create_dashboard_window_class_returns_window_object(qapp) -> None:
    # The factory returns an *instance* of the deferred window class.
    obj = ced._create_dashboard_window_class()
    assert obj is not None
    obj.deleteLater()


def test_cross_engine_dashboard_window_new_raises() -> None:
    with pytest.raises(NotImplementedError):
        ced.CrossEngineDashboardWindow()


def test_dashboard_window_on_run_with_no_engines(qapp) -> None:
    win = ced._create_dashboard_window_class()
    # Uncheck every engine and click Run — should refuse to start.
    for cb in win._engine_checks.values():
        cb.setChecked(False)
    win._on_run()
    assert "at least one" in win._status_label.text().lower()
    win.deleteLater()


def test_dashboard_window_on_run_starts_worker(qapp) -> None:
    win = ced._create_dashboard_window_class()
    # Ensure exactly one engine selected
    for name, cb in win._engine_checks.items():
        cb.setChecked(name == "pendulum_stub")
    # Patch the thread pool so the worker is captured but never run.
    started = []
    win._thread_pool = MagicMock()
    win._thread_pool.start.side_effect = lambda w: started.append(w)
    win._on_run()
    assert started, "worker should have been queued"
    assert win._status_label.text() == "Running…"
    win.deleteLater()


def test_dashboard_window_on_comparison_finished_reenables_button(qapp) -> None:
    win = ced._create_dashboard_window_class()
    win._run_btn.setEnabled(False)
    win._on_comparison_finished(
        ["pendulum_stub"],
        {
            "cv_total_energy_final": 0.1,
            "cv_end_effector_speed_final": 0.2,
            "cv_peak_end_effector_speed": 0.3,
        },
    )
    assert win._status_label.text() == "Done"
    assert win._run_btn.isEnabled()
    win.deleteLater()


def test_dashboard_window_on_comparison_error_reenables_button(qapp) -> None:
    win = ced._create_dashboard_window_class()
    win._run_btn.setEnabled(False)
    win._on_comparison_error("boom")
    assert "boom" in win._status_label.text()
    assert win._run_btn.isEnabled()
    win.deleteLater()


def test_dashboard_window_update_charts_with_empty_engines(qapp) -> None:
    win = ced._create_dashboard_window_class()
    # Should silently no-op when engine list is empty.
    win._update_charts([], {})
    win.deleteLater()


def test_dashboard_window_update_charts_redraws_canvases(qapp) -> None:
    win = ced._create_dashboard_window_class()
    cv = {
        "cv_total_energy_final": 0.1,
        "cv_end_effector_speed_final": 0.05,
        "cv_peak_end_effector_speed": 0.02,
    }
    win._update_charts(["pendulum_stub"], cv)
    win.deleteLater()
