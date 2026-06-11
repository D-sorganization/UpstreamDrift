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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from typing import Any

from src.launchers import cross_engine_dashboard as ced
from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
    CrossEngineRunResult,
    EngineTrialMetrics,
)


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
    eng: Any = ced._build_engine("pendulum_stub")
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


def test_cv_values_and_robustness_score_match_dashboard_order() -> None:
    summary = {
        "cv_total_energy_final": 0.1,
        "cv_end_effector_speed_final": 0.2,
        "cv_peak_end_effector_speed": 0.3,
    }

    values = ced._cv_values(summary)

    assert values == [0.1, 0.2, 0.3]
    assert ced._robustness_score(values) == pytest.approx(0.8)


def test_cv_values_default_missing_metrics_to_zero() -> None:
    assert ced._cv_values({"cv_total_energy_final": 0.25}) == [0.25, 0.0, 0.0]


def test_trial_zero_trajectories_extracts_only_2d_capable_runs() -> None:
    good_traj = np.ones((3, 2))
    one_dimensional = np.ones((3,))
    too_few_columns = np.ones((3, 1))
    results = {
        "good": CrossEngineRunResult(
            "good",
            metrics_per_trial=[
                EngineTrialMetrics(1.0, 2.0, 3.0, good_traj, np.zeros((3, 2)))
            ],
        ),
        "empty": CrossEngineRunResult("empty"),
        "one_dimensional": CrossEngineRunResult(
            "one_dimensional",
            metrics_per_trial=[
                EngineTrialMetrics(1.0, 2.0, 3.0, one_dimensional, np.zeros((3,)))
            ],
        ),
        "too_few_columns": CrossEngineRunResult(
            "too_few_columns",
            metrics_per_trial=[
                EngineTrialMetrics(1.0, 2.0, 3.0, too_few_columns, np.zeros((3, 1)))
            ],
        ),
    }

    trajectories = ced._trial_zero_trajectories(results)

    assert list(trajectories) == ["good"]
    np.testing.assert_array_equal(trajectories["good"], good_traj)


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
    obj = ced._build_qt_window()
    assert obj is not None
    obj.deleteLater()


def test_cross_engine_dashboard_window_factory_returns_window(qapp) -> None:
    win = ced.CrossEngineDashboardWindow()
    assert win is not None
    win.deleteLater()


def test_load_dashboard_mpl_bindings_disables_broken_subplot(monkeypatch) -> None:
    healthy_bindings = ced._load_dashboard_mpl_bindings()
    figure_class = healthy_bindings.Figure
    original_add_subplot = figure_class.add_subplot

    def broken_add_subplot(self, *args, **kwargs):
        raise TypeError("broken subplot")

    monkeypatch.setattr(figure_class, "add_subplot", broken_add_subplot)
    try:
        bindings = ced._load_dashboard_mpl_bindings()
    finally:
        monkeypatch.setattr(figure_class, "add_subplot", original_add_subplot)

    assert bindings.has_mpl is False


def test_dashboard_window_on_run_with_no_engines(qapp) -> None:
    win = ced._build_qt_window()
    # Uncheck every engine and click Run — should refuse to start.
    for cb in win._engine_checks.values():
        cb.setChecked(False)
    win._on_run()
    assert "at least one" in win._status_label.text().lower()
    win.deleteLater()


def test_dashboard_window_on_run_starts_worker(qapp) -> None:
    win = ced._build_qt_window()
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
    win = ced._build_qt_window()
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
    win = ced._build_qt_window()
    win._run_btn.setEnabled(False)
    win._on_comparison_error("boom")
    assert "boom" in win._status_label.text()
    assert win._run_btn.isEnabled()
    win.deleteLater()


def test_dashboard_window_update_charts_with_empty_engines(qapp) -> None:
    win = ced._build_qt_window()
    # Should silently no-op when engine list is empty.
    win._update_charts([], {})
    win.deleteLater()


def test_dashboard_window_update_charts_redraws_canvases(qapp) -> None:
    win = ced._build_qt_window()
    cv = {
        "cv_total_energy_final": 0.1,
        "cv_end_effector_speed_final": 0.05,
        "cv_peak_end_effector_speed": 0.02,
    }
    win._update_charts(["pendulum_stub"], cv)
    win.deleteLater()
