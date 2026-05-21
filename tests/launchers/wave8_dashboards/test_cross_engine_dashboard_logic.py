"""Non-GUI logic tests for ``src.launchers.cross_engine_dashboard``.

These tests target the engine-registry helpers, palette/style construction,
headless runner, argparse builder, and main() routing.  Qt / matplotlib /
real-engine imports are mocked out heavily.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.launchers import cross_engine_dashboard as ced
from src.shared.python.pendulum_simulator.cross_engine_perturbation import (
    CrossEngineSimConfig,
)
from src.shared.python.plot_style import (
    MarkerShape,
    MarkerStyle,
    PaletteColor,
    StaticColor,
)

# ---------------------------------------------------------------------------
# _StubEngine
# ---------------------------------------------------------------------------


class TestStubEngine:
    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ced._StubEngine("")

    def test_default_state_zero(self) -> None:
        eng = ced._StubEngine("foo", n_dof=3)
        q, v = eng.get_state()
        assert q.shape == (3,)
        assert v.shape == (3,)
        assert np.all(q == 0.0)
        assert np.all(v == 0.0)

    def test_set_control_applies_to_velocity(self) -> None:
        eng = ced._StubEngine("foo", n_dof=2)
        eng.set_control(np.array([10.0, 20.0]))
        _, v = eng.get_state()
        # gain 0.01
        assert v[0] == pytest.approx(0.1)
        assert v[1] == pytest.approx(0.2)

    def test_set_control_truncates_to_n_dof(self) -> None:
        eng = ced._StubEngine("foo", n_dof=2)
        eng.set_control(np.array([1.0, 2.0, 3.0, 4.0]))
        _, v = eng.get_state()
        assert v.shape == (2,)

    def test_step_integrates_with_default_dt(self) -> None:
        eng = ced._StubEngine("foo", n_dof=1)
        eng.set_control(np.array([100.0]))  # v = 1.0
        q_before, _ = eng.get_state()
        eng.step()
        q_after, v_after = eng.get_state()
        assert q_after[0] > q_before[0]
        # velocity is damped
        assert v_after[0] < 1.0

    def test_step_custom_dt(self) -> None:
        eng = ced._StubEngine("foo", n_dof=1)
        eng.set_control(np.array([100.0]))
        eng.step(dt=0.1)
        q, _ = eng.get_state()
        assert q[0] == pytest.approx(0.1)

    def test_reset(self) -> None:
        eng = ced._StubEngine("foo", n_dof=2)
        eng.set_control(np.array([5.0, 5.0]))
        eng.step()
        eng.reset()
        q, v = eng.get_state()
        assert np.all(q == 0.0)
        assert np.all(v == 0.0)

    def test_get_state_returns_copies(self) -> None:
        eng = ced._StubEngine("foo", n_dof=2)
        q, v = eng.get_state()
        q[0] = 999.0
        v[0] = 999.0
        q2, v2 = eng.get_state()
        assert q2[0] == 0.0
        assert v2[0] == 0.0


# ---------------------------------------------------------------------------
# palette / style helpers
# ---------------------------------------------------------------------------


class TestEnginePalette:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("drake", 0),
            ("mujoco", 1),
            ("pinocchio", 2),
            ("opensim", 3),
            ("simscape", 4),
            ("pendulum_stub", 7),
        ],
    )
    def test_known_engines(self, name: str, expected: int) -> None:
        assert ced._engine_palette_index(name) == expected

    def test_unknown_engine_wraps_modulo_10(self) -> None:
        idx = ced._engine_palette_index("some_unknown_engine")
        assert 0 <= idx < 10


class TestBuildEngineMarkerStyle:
    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ced._build_engine_marker_style("")

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ValueError):
            ced._build_engine_marker_style(123)  # type: ignore[arg-type]

    def test_default_fallback_no_template(self) -> None:
        style = ced._build_engine_marker_style("drake", template=None)
        assert isinstance(style, MarkerStyle)
        assert style.shape == MarkerShape.SPHERE
        assert isinstance(style.fill_color, PaletteColor)
        assert style.fill_color.palette_index == 0
        assert style.size_px == 6.0

    def test_shape_per_engine_false_uses_sphere(self) -> None:
        style = ced._build_engine_marker_style(
            "mujoco", shape_per_engine=False, template=None
        )
        assert style.shape == MarkerShape.SPHERE

    def test_shape_per_engine_true_distinct(self) -> None:
        s_drake = ced._build_engine_marker_style("drake", template=None)
        s_mujoco = ced._build_engine_marker_style("mujoco", template=None)
        assert s_drake.shape != s_mujoco.shape

    def test_uses_template_attributes(self) -> None:
        template = MarkerStyle(
            shape=MarkerShape.SPHERE,
            size_px=10.0,
            edge_color="#abcdef",
            edge_width=2.0,
            fill_color=StaticColor(hex_value="#ffffff"),
            opacity=0.5,
        )
        style = ced._build_engine_marker_style("drake", template=template)
        assert style.size_px == 10.0
        assert style.edge_color == "#abcdef"
        assert style.edge_width == 2.0
        assert style.opacity == 0.5


def test_default_marker_style_template_handles_missing(monkeypatch) -> None:
    # Force PresetLibrary.default() to raise so we hit the fallback path.
    monkeypatch.setattr(
        ced.PresetLibrary,
        "default",
        classmethod(lambda cls: (_ for _ in ()).throw(RuntimeError("nope"))),
    )
    assert ced._default_marker_style_template() is None


def test_default_marker_style_template_missing_preset_key(monkeypatch) -> None:
    fake_lib = MagicMock()
    fake_lib.__contains__ = lambda self, k: False
    monkeypatch.setattr(
        ced.PresetLibrary,
        "default",
        classmethod(lambda cls: fake_lib),
    )
    assert ced._default_marker_style_template() is None


def test_build_dashboard_style_set_returns_expected_entries() -> None:
    styles = ced.build_dashboard_style_set(["drake", "mujoco"])
    assert len(styles.entries) == 2
    names = [e.name for e in styles.entries]
    targets = [e.target for e in styles.entries]
    assert names == ["drake", "mujoco"]
    assert targets == ["trace:drake", "trace:mujoco"]


def test_build_dashboard_style_set_default_engine_names() -> None:
    styles = ced.build_dashboard_style_set()
    names = [e.name for e in styles.entries]
    assert "pendulum_stub" in names


# ---------------------------------------------------------------------------
# _render_trajectory_overlay
# ---------------------------------------------------------------------------


class TestRenderTrajectoryOverlay:
    def _make_renderer(self, ax):
        renderer = MagicMock()
        renderer._default_ax = ax
        renderer.add_markers = MagicMock(
            side_effect=lambda pos, st, label: f"h_{label}"
        )
        return renderer

    def test_empty_trajectories_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ced._render_trajectory_overlay(MagicMock(), {}, MagicMock())

    def test_renderer_axes_mismatch_raises(self) -> None:
        ax = object()
        renderer = MagicMock()
        renderer._default_ax = object()  # different
        with pytest.raises(RuntimeError, match="must be bound"):
            ced._render_trajectory_overlay(ax, {"drake": np.zeros((3, 2))}, renderer)

    def test_bad_shape_raises(self) -> None:
        ax = object()
        renderer = self._make_renderer(ax)
        with pytest.raises(ValueError, match="shape"):
            ced._render_trajectory_overlay(ax, {"drake": np.zeros((3,))}, renderer)

    def test_single_dim_raises(self) -> None:
        ax = object()
        renderer = self._make_renderer(ax)
        with pytest.raises(ValueError, match="shape"):
            ced._render_trajectory_overlay(ax, {"drake": np.zeros((3, 1))}, renderer)

    def test_returns_one_handle_per_engine(self) -> None:
        ax = object()
        renderer = self._make_renderer(ax)
        trajs = {
            "drake": np.array([[0.0, 1.0], [2.0, 3.0]]),
            "mujoco": np.array([[4.0, 5.0], [6.0, 7.0]]),
        }
        handles = ced._render_trajectory_overlay(ax, trajs, renderer)
        assert set(handles.keys()) == {"drake", "mujoco"}
        assert handles["drake"] == "h_drake"
        assert renderer.add_markers.call_count == 2


# ---------------------------------------------------------------------------
# Engine builders
# ---------------------------------------------------------------------------


class TestBuildEngine:
    def test_pendulum_stub_returns_stub(self) -> None:
        eng = ced._build_engine("pendulum_stub")
        assert isinstance(eng, ced._StubEngine)

    def test_unknown_engine_returns_stub(self) -> None:
        with patch.object(ced, "_try_build_real_engine", return_value=None):
            eng = ced._build_engine("unknown")
            assert isinstance(eng, ced._StubEngine)

    def test_returns_real_engine_when_available(self) -> None:
        sentinel = object()
        with patch.object(ced, "_try_build_real_engine", return_value=sentinel):
            assert ced._build_engine("drake") is sentinel

    def test_try_build_unknown_returns_none(self) -> None:
        # Names not handled fall through to None.
        assert ced._try_build_real_engine("not_a_real_engine") is None

    def test_try_build_handles_import_error(self) -> None:
        # Force an ImportError path by patching builtins.__import__.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if "physics_engine" in name:
                raise ImportError("forced")
            return real_import(name, *a, **kw)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            assert ced._try_build_real_engine("mujoco") is None
            assert ced._try_build_real_engine("drake") is None
            assert ced._try_build_real_engine("pinocchio") is None


# ---------------------------------------------------------------------------
# _run_with_results / _run_headless
# ---------------------------------------------------------------------------


def _fast_config() -> CrossEngineSimConfig:
    return CrossEngineSimConfig(t_end=0.05, dt=0.01, noise_amplitude=0.05, n_trials=2)


class TestRunWithResults:
    def test_empty_engine_list_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            ced._run_with_results([], _fast_config())

    def test_returns_results_and_cv_summary(self) -> None:
        results, cv = ced._run_with_results(["pendulum_stub"], _fast_config())
        assert "pendulum_stub" in results
        assert "cv_total_energy_final" in cv
        assert "cv_end_effector_speed_final" in cv
        assert "cv_peak_end_effector_speed" in cv


class TestRunHeadless:
    def test_logs_and_returns_cv_summary(self, caplog) -> None:
        caplog.set_level(logging.INFO, logger=ced.logger.name)
        cv = ced._run_headless(["pendulum_stub"], _fast_config())
        assert isinstance(cv, dict)
        assert "cv_total_energy_final" in cv
        assert any("Cross-Engine" in r.message for r in caplog.records)
        assert any("CV Summary" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


class TestArgParser:
    def test_defaults(self) -> None:
        parser = ced._build_arg_parser()
        args = parser.parse_args([])
        assert args.no_gui is False
        assert args.engines == "pendulum_stub"
        assert args.n_trials == 10
        assert args.amplitude == 0.1
        assert args.t_end == 1.5
        assert args.dt == 0.01
        assert args.shape_per_engine is True

    def test_no_shape_per_engine_flag(self) -> None:
        parser = ced._build_arg_parser()
        args = parser.parse_args(["--no-shape-per-engine"])
        assert args.shape_per_engine is False

    def test_no_gui_flag(self) -> None:
        parser = ced._build_arg_parser()
        args = parser.parse_args(["--no-gui"])
        assert args.no_gui is True

    def test_engines_csv(self) -> None:
        parser = ced._build_arg_parser()
        args = parser.parse_args(["--engines", "drake,mujoco"])
        assert args.engines == "drake,mujoco"


# ---------------------------------------------------------------------------
# main() routing
# ---------------------------------------------------------------------------


class TestMain:
    def test_no_engines_exits(self) -> None:
        with pytest.raises(SystemExit):
            ced.main(["--no-gui", "--engines", ","])

    def test_no_gui_runs_headless(self) -> None:
        with patch.object(ced, "_run_headless") as mock_run:
            ced.main(
                [
                    "--no-gui",
                    "--engines",
                    "pendulum_stub",
                    "--n-trials",
                    "2",
                    "--t-end",
                    "0.05",
                ]
            )
            mock_run.assert_called_once()
            args, _ = mock_run.call_args
            assert args[0] == ["pendulum_stub"]
            assert isinstance(args[1], CrossEngineSimConfig)
            assert args[1].n_trials == 2

    def test_gui_falls_back_to_headless_when_qt_missing(self) -> None:
        # Simulate PyQt6 import failure by patching the builtin import to raise
        # only for PyQt6.QtWidgets.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "PyQt6.QtWidgets":
                raise ImportError("no qt")
            return real_import(name, *a, **kw)

        with (
            patch.object(builtins, "__import__", side_effect=fake_import),
            patch.object(ced, "_run_headless") as mock_run,
        ):
            ced.main(["--engines", "pendulum_stub"])
            mock_run.assert_called_once()

    def test_gui_path_calls_build_qt_window(self) -> None:
        fake_window = MagicMock()
        fake_app = MagicMock()
        fake_app.exec.return_value = 0
        with (
            patch.object(ced, "_build_qt_window", return_value=fake_window) as mb,
            patch("PyQt6.QtWidgets.QApplication") as qapp_cls,
        ):
            qapp_cls.instance.return_value = fake_app
            with pytest.raises(SystemExit):
                ced.main(["--engines", "pendulum_stub"])
            mb.assert_called_once()
            fake_window.show.assert_called_once()


# ---------------------------------------------------------------------------
# _build_qt_window — just verify it delegates to the class factory
# ---------------------------------------------------------------------------


def test_build_qt_window_invokes_class_factory() -> None:
    sentinel_cls = MagicMock(return_value="window_instance")
    with patch.object(ced, "_create_dashboard_window_class", return_value=sentinel_cls):
        result = ced._build_qt_window(shape_per_engine=False)
        sentinel_cls.assert_called_once_with(shape_per_engine=False)
        assert result == "window_instance"


def test_cross_engine_dashboard_window_direct_instantiation_blocked() -> None:
    with pytest.raises(NotImplementedError, match="create_window"):
        ced.CrossEngineDashboardWindow()
