# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for issue #217: decomposed helpers from oversized functions.

Covers new helpers extracted to bring each target function to <= 30 LOC:

- cli._add_body_args
- cli._add_run_args
- cli._unpack_exercise_config
- cli._validate_cli_args
- cli._log_optimization_start
- cli._log_optimization_done
- LagrangianDynamics._numpy_inverse_dynamics_batch
- TrajectoryOptimizer._check_solution_feasibility
- ExerciseTab._build_grid_axes
- ExerciseTab._configure_anim_axis
- ExerciseTab._render_analysis_plots
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication

    _QT_AVAILABLE = True
    _app = QApplication.instance()
    if _app is None:
        _app = QApplication([])
except (ImportError, OSError):
    _QT_AVAILABLE = False

from movement_optimizer.cli import (
    _add_body_args,
    _add_run_args,
    _build_parser,
    _log_optimization_done,
    _log_optimization_start,
    _unpack_exercise_config,
    _validate_cli_args,
)
from movement_optimizer.models import BodyModel, make_squat_config
from movement_optimizer.models.lagrangian_dynamics import LagrangianDynamics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lagrangian(body: BodyModel | None = None) -> LagrangianDynamics:
    """Return a LagrangianDynamics built from a squat configuration."""
    if body is None:
        body = BodyModel(75.0, 1.75)
    dyn, _qs, _qe, _qb = make_squat_config(body, 60.0)
    return dyn  # type: ignore[return-value]


def _make_fresh_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser()


# ---------------------------------------------------------------------------
# cli._add_body_args
# ---------------------------------------------------------------------------


class TestAddBodyArgs:
    def test_body_mass_default(self) -> None:
        p = _make_fresh_parser()
        _add_body_args(p)
        args = p.parse_args([])
        assert args.body_mass == pytest.approx(75.0)

    def test_height_default(self) -> None:
        p = _make_fresh_parser()
        _add_body_args(p)
        args = p.parse_args([])
        assert args.height == pytest.approx(1.75)

    def test_bar_mass_default(self) -> None:
        p = _make_fresh_parser()
        _add_body_args(p)
        args = p.parse_args([])
        assert args.bar_mass == pytest.approx(60.0)

    def test_custom_body_mass(self) -> None:
        p = _make_fresh_parser()
        _add_body_args(p)
        args = p.parse_args(["--body-mass", "90"])
        assert args.body_mass == pytest.approx(90.0)

    def test_custom_height(self) -> None:
        p = _make_fresh_parser()
        _add_body_args(p)
        args = p.parse_args(["--height", "1.90"])
        assert args.height == pytest.approx(1.90)

    def test_custom_bar_mass(self) -> None:
        p = _make_fresh_parser()
        _add_body_args(p)
        args = p.parse_args(["--bar-mass", "120"])
        assert args.bar_mass == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# cli._add_run_args
# ---------------------------------------------------------------------------


class TestAddRunArgs:
    def test_duration_default(self) -> None:
        p = _make_fresh_parser()
        _add_run_args(p)
        args = p.parse_args([])
        assert args.duration == pytest.approx(2.0)

    def test_smoothness_default(self) -> None:
        p = _make_fresh_parser()
        _add_run_args(p)
        args = p.parse_args([])
        assert args.smoothness == pytest.approx(1.0)

    def test_output_default_is_none(self) -> None:
        p = _make_fresh_parser()
        _add_run_args(p)
        args = p.parse_args([])
        assert args.output is None

    def test_verbose_default_is_false(self) -> None:
        p = _make_fresh_parser()
        _add_run_args(p)
        args = p.parse_args([])
        assert args.verbose is False

    def test_custom_duration(self) -> None:
        p = _make_fresh_parser()
        _add_run_args(p)
        args = p.parse_args(["--duration", "3.5"])
        assert args.duration == pytest.approx(3.5)

    def test_verbose_flag(self) -> None:
        p = _make_fresh_parser()
        _add_run_args(p)
        args = p.parse_args(["--verbose"])
        assert args.verbose is True


# ---------------------------------------------------------------------------
# cli._unpack_exercise_config
# ---------------------------------------------------------------------------


class TestUnpackExerciseConfig:
    def test_4tuple_sets_q_via_to_none(self) -> None:
        dyn, qs, qe, qb = object(), np.zeros(3), np.ones(3), np.zeros((3, 2))
        result = _unpack_exercise_config((dyn, qs, qe, qb))
        assert result[0] is dyn
        assert result[4] is None

    def test_5tuple_preserves_q_via(self) -> None:
        dyn = object()
        q_via = np.full(3, 0.5)
        config = (dyn, np.zeros(3), np.ones(3), np.zeros((3, 2)), q_via)
        _dyn, _qs, _qe, _qb, got_via = _unpack_exercise_config(config)
        assert np.array_equal(got_via, q_via)

    def test_4tuple_unpacks_all_fields(self) -> None:
        qs = np.array([0.1, 0.2, 0.3])
        qe = np.array([0.4, 0.5, 0.6])
        qb = np.zeros((3, 2))
        dyn = object()
        out_dyn, out_qs, out_qe, out_qb, out_via = _unpack_exercise_config((dyn, qs, qe, qb))
        assert out_dyn is dyn
        assert np.array_equal(out_qs, qs)
        assert np.array_equal(out_qe, qe)
        assert np.array_equal(out_qb, qb)
        assert out_via is None

    def test_real_squat_config_is_4tuple(self) -> None:
        """make_squat_config returns a 4-tuple; _unpack sets q_via=None."""
        body = BodyModel(75.0, 1.75)
        config = make_squat_config(body, 60.0)
        _, _, _, _, q_via = _unpack_exercise_config(config)
        assert q_via is None


# ---------------------------------------------------------------------------
# cli._validate_cli_args
# ---------------------------------------------------------------------------


class TestValidateCliArgs:
    def _make_args(self, **overrides: Any) -> argparse.Namespace:
        defaults = dict(
            body_mass=75.0,
            height=1.75,
            bar_mass=60.0,
            duration=2.0,
            smoothness=1.0,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_valid_args_does_not_raise(self) -> None:
        p = _build_parser()
        args = self._make_args()
        _validate_cli_args(p, args)  # must not raise / exit

    def test_negative_body_mass_exits(self) -> None:
        p = _build_parser()
        with pytest.raises(SystemExit):
            _validate_cli_args(p, self._make_args(body_mass=-1.0))

    def test_zero_body_mass_exits(self) -> None:
        p = _build_parser()
        with pytest.raises(SystemExit):
            _validate_cli_args(p, self._make_args(body_mass=0.0))

    def test_negative_height_exits(self) -> None:
        p = _build_parser()
        with pytest.raises(SystemExit):
            _validate_cli_args(p, self._make_args(height=-0.5))

    def test_negative_bar_mass_exits(self) -> None:
        p = _build_parser()
        with pytest.raises(SystemExit):
            _validate_cli_args(p, self._make_args(bar_mass=-10.0))

    def test_zero_bar_mass_is_allowed(self) -> None:
        """bar_mass=0 is valid (bodyweight exercise)."""
        p = _build_parser()
        _validate_cli_args(p, self._make_args(bar_mass=0.0))  # should not exit

    def test_zero_duration_exits(self) -> None:
        p = _build_parser()
        with pytest.raises(SystemExit):
            _validate_cli_args(p, self._make_args(duration=0.0))

    def test_negative_duration_exits(self) -> None:
        p = _build_parser()
        with pytest.raises(SystemExit):
            _validate_cli_args(p, self._make_args(duration=-1.0))


# ---------------------------------------------------------------------------
# cli._log_optimization_start
# ---------------------------------------------------------------------------


class TestLogOptimizationStart:
    def test_emits_info_log(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="movement_optimizer.cli"):
            _log_optimization_start("squat", 80.0, 1.80, 100.0, 2.5)
        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        assert "squat" in msg
        assert "80" in msg
        assert "100" in msg
        assert "2.5" in msg

    def test_log_level_is_info(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.DEBUG, logger="movement_optimizer.cli"):
            _log_optimization_start("deadlift", 75.0, 1.75, 60.0, 2.0)
        assert caplog.records[0].levelno == logging.INFO


# ---------------------------------------------------------------------------
# cli._log_optimization_done
# ---------------------------------------------------------------------------


class TestLogOptimizationDone:
    def test_emits_info_log(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="movement_optimizer.cli"):
            _log_optimization_done(elapsed=1.23, cost=45.6, success=True)
        assert len(caplog.records) == 1
        msg = caplog.records[0].message
        assert "1.2" in msg
        assert "45.6" in msg
        assert "True" in msg

    def test_failure_flag_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="movement_optimizer.cli"):
            _log_optimization_done(elapsed=3.0, cost=999.0, success=False)
        assert "False" in caplog.records[0].message


# ---------------------------------------------------------------------------
# LagrangianDynamics._numpy_inverse_dynamics_batch
# ---------------------------------------------------------------------------


class TestNumpyInverseDynamicsBatch:
    def test_output_shape(self) -> None:
        dyn = _make_lagrangian()
        rng = np.random.default_rng(0)
        n = 15
        q = rng.random((n, 3))
        qd = rng.random((n, 3)) * 0.1
        qdd = rng.random((n, 3)) * 0.05
        tau = dyn._numpy_inverse_dynamics_batch(q, qd, qdd)
        assert tau.shape == (n, 3)

    def test_matches_full_method(self) -> None:
        """_numpy_inverse_dynamics_batch must give the same result as the public method
        (which falls back to NumPy when the Rust extension is absent)."""
        dyn = _make_lagrangian()
        rng = np.random.default_rng(7)
        n = 20
        q = rng.random((n, 3)) * 0.4
        qd = rng.random((n, 3)) * 0.2
        qdd = rng.random((n, 3)) * 0.1
        tau_direct = dyn._numpy_inverse_dynamics_batch(q, qd, qdd)
        tau_full = dyn.inverse_dynamics_batch(q, qd, qdd)
        np.testing.assert_allclose(tau_direct, tau_full, rtol=1e-10)

    def test_zero_acceleration_only_gravity(self) -> None:
        """With qd=qdd=0, torques should equal gravity contribution."""
        dyn = _make_lagrangian()
        n = 5
        q = np.full((n, 3), np.pi / 4)
        qd = np.zeros((n, 3))
        qdd = np.zeros((n, 3))
        tau = dyn._numpy_inverse_dynamics_batch(q, qd, qdd)
        g_tau = dyn._batch_gravity_torques(q)
        np.testing.assert_allclose(tau, g_tau, rtol=1e-10)

    def test_all_finite(self) -> None:
        dyn = _make_lagrangian()
        rng = np.random.default_rng(42)
        n = 30
        q = rng.random((n, 3))
        qd = rng.random((n, 3))
        qdd = rng.random((n, 3))
        tau = dyn._numpy_inverse_dynamics_batch(q, qd, qdd)
        assert np.all(np.isfinite(tau))


# ---------------------------------------------------------------------------
# TrajectoryOptimizer._check_solution_feasibility
# ---------------------------------------------------------------------------


class TestCheckSolutionFeasibility:
    def _make_squat_optimizer(self):
        from movement_optimizer.trajectory import TrajectoryOptimizer

        body = BodyModel(75.0, 1.75)
        dyn, qs, qe, qb = make_squat_config(body, 60.0)
        return (
            TrajectoryOptimizer(
                body,
                dyn,
                "squat",
                60.0,
                qs,
                qe,
                qb,
                duration=2.0,
                n_waypoints=6,
                n_eval=20,
                n_starts=1,
            ),
            body,
        )

    def _fake_res(self, cost: float):
        res = MagicMock()
        res.fun = cost
        return res

    def test_finite_cost_in_bounds_is_success(self) -> None:
        opt, body = self._make_squat_optimizer()
        # COM inside inner BOS
        com_x = np.full(20, body.inner_center)
        q = np.zeros((20, 3))
        res = self._fake_res(1.0)
        success, _n_viol = opt._check_solution_feasibility(res, q, com_x)
        assert success is True

    def test_infinite_cost_is_failure(self) -> None:
        opt, body = self._make_squat_optimizer()
        com_x = np.full(20, body.inner_center)
        q = np.zeros((20, 3))
        res = self._fake_res(float("inf"))
        success, _ = opt._check_solution_feasibility(res, q, com_x)
        assert success is False

    def test_nan_cost_is_failure(self) -> None:
        opt, body = self._make_squat_optimizer()
        com_x = np.full(20, body.inner_center)
        q = np.zeros((20, 3))
        res = self._fake_res(float("nan"))
        success, _ = opt._check_solution_feasibility(res, q, com_x)
        assert success is False

    def test_out_of_bos_is_failure(self) -> None:
        opt, body = self._make_squat_optimizer()
        # COM far outside BOS
        com_x = np.full(20, body.inner_toe + 1.0)
        q = np.zeros((20, 3))
        res = self._fake_res(1.0)
        success, _ = opt._check_solution_feasibility(res, q, com_x)
        assert success is False

    def test_returns_joint_violation_count(self) -> None:
        opt, body = self._make_squat_optimizer()
        com_x = np.full(20, body.inner_center)
        # Push q way outside bounds
        q = np.full((20, 3), 999.0)
        res = self._fake_res(1.0)
        _, n_viol = opt._check_solution_feasibility(res, q, com_x)
        assert n_viol > 0

    def test_joint_limit_violation_is_failure(self) -> None:
        # Issue #492: a finite-cost solution whose spline overshoots the
        # joint bounds must be reported as infeasible, not success.
        opt, body = self._make_squat_optimizer()
        com_x = np.full(20, body.inner_center)
        q = np.full((20, 3), 999.0)
        res = self._fake_res(1.0)
        success, n_viol = opt._check_solution_feasibility(res, q, com_x)
        assert n_viol > 0
        assert success is False

    def test_tiny_joint_overshoot_within_tolerance_is_success(self) -> None:
        # Issue #492: benign sub-tolerance overshoot must not flip success.
        from movement_optimizer.constants import JOINT_FEASIBILITY_TOL_RAD

        opt, body = self._make_squat_optimizer()
        com_x = np.full(20, body.inner_center)
        assert opt.q_bounds is not None
        upper = opt.q_bounds[:, 1]
        # Exceed the upper bound by half the tolerance on every joint.
        q = np.tile(upper + JOINT_FEASIBILITY_TOL_RAD * 0.5, (20, 1))
        res = self._fake_res(1.0)
        success, n_viol = opt._check_solution_feasibility(res, q, com_x)
        assert n_viol == 0
        assert success is True


# ---------------------------------------------------------------------------
# ExerciseTab._build_grid_axes / _configure_anim_axis / _render_analysis_plots
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _QT_AVAILABLE, reason="Qt not available")
class TestExerciseTabHelpers:
    """Smoke tests for the three newly extracted ExerciseTab helpers."""

    @pytest.fixture()
    def tab(self):  # type: ignore[no-untyped-def]
        from movement_optimizer.gui.exercise_tab import ExerciseTab

        return ExerciseTab("squat")

    def test_build_grid_axes_returns_all_keys(self, tab) -> None:  # type: ignore[no-untyped-def]
        from matplotlib.gridspec import GridSpec

        tab.fig.clear()
        gs = GridSpec(3, 4, figure=tab.fig, height_ratios=[3, 1, 1])
        axes = tab._build_grid_axes(gs)
        expected = {
            "anim",
            "com_path",
            "angles",
            "torques",
            "power",
            "com_time",
            "spine_comp",
            "spine_shear",
        }
        assert set(axes.keys()) == expected

    def test_configure_anim_axis_sets_xlim(self, tab) -> None:  # type: ignore[no-untyped-def]
        ax = tab.axes["anim"]
        tab._configure_anim_axis(ax)
        xmin, xmax = ax.get_xlim()
        assert xmin == pytest.approx(-0.9)
        assert xmax == pytest.approx(0.9)

    def test_render_analysis_plots_calls_renderers(self, tab) -> None:  # type: ignore[no-untyped-def]
        from conftest import make_test_result

        from movement_optimizer.gui import exercise_tab

        result = make_test_result()
        body = BodyModel(75.0, 1.75)
        labels = ("ankle", "knee", "hip")
        with (
            patch.object(exercise_tab.plot_renderer, "plot_angles") as m_ang,
            patch.object(exercise_tab.plot_renderer, "plot_torques") as m_tor,
            patch.object(exercise_tab.plot_renderer, "plot_power") as m_pow,
            patch.object(exercise_tab.plot_renderer, "plot_com_path") as m_com,
            patch.object(exercise_tab.plot_renderer, "plot_com_balance") as m_bal,
            patch.object(exercise_tab.plot_renderer, "plot_spine_loads") as m_spi,
        ):
            tab._render_analysis_plots(result, body, 60.0, labels)
        m_ang.assert_called_once()
        m_tor.assert_called_once()
        m_pow.assert_called_once()
        m_com.assert_called_once()
        m_bal.assert_called_once()
        m_spi.assert_called_once()
