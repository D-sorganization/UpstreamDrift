"""Tests for upstream_drift_tools.process_calculators.ode_solver (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.upstream_drift_tools.process_calculators.ode_solver import (
    ODESolver,
)


class TestODESolverConstruction:
    def test_valid_construction(self) -> None:
        solver = ODESolver({"y": "-y"}, {})
        assert solver is not None

    def test_stores_derivatives(self) -> None:
        derivs = {"T": "k*(T_env - T)"}
        params = {"k": 0.3, "T_env": 350.0}
        solver = ODESolver(derivs, params)
        assert solver.derivatives == derivs

    def test_stores_parameters(self) -> None:
        params = {"k": 0.3, "T_env": 350.0}
        solver = ODESolver({"T": "k*(T_env - T)"}, params)
        assert solver.parameters["k"] == pytest.approx(0.3)

    def test_no_parameters(self) -> None:
        solver = ODESolver({"y": "-y"}, {})
        assert solver.parameters == {}

    def test_multiple_variables(self) -> None:
        derivs = {"x": "v", "v": "-x"}
        solver = ODESolver(derivs, {})
        assert len(solver.derivatives) == 2


class TestODESolverSolve:
    def _make_decay_solver(self) -> ODESolver:
        """Simple exponential decay: dy/dt = -k*y."""
        return ODESolver({"y": "-k*y"}, {"k": 1.0})

    def _make_heating_solver(self) -> ODESolver:
        """Newton cooling: dT/dt = k*(T_env - T)."""
        return ODESolver({"T": "k*(T_env - T)"}, {"k": 0.3, "T_env": 350.0})

    def test_solve_returns_solution(self) -> None:
        solver = self._make_decay_solver()
        sol = solver.solve((0.0, 1.0), [1.0])
        assert sol is not None

    def test_solve_success(self) -> None:
        solver = self._make_decay_solver()
        sol = solver.solve((0.0, 5.0), [1.0])
        assert sol.success

    def test_solve_preserves_initial_condition(self) -> None:
        solver = self._make_decay_solver()
        t_eval = np.linspace(0.0, 5.0, 50)
        sol = solver.solve((0.0, 5.0), [1.0], t_eval=t_eval)
        # y(0) should be close to initial condition
        assert sol.y[0, 0] == pytest.approx(1.0, rel=1e-3)

    def test_exponential_decay_accuracy(self) -> None:
        """dy/dt = -y, y(0) = 1 → y(t) = e^(-t)."""
        solver = ODESolver({"y": "-y"}, {})
        t_eval = np.linspace(0.0, 2.0, 100)
        sol = solver.solve((0.0, 2.0), [1.0], t_eval=t_eval)
        expected = np.exp(-t_eval)
        np.testing.assert_allclose(sol.y[0], expected, rtol=1e-2)

    def test_heating_approaches_steady_state(self) -> None:
        """T(t) should approach T_env = 350 as t → ∞."""
        solver = self._make_heating_solver()
        t_eval = np.linspace(0.0, 50.0, 200)
        sol = solver.solve((0.0, 50.0), [300.0], t_eval=t_eval)
        # At t=50, temperature should be close to 350
        assert abs(sol.y[0, -1] - 350.0) < 1.0

    def test_harmonic_oscillator(self) -> None:
        """dx/dt = v, dv/dt = -x → solution is cos/sin."""
        solver = ODESolver({"x": "v", "v": "-x"}, {})
        t_eval = np.linspace(0.0, 2 * np.pi, 200)
        sol = solver.solve((0.0, 2 * np.pi), [1.0, 0.0], t_eval=t_eval)
        # x should return to ~1.0 after full period
        assert abs(sol.y[0, -1] - 1.0) < 0.01

    def test_t_eval_gives_output_at_specified_times(self) -> None:
        solver = self._make_decay_solver()
        t_eval = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        sol = solver.solve((0.0, 2.0), [1.0], t_eval=t_eval)
        np.testing.assert_array_equal(sol.t, t_eval)

    def test_solution_decreases_monotonically_for_decay(self) -> None:
        solver = self._make_decay_solver()
        t_eval = np.linspace(0.0, 5.0, 50)
        sol = solver.solve((0.0, 5.0), [1.0], t_eval=t_eval)
        # Exponential decay should be strictly decreasing
        diffs = np.diff(sol.y[0])
        assert np.all(diffs < 0)
