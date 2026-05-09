"""Tests for upstream_drift_tools.process_calculators.ode_solver (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.upstream_drift_tools.process_calculators.ode_solver import (
    ODESolver,
)


class TestODESolverConstruction:
    def test_ode_solver_construction(self) -> None:
        solver = ODESolver({"T": "k*(T_env - T)"}, {"k": 0.3, "T_env": 350.0})
        assert solver is not None

    def test_stores_derivatives(self) -> None:
        solver = ODESolver({"T": "k*(T_env - T)"}, {"k": 0.3, "T_env": 350.0})
        assert "T" in solver.derivatives

    def test_stores_parameters(self) -> None:
        solver = ODESolver({"T": "k*(T_env - T)"}, {"k": 0.3, "T_env": 350.0})
        assert solver.parameters["k"] == pytest.approx(0.3)


class TestODESolverSolve:
    def setup_method(self) -> None:
        # Exponential decay: dT/dt = -k * T → T(t) = T0 * exp(-k*t)
        # Wait — requires T_env to be 0, or rewrite as dT/dt = -k * T
        self.method = ODESolver({"T": "k*(T_env - T)"}, {"k": 0.5, "T_env": 0.0})

    def test_solve_returns_solution(self) -> None:
        sol = self.method.solve((0.0, 5.0), [100.0])
        assert sol is not None

    def test_solution_has_expected_keys(self) -> None:
        sol = self.method.solve((0.0, 5.0), [100.0])
        assert hasattr(sol, "t")
        assert hasattr(sol, "y")

    def test_exponential_decay(self) -> None:
        # T(t) = T0 * exp(-k*t)
        t_eval = np.linspace(0.0, 4.0, 50)
        sol = self.method.solve((0.0, 4.0), [100.0], t_eval=t_eval)
        T0 = 100.0
        k = 0.5
        expected = T0 * np.exp(-k * sol.t)
        np.testing.assert_allclose(sol.y[0], expected, rtol=1e-2)

    def test_initial_condition_satisfied(self) -> None:
        t_eval = np.array([0.0, 1.0, 2.0])
        sol = self.method.solve((0.0, 2.0), [50.0], t_eval=t_eval)
        assert sol.y[0, 0] == pytest.approx(50.0, rel=1e-3)

    def test_decays_to_zero(self) -> None:
        # With T_env=0, T decays toward 0
        t_eval = np.linspace(0.0, 20.0, 100)
        sol = self.method.solve((0.0, 20.0), [100.0], t_eval=t_eval)
        assert sol.y[0, -1] < 1.0  # Nearly zero after long time

    def test_two_variable_system(self) -> None:
        # dx/dt = -y, dy/dt = x (simple harmonic oscillator)
        solver = ODESolver({"x": "-y", "y": "x"}, {})
        t_eval = np.linspace(0.0, 2 * np.pi, 200)
        sol = solver.solve((0.0, 2 * np.pi), [1.0, 0.0], t_eval=t_eval)
        # x(t) = cos(t), y(t) = sin(t)
        np.testing.assert_allclose(sol.y[0], np.cos(sol.t), atol=0.02)

    def test_solution_shape(self) -> None:
        t_eval = np.linspace(0.0, 5.0, 30)
        sol = self.method.solve((0.0, 5.0), [10.0], t_eval=t_eval)
        assert sol.y.shape[0] == 1  # 1 variable
        assert sol.y.shape[1] == len(t_eval)
