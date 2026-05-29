"""Tests for sidekick.process_calculators.ode_solver (ODESolver).

Covers all branches:
- __init__: derivative lambdification
- _lambdify_derivatives
- _rhs evaluation
- solve: simple IVP (exponential decay)
- plot: mocked plt.show()
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


class TestODESolver:
    """Full coverage of ODESolver class."""

    def _make_solver(self):
        from sidekick.process_calculators.ode_solver import ODESolver

        # Simple linear ODE: dT/dt = -k*(T - T_env)
        derivs = {"T": "k*(T_env - T)"}
        params = {"k": 0.3, "T_env": 350.0}
        return ODESolver(derivs, params)

    def test_init_stores_derivatives_and_parameters(self):
        solver = self._make_solver()
        assert "T" in solver.derivatives
        assert "k" in solver.parameters
        assert solver.parameters["k"] == pytest.approx(0.3)

    def test_init_creates_sympy_symbols(self):
        solver = self._make_solver()
        assert solver.t_sym is not None
        assert len(solver.var_syms) == 1  # one variable: T
        assert len(solver.param_syms) == 2  # k, T_env

    def test_init_lambdifies_derivatives(self):
        solver = self._make_solver()
        assert len(solver._functions) == 1

    def test_rhs_evaluation(self):
        solver = self._make_solver()
        # At T=300 K: dT/dt = 0.3*(350-300) = 15.0
        result = solver._rhs(0.0, [300.0])
        assert result[0] == pytest.approx(15.0, rel=1e-3)

    def test_solve_exponential_decay(self):
        solver = self._make_solver()
        t_eval = np.linspace(0.0, 10.0, 50)
        sol = solver.solve((0.0, 10.0), [300.0], t_eval=t_eval)
        assert sol.success
        # Temperature should approach T_env=350 from below
        assert sol.y[0, -1] > 300.0
        assert sol.y[0, -1] < 350.0

    def test_solve_without_t_eval(self):
        solver = self._make_solver()
        sol = solver.solve((0.0, 5.0), [300.0])
        assert sol.success
        assert len(sol.t) > 1

    def test_two_variable_system(self):
        """SHO: dx/dt = v, dv/dt = -x."""
        from sidekick.process_calculators.ode_solver import ODESolver

        derivs = {"x": "v", "v": "-x"}
        params: dict[str, float] = {}
        solver = ODESolver(derivs, params)
        sol = solver.solve((0.0, 6.28), [0.0, 1.0], t_eval=np.linspace(0, 6.28, 100))
        assert sol.success
        # x should oscillate back near 0 at t=2π
        assert abs(sol.y[0, -1]) < 0.2

    def test_plot_calls_plt_show(self):
        solver = self._make_solver()
        sol = solver.solve((0.0, 5.0), [300.0], t_eval=np.linspace(0, 5, 20))
        with patch("matplotlib.pyplot.show") as mock_show:
            solver.plot(sol)
        mock_show.assert_called_once()
