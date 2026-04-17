"""ode_solver.py module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate  # Added to fix F821
import sympy as sp
from scipy.integrate import solve_ivp


class ODESolver:
    """Numerical solver for systems of ODEs defined symbolically."""

    def __init__(
        self, derivatives: dict[str, str], parameters: dict[str, float]
    ) -> None:
        """Create an ODESolver.

        Parameters
        ----------
        derivatives : dict
            Mapping of variable name to symbolic expression for its derivative.
            Expressions may reference time ``t``, any variable in ``derivatives``
            and parameters provided in ``parameters``.
        parameters : dict
            Mapping of parameter name to numerical value.

        """
        if derivatives is None:
            raise ValueError("derivatives must be provided")
        self.derivatives = derivatives
        self.parameters = parameters

        # Setup sympy symbols for variables and parameters
        self.t_sym = sp.symbols("t")
        self.var_syms = [sp.symbols(v) for v in derivatives]
        self.param_syms = [sp.symbols(p) for p in parameters]

        self._functions = self._lambdify_derivatives()

    def _lambdify_derivatives(self) -> list[sp.Lambda]:
        """Convert symbolic derivatives to callable functions.

        Returns:
            List of lambda functions for each derivative expression.
        """
        funcs = []
        for expr in self.derivatives.values():
            sym_expr = sp.sympify(expr)
            syms = [self.t_sym, *self.var_syms, *self.param_syms]
            funcs.append(sp.lambdify(syms, sym_expr, modules="numpy"))
        return funcs

    def _rhs(self, t: float, y: Sequence[float]) -> list[float]:
        """Right-hand side function for ODE system.

        Args:
            t: Current time value
            y: Current state vector

        Returns:
            List of derivatives for each variable.
        """
        if t is None:
            raise ValueError("t must be provided")
        args = [t] + list(y) + [self.parameters[p] for p in self.parameters]
        return [func(*args) for func in self._functions]

    def solve(
        self,
        t_span: Sequence[float],
        y0: Sequence[float],
        t_eval: Any = None,
        **kwargs: Any,
    ) -> scipy.integrate.OdeSolution:
        """Solve the ODE system.

        Parameters
        ----------
        t_span : sequence
            Time interval as ``(t_start, t_end)``.
        y0 : sequence
            Initial values for the variables (same order as ``derivatives``).
        t_eval : sequence, optional
            Times at which to store the computed solution.
        kwargs : dict
            Additional options forwarded to :func:`scipy.integrate.solve_ivp`.

        """
        return solve_ivp(
            self._rhs,
            t_span,
            y0,
            t_eval=t_eval,
            vectorized=False,
            **kwargs,
        )

    def plot(self, solution: Any) -> None:
        """Plot the solution returned by :meth:`solve`."""
        for idx, name in enumerate(self.derivatives.keys()):
            plt.plot(solution.t, solution.y[idx], label=name)
        plt.xlabel("t")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage: heating vessel
    derivs = {"T": "k*(T_env - T)"}
    params = {"k": 0.3, "T_env": 350.0}
    solver = ODESolver(derivs, params)
    t_eval = np.linspace(0.0, 20.0, 100)
    sol = solver.solve((0.0, 20.0), [300.0], t_eval=t_eval)
    solver.plot(sol)
