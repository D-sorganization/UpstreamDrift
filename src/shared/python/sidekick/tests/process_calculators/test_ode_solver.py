import numpy as np
import pytest
from sidekick.process_calculators.ode_solver import ODESolver


def test_ode_solver_initialization() -> None:
    derivs = {"T": "k*(T_env - T)"}
    params = {"k": 0.3, "T_env": 350.0}
    solver = ODESolver(derivs, params)

    assert list(solver.derivatives.keys()) == ["T"]
    assert solver.parameters["k"] == 0.3
    assert len(solver._functions) == 1


def test_ode_solver_solve() -> None:
    derivs = {"T": "k*(T_env - T)"}
    params = {"k": 0.3, "T_env": 350.0}
    solver = ODESolver(derivs, params)

    t_eval = np.linspace(0.0, 20.0, 100)
    sol = solver.solve((0.0, 20.0), [300.0], t_eval=t_eval)

    assert sol.success
    assert len(sol.t) == 100
    assert len(sol.y) == 1
    assert sol.y[0][0] == pytest.approx(300.0)
    assert sol.y[0][-1] > 300.0
    assert sol.y[0][-1] < 350.0


def test_ode_solver_multiple_equations() -> None:
    derivs = {"A": "-k1*A", "B": "k1*A - k2*B"}
    params = {"k1": 0.1, "k2": 0.05}
    solver = ODESolver(derivs, params)

    t_eval = np.linspace(0.0, 50.0, 50)
    sol = solver.solve((0.0, 50.0), [1.0, 0.0], t_eval=t_eval)

    assert sol.success
    assert sol.y[0][0] == 1.0
    assert sol.y[1][0] == 0.0
    # A should decrease
    assert sol.y[0][-1] < 1.0
    # B should increase then decrease or just be > 0
    assert sol.y[1][-1] > 0.0
