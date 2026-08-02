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


@pytest.mark.scientific
def test_exp_decay_matches_closed_form_reference() -> None:
    """dy/dt = -2y has the exact solution y(1) = exp(-2) (#3391)."""
    solver = ODESolver({"y": "-k*y"}, {"k": 2.0})

    sol = solver.solve(
        (0.0, 1.0),
        [1.0],
        t_eval=[0.0, 1.0],
        rtol=1e-10,
        atol=1e-12,
    )

    assert sol.success
    assert sol.y[0, -1] == pytest.approx(np.exp(-2.0), rel=1e-8)


@pytest.mark.scientific
def test_harmonic_oscillator_energy_drift_reference() -> None:
    """x'' = -x conserves 0.5*(x^2 + v^2) over 100 periods (#3391)."""
    solver = ODESolver({"x": "v", "v": "-x"}, {})
    t_end = 100.0 * 2.0 * np.pi
    t_eval = np.linspace(0.0, t_end, 2001)

    sol = solver.solve(
        (0.0, t_end),
        [1.0, 0.0],
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    assert sol.success
    energy = 0.5 * (sol.y[0] ** 2 + sol.y[1] ** 2)
    assert np.max(np.abs(energy - energy[0])) < 1e-7
