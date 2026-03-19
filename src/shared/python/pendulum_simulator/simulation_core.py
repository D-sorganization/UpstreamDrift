"""
Shared ODE integration core for all pendulum simulation runners.

Extracts the common ODE scaffolding (time grid, solve_ivp call, result
validation) that was previously duplicated across simulation.py,
simulation_triple.py, and simulation_golfer.py.

Design by Contract
------------------
- integrate_ode() requires a callable ode_rhs, finite initial state, positive t_end.
- Post: returned (t, states) has shape (N,) and (N, state_dim) with N >= 2.

DRY
---
Single integration function replaces three near-identical solve_ivp calls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


def integrate_ode(
    ode_rhs: Callable[[float, np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    t_end: float,
    dt: float = 0.005,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate an ODE system with uniform output grid.

    Parameters
    ----------
    ode_rhs : callable(t, y) -> dy/dt
        Right-hand side of the ODE system.
    initial_state : np.ndarray
        Initial conditions vector.
    t_end : float
        End time for integration.
    dt : float
        Output time step (solver adapts internally).
    method : str
        ODE solver method (RK45, DOP853, etc.).
    rtol, atol : float
        Solver relative and absolute tolerances.
    max_step : float or None
        Maximum internal step size. None lets solver choose adaptively.

    Returns
    -------
    (t, states) : tuple[np.ndarray, np.ndarray]
        t has shape (N,), states has shape (N, state_dim).

    Raises
    ------
    RuntimeError
        If integration fails.

    Design by Contract
    ------------------
    Pre: initial_state is finite, t_end > 0, 0 < dt < t_end.
    Post: t has N >= 2 points, states are all finite.
    """
    assert np.all(np.isfinite(initial_state)), "Initial state must be finite"
    assert t_end > 0, f"t_end must be positive, got {t_end}"
    assert 0 < dt < t_end, f"dt must be in (0, t_end), got {dt}"

    t_eval = np.arange(0.0, t_end, dt)

    kwargs: dict = {
        "t_span": (0.0, t_end),
        "y0": initial_state,
        "t_eval": t_eval,
        "method": method,
        "rtol": rtol,
        "atol": atol,
    }
    if max_step is not None:
        kwargs["max_step"] = max_step

    sol = solve_ivp(ode_rhs, **kwargs)

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    t = sol.t
    states = sol.y.T

    assert len(t) >= 2, "Simulation must produce at least 2 time points"
    return t, states
