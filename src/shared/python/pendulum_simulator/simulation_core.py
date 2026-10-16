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
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationConfig:
    """Optional run settings shared by the double/triple/golfer runners.

    Every field defaults to ``None`` meaning "use the runner's own default",
    so one parameter object serves all three ``run_simulation`` variants
    (DRY, #12844). Runners only accept the subset they understand; see
    :meth:`resolve`.

    Design by Contract
    ------------------
    - Immutable: a config can be shared safely between runs and threads.
    - ``resolve`` raises ``TypeError`` for unknown names or for settings the
      calling runner does not support (matching the old keyword signatures).
    """

    rtol: float | None = None
    atol: float | None = None
    alpha: float | None = None
    beta: float | None = None
    torque_limits: Any = None
    limits: Any = None
    clamp: Any = None
    coeffs: Sequence[float] | None = None
    n_coeffs_per_joint: int | None = None

    @classmethod
    def resolve(
        cls,
        config: SimulationConfig | None,
        overrides: Mapping[str, Any],
        supported: Collection[str],
    ) -> SimulationConfig:
        """Merge keyword ``overrides`` onto ``config`` and validate them.

        Pre: every override name is a ``SimulationConfig`` field, and every
        non-``None`` setting is in ``supported``.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise TypeError(f"run_simulation() got unexpected keyword {unknown[0]!r}")
        if config is not None and not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig or None")
        merged = replace(config or cls(), **overrides)
        for name in sorted(known - set(supported)):
            if getattr(merged, name) is not None:
                raise TypeError(f"run_simulation() got unexpected keyword {name!r}")
        return merged

    @property
    def effective_torque_limits(self) -> Any:
        """``torque_limits`` if given, else the legacy ``clamp`` alias."""
        return self.torque_limits if self.torque_limits is not None else self.clamp

    def solver_tolerances(self, rtol: float, atol: float) -> tuple[float, float]:
        """Return ``(rtol, atol)`` with this config's overrides applied."""
        return (
            rtol if self.rtol is None else self.rtol,
            atol if self.atol is None else self.atol,
        )


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
    if not (np.all(np.isfinite(initial_state))):
        raise ValueError("Initial state must be finite")
    if not (t_end > 0):
        raise ValueError(f"t_end must be positive, got {t_end}")
    if not (0 < dt < t_end):
        raise ValueError(f"dt must be in (0, t_end), got {dt}")

    t_eval = np.arange(0.0, t_end, dt)
    if t_eval.size == 0 or not np.isclose(t_eval[-1], t_end):
        t_eval = np.concatenate((t_eval, np.asarray([t_end], dtype=np.float64)))

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

    if not (len(t) >= 2):
        raise ValueError("Simulation must produce at least 2 time points")
    return t, states
