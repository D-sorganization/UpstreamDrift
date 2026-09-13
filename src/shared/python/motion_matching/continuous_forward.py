"""One-initial-state integration for explicit Euclidean forward state equations.

Engine adapters own constrained accelerations and physical state validation.
This helper never injects target states, feedback, or intermediate resets.
"""

from collections.abc import Callable
from time import perf_counter
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from src.shared.python.motion_matching.grouped_dop853 import grouped_dop853

Array = NDArray[np.float64]


class ContinuousForwardResult(NamedTuple):
    """Read-only requested samples plus execution diagnostics."""

    time: Array
    state: Array
    evaluations: int
    elapsed_s: float


def integrate_forward(
    initial_state: Array,
    time: Array,
    derivative: Callable[[float, Array], Array],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    max_step: float = 0.001,
    max_evaluations: int | None = None,
    error_block_sizes: tuple[int, ...] | None = None,
) -> ContinuousForwardResult:
    """Integrate once from zero; return finite states at every requested time.

    The derivative receives absolute physical seconds and an internal trial
    state. It must return a finite vector of identical shape. Integration
    failure raises; partial trajectories are never returned as successful.
    max_evaluations optionally caps actual derivative calls, including rejected
    steps. Exhaustion raises before the next callback; it is not nonconvergence.
    error_block_sizes optionally partitions the state and uses the maximum of
    SciPy's DOP853 block error norms; None retains the original global norm.
    """
    initial = np.array(initial_state, dtype=float, copy=True)
    clock = np.array(time, dtype=float, copy=True)
    if initial.ndim != 1 or initial.size == 0 or not np.isfinite(initial).all():
        raise ValueError("initial_state must be a nonempty finite vector")
    if (
        clock.ndim != 1
        or clock.size < 2
        or not np.isfinite(clock).all()
        or clock[0] != 0
        or np.any(np.diff(clock) <= 0)
    ):
        raise ValueError("time must be finite and strictly increasing from zero")
    if any(not np.isfinite(value) or value <= 0 for value in (rtol, atol, max_step)):
        raise ValueError(
            "Integration tolerances and max_step must be finite and positive"
        )

    if max_evaluations is not None and (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, int)
        or max_evaluations <= 0
    ):
        raise ValueError("max_evaluations must be a positive integer or None")
    method = (
        "DOP853"
        if error_block_sizes is None
        else grouped_dop853(error_block_sizes, initial.size)
    )
    evaluations = 0

    def checked_derivative(t: float, state: Array) -> Array:
        nonlocal evaluations
        if max_evaluations is not None and evaluations >= max_evaluations:
            raise RuntimeError(
                f"Forward evaluation budget exhausted ({max_evaluations})"
            )
        evaluations += 1
        value = np.asarray(derivative(t, state), dtype=float)
        if value.shape != initial.shape or not np.isfinite(value).all():
            raise ValueError(
                "derivative must return a finite vector matching initial_state"
            )
        return value

    start = perf_counter()
    solution = solve_ivp(
        checked_derivative,
        (0.0, float(clock[-1])),
        initial,
        t_eval=clock,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    elapsed = perf_counter() - start
    if not solution.success:
        raise RuntimeError(f"Continuous forward integration failed: {solution.message}")
    states = np.asarray(solution.y.T)
    if states.shape != (clock.size, initial.size) or not np.isfinite(states).all():
        raise RuntimeError("Continuous forward integration returned incomplete states")
    clock.setflags(write=False)
    states.setflags(write=False)
    return ContinuousForwardResult(clock, states, int(solution.nfev), elapsed)
