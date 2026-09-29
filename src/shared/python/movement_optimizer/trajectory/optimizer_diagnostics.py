# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Stall detection, solver callbacks, and single-start orchestration."""

from __future__ import annotations

from collections.abc import Callable

from numpy.typing import NDArray
from scipy.optimize import minimize

from .result import CancelledError
from .tuning import MAX_ITER_PER_START


def make_cancel_callback(cancel_event) -> Callable[[NDArray], None]:
    """Return an SLSQP iteration callback that raises on cancellation.

    The returned callable accepts the current iterate ``xk`` and raises
    ``CancelledError`` when ``cancel_event`` is set, aborting the solver
    immediately.
    """

    def _cancel_callback(_xk: NDArray) -> None:
        if cancel_event.is_set():
            raise CancelledError("Optimization cancelled by user")

    return _cancel_callback


def run_minimize(
    x0: NDArray,
    cost_fn: Callable[[NDArray], float],
    bounds: list[tuple[float, float]],
    constraints: list[dict],
    cancel_event,
    max_iter: int = MAX_ITER_PER_START,
) -> object:
    """Run one SLSQP solve.

    Preconditions:
        len(x0) == len(bounds).
        constraints is a list of scipy constraint dicts.

    Parameters
    ----------
    x0:
        Flattened initial waypoint guess.
    cost_fn:
        Objective function (may include eval counting).
    bounds:
        List of (lo, hi) tuples for each decision variable.
    constraints:
        List of scipy SLSQP constraint dicts.
    cancel_event:
        threading.Event used to abort the solve early.
    max_iter:
        Maximum iterations for the solver.

    Returns
    -------
    The scipy OptimizeResult object.
    """
    cancel_cb = make_cancel_callback(cancel_event)
    return minimize(
        cost_fn,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=cancel_cb,
        options={"maxiter": max_iter, "ftol": 1e-6, "disp": False},
    )


def run_single_start(
    seed: int,
    perturbed_guess_fn: Callable[[int], NDArray],
    compute_cost_fn: Callable[[NDArray], float],
    bounds: list[tuple[float, float]],
    constraints: list[dict],
    cancel_event,
) -> tuple[object, int] | None:
    """Run one SLSQP solve with a perturbed initial guess.

    Preconditions:
        seed >= 0.

    Returns ``(scipy_result, eval_count)`` or ``None`` if cancelled.
    """
    if cancel_event.is_set():
        return None

    wp0 = perturbed_guess_fn(seed)
    eval_count = [0]

    def cost_fn(x: NDArray) -> float:
        if cancel_event.is_set():
            raise CancelledError("cancelled")
        eval_count[0] += 1
        return compute_cost_fn(x)

    try:
        res = run_minimize(wp0.flatten(), cost_fn, bounds, constraints, cancel_event)
    except CancelledError:
        return None

    if cancel_event.is_set():
        return None

    return res, eval_count[0]
