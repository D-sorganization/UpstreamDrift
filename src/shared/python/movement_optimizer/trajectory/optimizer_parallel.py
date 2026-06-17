# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Parallel multi-start driver for the trajectory optimiser.

These free functions handle the concurrent execution of multiple SLSQP
starts and selection of the best result.  Separating them keeps
:class:`TrajectoryOptimizer` focused on setup and orchestration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Protocol

from .result import CancelledError


class _HasFun(Protocol):
    """Minimal protocol for a scipy OptimizeResult used by parallel helpers."""

    @property
    def fun(self) -> float:
        """Objective function value."""
        ...


logger = logging.getLogger(__name__)

__all__ = [
    "collect_future_results",
    "run_parallel_starts",
    "select_best_result",
]


def collect_future_results(
    pending: set[Future],
    cancel_check: Callable[[], bool],
    record_progress: Callable[[float, int], None],
) -> list[tuple[Any, int]]:
    """Drain *pending* futures, record progress, and honour cancellation.

    Parameters:
        pending: set of in-flight futures returning (res, n_evals) | None.
        cancel_check: callable returning True when cancellation is requested.
        record_progress: callable(cost_val, total_evals) for progress reporting.

    Returns:
        List of (scipy_result, n_evals) pairs from completed futures.

    Raises:
        CancelledError if cancel_check() returns True while waiting.

    Preconditions:
        All callables are thread-safe (called from the scheduling thread).

    Complexity:
        O(R) scheduler-side time and O(R) memory for ``R`` completed starts,
        excluding the worker cost of each optimization.
    """
    results: list[tuple[Any, int]] = []
    total_evals = 0

    while pending:
        if cancel_check():
            for f in pending:
                f.cancel()
            raise CancelledError("Optimization cancelled by user")

        done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)

        for future in done:
            result = future.result()
            if result is not None:
                results.append(result)
                res, n_evals = result
                total_evals += n_evals
                cost_val = float(res.fun)
                record_progress(cost_val, total_evals)

    return results


def run_parallel_starts(
    n_starts: int,
    n_workers: int,
    run_single_fn: Callable[[int], tuple[Any, int] | None],
    cancel_check: Callable[[], bool],
    record_progress: Callable[[float, int], None],
) -> list[tuple[Any, int]]:
    """Dispatch *n_starts* optimizer runs across *n_workers* threads.

    Parameters:
        n_starts: total number of random-restart seeds to run.
        n_workers: maximum thread-pool size.
        run_single_fn: callable(seed) -> (scipy_result, n_evals) or None.
        cancel_check: callable returning True when the user cancels.
        record_progress: callable(cost_val, total_evals) for live reporting.

    Returns:
        Non-None results from completed starts (may be empty if all cancelled).

    Raises:
        CancelledError if the user cancels while starts are pending.

    Preconditions:
        n_starts >= 1
        n_workers >= 1

    Complexity:
        O(n_starts) scheduling work and O(n_starts) future bookkeeping, plus the
        optimizer work performed by each submitted start.
    """
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        pending: set[Future] = {pool.submit(run_single_fn, seed) for seed in range(n_starts)}
        return collect_future_results(pending, cancel_check, record_progress)


def select_best_result(
    results: list[tuple[_HasFun, int]],
) -> tuple[_HasFun, int]:
    """Return the (scipy_result, total_evals) pair with the lowest cost.

    Preconditions:
        len(results) >= 1

    Complexity:
        O(R) time and O(1) extra memory for ``R`` completed start results.
    """
    if not results:
        raise ValueError("select_best_result requires at least one result")
    best_res, _ = min(results, key=lambda r: float(r[0].fun))
    total_evals = sum(n for _, n in results)
    return best_res, total_evals
