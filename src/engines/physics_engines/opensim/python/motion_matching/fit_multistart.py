"""Deterministic multistart orchestration for the OpenSim fit driver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.cost import SimOutput

from .fit_swing import FitOptions, FitResult, fit_swing_opensim

StartFailure = dict[str, Any]
StartRunner = Callable[[ClubTarget, FitOptions], FitResult]
SimulateFactory = Callable[[int, int], Callable[[NDArray[np.float64]], Any]]


__all__ = [
    "AllStartsFailedError",
    "MultistartOptions",
    "fit_swing_opensim_multistart",
    "generate_multistart_seeds",
]


@dataclass(frozen=True)
class MultistartOptions:
    """Options for deterministic OpenSim multistart fitting."""

    n_starts: int = 4
    base_seed: int = 42
    max_workers: int = 1
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if self.n_starts < 1:
            raise ValueError(
                f"MultistartOptions.n_starts must be >= 1, got {self.n_starts}"
            )
        if self.base_seed < 0:
            raise ValueError(
                f"MultistartOptions.base_seed must be non-negative, got {self.base_seed}"
            )
        if self.max_workers < 1:
            raise ValueError(
                f"MultistartOptions.max_workers must be >= 1, got {self.max_workers}"
            )


class AllStartsFailedError(RuntimeError):
    """Raised when every deterministic start fails."""

    def __init__(self, failures: tuple[StartFailure, ...]) -> None:
        self.failures = failures
        super().__init__(
            f"All {len(failures)} OpenSim multistart attempts failed: "
            + "; ".join(
                f"start {failure['start_index']} seed {failure['seed']}: "
                f"{failure['error_type']}: {failure['message']}"
                for failure in failures
            )
        )


def generate_multistart_seeds(*, n_starts: int, base_seed: int) -> tuple[int, ...]:
    """Generate an ordered deterministic seed list from ``base_seed``."""
    if n_starts < 1:
        raise ValueError(f"n_starts must be >= 1, got {n_starts}")
    if base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}")
    rng = np.random.default_rng(base_seed)
    return tuple(
        int(seed) for seed in rng.integers(0, 2**32, size=n_starts, dtype=np.uint32)
    )


def fit_swing_opensim_multistart(
    target: ClubTarget,
    *,
    fit_options: FitOptions | None = None,
    multistart_options: MultistartOptions | None = None,
    start_runner: StartRunner | None = None,
    simulate_factory: SimulateFactory | None = None,
) -> FitResult:
    """Run deterministic OpenSim fit starts and return the best success.

    The wrapper only orchestrates starts. It delegates the single-start fit to
    :func:`fit_swing_opensim`, ranks successful :class:`FitResult` objects by
    ``final_cost``, and records start diagnostics in ``result.meta``.
    """
    if fit_options is None:
        fit_options = FitOptions()
    if multistart_options is None:
        multistart_options = MultistartOptions()
    runner = start_runner or fit_swing_opensim

    seeds = generate_multistart_seeds(
        n_starts=multistart_options.n_starts,
        base_seed=multistart_options.base_seed,
    )
    successes: list[tuple[int, int, FitResult]] = []
    failures: list[StartFailure] = []

    # Serial execution is intentionally the first production slice. The public
    # options shape leaves room for process pools without exposing live OpenSim
    # model objects across process boundaries.
    for start_index, seed in enumerate(seeds):
        start_options = _options_for_start(
            fit_options,
            start_index=start_index,
            seed=seed,
            simulate_factory=simulate_factory,
        )
        try:
            result = runner(target, start_options)
            _validate_result(result, start_index=start_index, seed=seed)
        except Exception as exc:  # noqa: BLE001 - captured for diagnostics
            failure = _failure_dict(start_index=start_index, seed=seed, exc=exc)
            failures.append(failure)
            if multistart_options.fail_fast:
                raise
            continue
        if result.solver_status == "success":
            successes.append((start_index, seed, result))
        else:
            failures.append(
                {
                    "start_index": start_index,
                    "seed": seed,
                    "error_type": "SolverStatus",
                    "message": result.solver_status,
                }
            )

    if not successes:
        raise AllStartsFailedError(tuple(failures))

    best_index, best_seed, best_result = min(
        successes, key=lambda item: item[2].final_cost
    )
    return _with_multistart_meta(
        best_result,
        best_start_index=best_index,
        best_seed=best_seed,
        seeds=seeds,
        successes=successes,
        failures=failures,
        options=multistart_options,
    )


def _options_for_start(
    fit_options: FitOptions,
    *,
    start_index: int,
    seed: int,
    simulate_factory: SimulateFactory | None,
) -> FitOptions:
    simulate_fn = fit_options.simulate_fn
    if simulate_factory is not None:
        simulate_fn = cast(
            Callable[[NDArray[np.float64]], SimOutput],
            simulate_factory(start_index, seed),
        )
    return replace(fit_options, rng_seed=seed, theta0=None, simulate_fn=simulate_fn)


def _validate_result(result: FitResult, *, start_index: int, seed: int) -> None:
    if not isinstance(result, FitResult):
        raise TypeError(
            "OpenSim multistart runner must return FitResult, got "
            f"{type(result).__name__} for start {start_index} seed {seed}"
        )
    if not np.isfinite(result.final_cost):
        raise ValueError(
            f"OpenSim multistart start {start_index} seed {seed} returned "
            f"non-finite final_cost {result.final_cost!r}"
        )


def _failure_dict(*, start_index: int, seed: int, exc: Exception) -> StartFailure:
    return {
        "start_index": start_index,
        "seed": seed,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def _with_multistart_meta(
    result: FitResult,
    *,
    best_start_index: int,
    best_seed: int,
    seeds: tuple[int, ...],
    successes: list[tuple[int, int, FitResult]],
    failures: list[StartFailure],
    options: MultistartOptions,
) -> FitResult:
    starts = [
        {
            "start_index": start_index,
            "seed": seed,
            "final_cost": start_result.final_cost,
            "solver_status": start_result.solver_status,
        }
        for start_index, seed, start_result in successes
    ]
    multistart_meta = {
        "n_starts": options.n_starts,
        "base_seed": options.base_seed,
        "max_workers": options.max_workers,
        "fail_fast": options.fail_fast,
        "seeds": list(seeds),
        "best_start_index": best_start_index,
        "best_seed": best_seed,
        "successful_start_count": len(successes),
        "failed_start_count": len(failures),
        "starts": starts,
        "failures": failures,
    }
    return replace(
        result,
        meta={
            **result.meta,
            "multistart": multistart_meta,
        },
    )
