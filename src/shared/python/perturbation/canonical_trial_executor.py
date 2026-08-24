"""Deterministic serial execution over canonical Tools variation samples."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

import numpy as np

from .trial_evidence import CanonicalTrialEvidence

RECOVERABLE_TRIAL_ERRORS = (ValueError, RuntimeError, FloatingPointError)
TrialRunner = Callable[[np.ndarray], object]
BatchTrialRunner = Callable[[np.ndarray], object]


class VariationSampler(Protocol):
    """Narrow Tools-gateway operation required by the executor."""

    def sample_inputs(self, plan: object) -> object:
        """Return the deterministic canonical sample matrix."""
        ...


class TrialEvidenceCollector(Protocol):
    """Engine-owned mapping from raw execution to canonical trial evidence."""

    def collect_success(
        self,
        trial_index: int,
        plan_seed: int,
        sampled_row: np.ndarray,
        result: object,
    ) -> CanonicalTrialEvidence:
        """Map one successful engine return, including a legitimate miss."""
        ...

    def collect_failure(
        self,
        trial_index: int,
        plan_seed: int,
        sampled_row: np.ndarray,
        error: Exception,
    ) -> CanonicalTrialEvidence:
        """Map one declared numerical failure without fabricating outputs."""
        ...


def _plan_execution_identity(plan: object) -> tuple[int, int]:
    n_runs = getattr(plan, "n_runs", None)
    seed = getattr(plan, "seed", None)
    if type(n_runs) is not int or n_runs <= 0:
        raise ValueError("plan n_runs must be a positive integer")
    if type(seed) is not int:
        raise ValueError("plan seed must be an integer")
    return n_runs, seed


def _sample_matrix(value: object, n_runs: int) -> np.ndarray:
    samples = np.array(value, dtype=float, copy=True)
    if (
        samples.ndim != 2
        or samples.shape[0] != n_runs
        or samples.shape[1] == 0
        or not np.isfinite(samples).all()
    ):
        raise ValueError(
            f"Tools sample matrix must be finite with shape ({n_runs}, n_inputs>0)"
        )
    samples.flags.writeable = False
    return samples


def _validate_dependencies(
    gateway: VariationSampler,
    runner: TrialRunner,
    collector: TrialEvidenceCollector,
) -> None:
    if not callable(getattr(gateway, "sample_inputs", None)):
        raise TypeError("gateway must expose callable sample_inputs")
    if not callable(runner):
        raise TypeError("runner must be callable")
    for name in ("collect_success", "collect_failure"):
        if not callable(getattr(collector, name, None)):
            raise TypeError(f"collector must expose callable {name}")


def _validate_collected_record(
    record: object, trial_index: int, plan_seed: int
) -> CanonicalTrialEvidence:
    if not isinstance(record, CanonicalTrialEvidence):
        raise TypeError("collector must return CanonicalTrialEvidence")
    if record.trial_index != trial_index or record.seed != plan_seed:
        raise ValueError("collector returned trial identity drift")
    return record


def _readonly_row(samples: np.ndarray, trial_index: int) -> np.ndarray:
    row = np.array(samples[trial_index], copy=True)
    row.flags.writeable = False
    return row


def _collect_result(
    collector: TrialEvidenceCollector,
    trial_index: int,
    plan_seed: int,
    row: np.ndarray,
    result: object,
) -> CanonicalTrialEvidence:
    if isinstance(result, RECOVERABLE_TRIAL_ERRORS):
        record = collector.collect_failure(trial_index, plan_seed, row, result)
    elif isinstance(result, Exception):
        raise result
    else:
        record = collector.collect_success(trial_index, plan_seed, row, result)
    return _validate_collected_record(record, trial_index, plan_seed)


def execute_serial_variation(
    plan: object,
    gateway: VariationSampler,
    runner: TrialRunner,
    collector: TrialEvidenceCollector,
) -> tuple[CanonicalTrialEvidence, ...]:
    """Execute all canonical rows once and retain one typed record per row.

    Only declared numerical/runtime failures become trial data. Type errors,
    assertions, interrupts, and other programming/control failures propagate.
    This function owns no sampling, engine physics, or outcome interpretation.
    """

    _validate_dependencies(gateway, runner, collector)
    n_runs, plan_seed = _plan_execution_identity(plan)
    samples = _sample_matrix(gateway.sample_inputs(plan), n_runs)
    records: list[CanonicalTrialEvidence] = []
    for trial_index in range(n_runs):
        row = _readonly_row(samples, trial_index)
        try:
            result = runner(row)
        except RECOVERABLE_TRIAL_ERRORS as exc:
            result = exc
        records.append(_collect_result(collector, trial_index, plan_seed, row, result))
    return tuple(records)


def execute_batched_variation(
    plan: object,
    gateway: VariationSampler,
    batch_runner: BatchTrialRunner,
    collector: TrialEvidenceCollector,
) -> tuple[CanonicalTrialEvidence, ...]:
    """Execute one injected batch and retain one ordered record per sample.

    The batch adapter returns one raw result or declared exception per input
    row. Whole-batch exceptions propagate because truthful per-row attribution
    is then unavailable.
    """

    _validate_dependencies(gateway, batch_runner, collector)
    n_runs, plan_seed = _plan_execution_identity(plan)
    samples = _sample_matrix(gateway.sample_inputs(plan), n_runs)
    raw_results = batch_runner(samples)
    if isinstance(raw_results, (str, bytes)) or not isinstance(raw_results, Iterable):
        raise ValueError("batch runner must return one result per sample")
    results: tuple[object, ...] = tuple(raw_results)
    if len(results) != n_runs:
        raise ValueError("batch runner must return one result per sample")
    return tuple(
        _collect_result(
            collector,
            trial_index,
            plan_seed,
            _readonly_row(samples, trial_index),
            result,
        )
        for trial_index, result in enumerate(results)
    )


__all__ = [
    "RECOVERABLE_TRIAL_ERRORS",
    "BatchTrialRunner",
    "TrialEvidenceCollector",
    "TrialRunner",
    "VariationSampler",
    "execute_batched_variation",
    "execute_serial_variation",
]
