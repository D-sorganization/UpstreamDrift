"""Auditable primitives for pathway attribution and task-level robustness.

These functions are deliberately model-agnostic.  They do not turn a pathway
decomposition into a causal biological claim, and they define stability only
relative to declared perturbations and declared task outcomes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _finite_vector(name: str, value: npt.ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return array


@dataclass(frozen=True, slots=True)
class PathwayBudget:
    """Signed work ledger for one declared distal-energy balance."""

    work_j: dict[str, float]
    energy_change_j: float
    closure_residual_j: float
    cancellation_index: float


@dataclass(frozen=True, slots=True)
class PerturbationEnsemble:
    """Paired baseline/candidate outcomes under common perturbations."""

    perturbations: FloatArray
    baseline_outcomes: FloatArray
    candidate_outcomes: FloatArray
    outcome_names: tuple[str, ...]

    def __post_init__(self) -> None:
        perturbations = np.asarray(self.perturbations, dtype=np.float64)
        baseline = np.asarray(self.baseline_outcomes, dtype=np.float64)
        candidate = np.asarray(self.candidate_outcomes, dtype=np.float64)
        if perturbations.ndim != 2 or perturbations.shape[0] < 3:
            raise ValueError("perturbations must contain at least three rows")
        if baseline.shape != candidate.shape or baseline.ndim != 2:
            raise ValueError(
                "baseline and candidate outcomes must have equal 2-D shapes"
            )
        if baseline.shape[0] != perturbations.shape[0]:
            raise ValueError("perturbation and outcome sample counts must match")
        if baseline.shape[1] != len(self.outcome_names):
            raise ValueError("outcome_names must label every outcome column")
        if len(set(self.outcome_names)) != len(self.outcome_names):
            raise ValueError("outcome_names must be unique")
        if not all(
            np.all(np.isfinite(array)) for array in (perturbations, baseline, candidate)
        ):
            raise ValueError("ensemble arrays must contain only finite values")


@dataclass(frozen=True, slots=True)
class OutcomeLinearization:
    """Central finite-difference map from declared inputs to outcomes."""

    center: FloatArray
    steps: FloatArray
    jacobian: FloatArray
    input_names: tuple[str, ...]
    outcome_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskVariancePartition:
    """Local UCM-style variance split for a declared outcome Jacobian."""

    task_rank: int
    nullity: int
    null_variance: float
    task_relevant_variance: float
    synergy_index: float


def compute_pathway_budget(
    time_s: npt.ArrayLike,
    pathway_power_w: Mapping[str, npt.ArrayLike],
    *,
    energy_change_j: float,
    closure_tolerance_j: float = 1e-8,
) -> PathwayBudget:
    """Integrate signed pathways and fail closed when the ledger does not close."""

    time = _finite_vector("time_s", time_s)
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    if not pathway_power_w:
        raise ValueError("at least one pathway is required")
    if not np.isfinite(energy_change_j):
        raise ValueError("energy_change_j must be finite")
    if not np.isfinite(closure_tolerance_j) or closure_tolerance_j <= 0.0:
        raise ValueError("closure_tolerance_j must be finite and positive")
    work: dict[str, float] = {}
    for name, values in pathway_power_w.items():
        if not name.strip() or name in work:
            raise ValueError("pathway names must be nonempty and unique")
        power = np.asarray(values, dtype=np.float64)
        if power.shape != time.shape or not np.all(np.isfinite(power)):
            raise ValueError(f"pathway {name!r} must match time_s and be finite")
        work[name] = float(np.trapezoid(power, x=time))
    residual = float(sum(work.values()) - energy_change_j)
    if abs(residual) > closure_tolerance_j:
        raise ValueError(
            "pathway closure residual exceeds tolerance: "
            f"{residual:.3e} J > {closure_tolerance_j:.3e} J"
        )
    absolute_sum = sum(abs(value) for value in work.values())
    cancellation = (
        0.0 if absolute_sum == 0.0 else 1.0 - abs(sum(work.values())) / absolute_sum
    )
    return PathwayBudget(
        work_j=work,
        energy_change_j=float(energy_change_j),
        closure_residual_j=residual,
        cancellation_index=float(np.clip(cancellation, 0.0, 1.0)),
    )


def perturbation_summary(ensemble: PerturbationEnsemble) -> dict[str, dict[str, float]]:
    """Summarize paired performance without hiding lower-tail behavior."""

    baseline = np.asarray(ensemble.baseline_outcomes, dtype=np.float64)
    candidate = np.asarray(ensemble.candidate_outcomes, dtype=np.float64)
    perturbations = np.asarray(ensemble.perturbations, dtype=np.float64)
    centered = perturbations - np.mean(perturbations, axis=0)
    scale = np.linalg.norm(centered)
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(ensemble.outcome_names):
        base = baseline[:, index]
        cand = candidate[:, index]
        base_centered = base - np.mean(base)
        cand_centered = cand - np.mean(cand)
        result[name] = {
            "baseline_mean": float(np.mean(base)),
            "candidate_mean": float(np.mean(cand)),
            "baseline_q10": float(np.quantile(base, 0.10)),
            "candidate_q10": float(np.quantile(cand, 0.10)),
            "baseline_std": float(np.std(base, ddof=1)),
            "candidate_std": float(np.std(cand, ddof=1)),
            "paired_mean_delta": float(np.mean(cand - base)),
            "baseline_amplification": (
                0.0 if scale == 0.0 else float(np.linalg.norm(base_centered) / scale)
            ),
            "candidate_amplification": (
                0.0 if scale == 0.0 else float(np.linalg.norm(cand_centered) / scale)
            ),
        }
    return result


def finite_difference_outcome_jacobian(
    evaluator: Callable[[FloatArray], npt.ArrayLike],
    *,
    center: npt.ArrayLike,
    steps: npt.ArrayLike,
    input_names: tuple[str, ...],
    outcome_names: tuple[str, ...],
) -> OutcomeLinearization:
    """Evaluate a scale-explicit central local input--outcome map."""

    point = _finite_vector("center", center)
    delta = np.asarray(steps, dtype=np.float64)
    if (
        delta.shape != point.shape
        or np.any(delta <= 0.0)
        or not np.all(np.isfinite(delta))
    ):
        raise ValueError("steps must match center and be finite and positive")
    if len(input_names) != point.size or len(set(input_names)) != len(input_names):
        raise ValueError("input_names must uniquely label center")
    reference = np.asarray(evaluator(point.copy()), dtype=np.float64)
    if reference.ndim != 1 or reference.size != len(outcome_names):
        raise ValueError("evaluator output must match outcome_names")
    jacobian = np.empty((reference.size, point.size), dtype=np.float64)
    for index, step in enumerate(delta):
        offset = np.zeros_like(point)
        offset[index] = step
        upper = np.asarray(evaluator(point + offset), dtype=np.float64)
        lower = np.asarray(evaluator(point - offset), dtype=np.float64)
        if upper.shape != reference.shape or lower.shape != reference.shape:
            raise ValueError("evaluator output shape changed during linearization")
        jacobian[:, index] = (upper - lower) / (2.0 * step)
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("outcome Jacobian contains nonfinite values")
    return OutcomeLinearization(
        center=point,
        steps=delta.copy(),
        jacobian=jacobian,
        input_names=input_names,
        outcome_names=outcome_names,
    )


def task_variance_partition(
    outcome_jacobian: npt.ArrayLike,
    elemental_samples: npt.ArrayLike,
    *,
    rank_tolerance: float | None = None,
) -> TaskVariancePartition:
    """Partition elemental variance into local null and task-relevant subspaces."""

    jacobian = np.asarray(outcome_jacobian, dtype=np.float64)
    samples = np.asarray(elemental_samples, dtype=np.float64)
    if jacobian.ndim != 2 or samples.ndim != 2 or samples.shape[1] != jacobian.shape[1]:
        raise ValueError("Jacobian and elemental sample dimensions are incompatible")
    if (
        samples.shape[0] < 3
        or not np.all(np.isfinite(jacobian))
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError(
            "finite Jacobian and at least three finite samples are required"
        )
    _, singular, right = np.linalg.svd(jacobian, full_matrices=True)
    tolerance = (
        max(jacobian.shape)
        * np.finfo(float).eps
        * (singular[0] if singular.size else 0.0)
        if rank_tolerance is None
        else rank_tolerance
    )
    rank = int(np.sum(singular > tolerance))
    centered = samples - np.mean(samples, axis=0)
    task_coordinates = centered @ right[:rank].T
    null_coordinates = centered @ right[rank:].T
    task_variance = float(np.sum(np.var(task_coordinates, axis=0, ddof=1)))
    null_variance = float(np.sum(np.var(null_coordinates, axis=0, ddof=1)))
    total = task_variance + null_variance
    synergy = 0.0 if total == 0.0 else (null_variance - task_variance) / total
    return TaskVariancePartition(
        task_rank=rank,
        nullity=int(jacobian.shape[1] - rank),
        null_variance=null_variance,
        task_relevant_variance=task_variance,
        synergy_index=float(synergy),
    )


def nondominated_indices(objectives: npt.ArrayLike) -> tuple[int, ...]:
    """Return rows not Pareto-dominated when all columns are minimized."""

    values = np.asarray(objectives, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or not np.all(np.isfinite(values)):
        raise ValueError("objectives must be a nonempty finite matrix")
    keep: list[int] = []
    for index, row in enumerate(values):
        dominated = any(
            np.all(other <= row) and np.any(other < row)
            for other_index, other in enumerate(values)
            if other_index != index
        )
        if not dominated:
            keep.append(index)
    return tuple(keep)


__all__ = [
    "OutcomeLinearization",
    "PathwayBudget",
    "PerturbationEnsemble",
    "TaskVariancePartition",
    "compute_pathway_budget",
    "finite_difference_outcome_jacobian",
    "nondominated_indices",
    "perturbation_summary",
    "task_variance_partition",
]
