"""Bounded F02 loop tuning with frozen-controller coupling diagnostics (F04).

The evaluator owns the plant and calls the F02 controller at actual integration
boundaries. This module owns only bounded outer tuning and descriptive evidence.
It does not infer causal neural control from covariance or synthetic tracking.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

Array: TypeAlias = NDArray[np.float64]
ParameterKind = Literal["gain", "task_weight"]
TrialSplit = Literal["train", "holdout"]


def _readonly(values: Array) -> Array:
    """Detach result arrays so later caller edits cannot rewrite evidence."""
    frozen = np.array(values, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen


@dataclass(frozen=True)
class LoopParameter:
    """One bounded, scaled control parameter; nominal feedforward is frozen."""

    name: str
    group: str
    kind: ParameterKind
    initial: float
    lower: float
    upper: float
    scale: float

    def __post_init__(self) -> None:
        values = np.array([self.initial, self.lower, self.upper, self.scale])
        if not self.name or not self.group or self.kind not in ("gain", "task_weight"):
            raise ValueError("parameter name, group and kind are required")
        if (
            not np.isfinite(values).all()
            or self.lower >= self.upper
            or not self.lower <= self.initial <= self.upper
            or self.scale <= 0
        ):
            raise ValueError("parameter bounds, initial and scale must be finite")


@dataclass(frozen=True)
class TuningTrial:
    trial_id: str
    split: TrialSplit

    def __post_init__(self) -> None:
        if not self.trial_id or self.split not in ("train", "holdout"):
            raise ValueError("named train/holdout trial is required")


@dataclass(frozen=True)
class TuningEvaluation:
    """One actual controller/plant rollout with phase/group squared losses."""

    phase_losses: Array
    effort: float
    robustness: float
    constraint_violation: float
    saturation_fraction: float

    def __post_init__(self) -> None:
        losses = np.asarray(self.phase_losses, dtype=float)
        values = np.array(
            [
                self.effort,
                self.robustness,
                self.constraint_violation,
                self.saturation_fraction,
            ]
        )
        if (
            losses.ndim != 2
            or not np.isfinite(losses).all()
            or np.any(losses < 0)
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or self.saturation_fraction > 1
        ):
            raise ValueError("tuning evaluation must be finite nonnegative evidence")
        object.__setattr__(self, "phase_losses", _readonly(losses))


Evaluator = Callable[[Array, TuningTrial, tuple[str, ...]], TuningEvaluation]


@dataclass(frozen=True)
class LoopTuningProblem:
    parameters: tuple[LoopParameter, ...]
    groups: tuple[str, ...]
    phases: tuple[str, ...]
    trials: tuple[TuningTrial, ...]
    evaluate: Evaluator
    effort_weight: float
    robustness_weight: float
    regularization_weight: float

    def __post_init__(self) -> None:
        if (
            not self.parameters
            or not self.groups
            or not self.phases
            or len(set(self.groups)) != len(self.groups)
            or len(set(self.phases)) != len(self.phases)
            or any(not label for label in (*self.groups, *self.phases))
            or len({item.name for item in self.parameters}) != len(self.parameters)
            or any(item.group not in self.groups for item in self.parameters)
            or len({trial.trial_id for trial in self.trials}) != len(self.trials)
            or not any(trial.split == "train" for trial in self.trials)
            or not any(trial.split == "holdout" for trial in self.trials)
            or not callable(self.evaluate)
        ):
            raise ValueError(
                "unique groups/parameters and train/holdout trials required"
            )
        weights = np.array(
            [self.effort_weight, self.robustness_weight, self.regularization_weight]
        )
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("tuning objective weights must be finite nonnegative")

    @property
    def initial(self) -> Array:
        return np.array([item.initial for item in self.parameters], dtype=float)

    @property
    def train_trials(self) -> tuple[TuningTrial, ...]:
        return tuple(trial for trial in self.trials if trial.split == "train")

    @property
    def holdout_trials(self) -> tuple[TuningTrial, ...]:
        return tuple(trial for trial in self.trials if trial.split == "holdout")


@dataclass(frozen=True)
class LoopTuningConfig:
    max_passes: int = 2
    block_max_evaluations: int = 30
    joint_max_evaluations: int = 60
    trust_radius_scaled: float = 1.0
    max_cross_group_regression: float = 0.1
    max_phase_group_regression: float = 1.0
    max_constraint_violation: float = 0.0
    max_diagnostic_evaluations: int = 256
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            self.max_passes <= 0
            or self.block_max_evaluations <= 0
            or self.joint_max_evaluations <= 0
            or self.max_diagnostic_evaluations <= 0
            or self.seed < 0
            or not np.isfinite(
                [
                    self.trust_radius_scaled,
                    self.max_cross_group_regression,
                    self.max_phase_group_regression,
                    self.max_constraint_violation,
                ]
            ).all()
            or self.trust_radius_scaled <= 0
            or self.max_cross_group_regression < 0
            or self.max_phase_group_regression < 0
            or self.max_constraint_violation < 0
        ):
            raise ValueError("loop tuning budgets and gates must be valid")


@dataclass(frozen=True)
class TuningScore:
    objective: float
    group_losses: Array
    phase_group_losses: Array
    tracking_rmse: float
    effort: float
    robustness: float
    constraint_violation: float
    saturation_fraction: float
    wall_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_losses", _readonly(self.group_losses))
        object.__setattr__(
            self, "phase_group_losses", _readonly(self.phase_group_losses)
        )


@dataclass(frozen=True)
class TuningCheckpoint:
    pass_index: int
    stage: str
    accepted: bool
    reason: str
    parameters: tuple[float, ...]
    objective: float
    group_losses: tuple[float, ...]
    constraint_violation: float
    saturation_fraction: float
    solver_success: bool
    evaluations: int


@dataclass(frozen=True)
class FrozenCouplingDiagnostics:
    jacobian: Array
    cross_jacobian_norms: Array
    cross_hessian: Array
    rank_deficient: bool
    singular_values: Array
    interpretation: str = "frozen_controller_association_not_causation"

    def __post_init__(self) -> None:
        for name in (
            "jacobian",
            "cross_jacobian_norms",
            "cross_hessian",
            "singular_values",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True)
class PhaseCovariance:
    pooled_covariance: Array
    within_phase_covariance: Array
    phases: tuple[str, ...]
    groups: tuple[str, ...]
    causal_claim: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "pooled_covariance", _readonly(self.pooled_covariance))
        object.__setattr__(
            self, "within_phase_covariance", _readonly(self.within_phase_covariance)
        )


@dataclass(frozen=True)
class LoopTuningResult:
    parameters: Array
    parameter_identity: tuple[tuple[str, str, str, float], ...]
    initial_objective: float
    objective: float
    checkpoints: tuple[TuningCheckpoint, ...]
    checkpoint_digest: str
    coupling: FrozenCouplingDiagnostics
    holdout_full: TuningScore
    holdout_reduced: tuple[TuningScore, ...]
    reduced_groupings: tuple[tuple[str, ...], ...]
    phase_covariance: PhaseCovariance | None
    generalization_status: str
    feedforward_policy: str = "frozen_not_jointly_identified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _readonly(self.parameters))


@dataclass(frozen=True)
class PerturbationComparison:
    """Frozen-controller response versus a separately labeled refit."""

    frozen_holdout: TuningScore
    reoptimized_holdout: TuningScore
    reoptimized_parameters: Array
    scaled_parameter_shift: float
    interpretation: str = "refit_compensation_not_frozen_controller_sensitivity"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reoptimized_parameters", _readonly(self.reoptimized_parameters)
        )


def _score(
    problem: LoopTuningProblem,
    values: Array,
    trials: tuple[TuningTrial, ...],
    active: tuple[str, ...],
) -> TuningScore:
    started = time.perf_counter()
    rows = []
    for trial in trials:
        item = problem.evaluate(values.copy(), trial, active)
        if not isinstance(item, TuningEvaluation) or item.phase_losses.shape != (
            len(problem.phases),
            len(problem.groups),
        ):
            raise ValueError("evaluator returned incompatible phase/group evidence")
        rows.append(item)
    phase_losses = np.stack([row.phase_losses for row in rows])
    group_losses = np.mean(phase_losses, axis=(0, 1))
    phase_group_losses = np.mean(phase_losses, axis=0)
    effort = float(np.mean([row.effort for row in rows]))
    robustness = float(np.mean([row.robustness for row in rows]))
    constraint = max(row.constraint_violation for row in rows)
    saturation = float(np.mean([row.saturation_fraction for row in rows]))
    prior = (values - problem.initial) / np.array(
        [item.scale for item in problem.parameters]
    )
    objective = (
        float(np.sum(group_losses))
        + problem.effort_weight * effort
        + problem.robustness_weight * robustness
        + problem.regularization_weight * float(prior @ prior)
    )
    return TuningScore(
        objective,
        group_losses,
        phase_group_losses,
        float(np.sqrt(np.mean(phase_losses))),
        effort,
        robustness,
        constraint,
        saturation,
        time.perf_counter() - started,
    )


def _trial_phase_mean(problem: LoopTuningProblem, values: Array) -> Array:
    rows = [
        problem.evaluate(values.copy(), trial, problem.groups).phase_losses
        for trial in problem.train_trials
    ]
    if any(row.shape != (len(problem.phases), len(problem.groups)) for row in rows):
        raise ValueError("phase/group response shape changed under perturbation")
    return np.mean(np.stack(rows), axis=0)


def _objective_hessian(problem: LoopTuningProblem, point: Array) -> Array:
    """Bound-aware finite differences of the frozen training objective.

    Each coordinate uses a central stencil when possible and a one-sided
    stencil at a bound. The result is scaled by declared parameter scales.
    """
    dimensions = len(point)
    hessian = np.empty((dimensions, dimensions))
    steps = np.array(
        [
            min(1e-3 * item.scale, (item.upper - item.lower) / 4)
            for item in problem.parameters
        ]
    )

    def objective(offsets: tuple[tuple[int, float], ...]) -> float:
        values = point.copy()
        for index, offset in offsets:
            values[index] += offset
        return _score(problem, values, problem.train_trials, problem.groups).objective

    for i, item in enumerate(problem.parameters):
        step = steps[i]
        if point[i] - step >= item.lower and point[i] + step <= item.upper:
            offsets = (-step, 0.0, step)
        elif point[i] + 2 * step <= item.upper:
            offsets = (0.0, step, 2 * step)
        else:
            offsets = (-2 * step, -step, 0.0)
        hessian[i, i] = (
            (
                objective(((i, offsets[0]),))
                - 2 * objective(((i, offsets[1]),))
                + objective(((i, offsets[2]),))
            )
            / (step * step)
            * item.scale
            * item.scale
        )
        for j in range(i + 1, dimensions):
            other = problem.parameters[j]
            span_i = (step, -step) if point[i] - step >= item.lower else (step, 0.0)
            span_j = (
                (steps[j], -steps[j])
                if point[j] - steps[j] >= other.lower
                else (steps[j], 0.0)
            )
            if point[i] + span_i[0] > item.upper:
                span_i = (0.0, -step)
            if point[j] + span_j[0] > other.upper:
                span_j = (0.0, -steps[j])
            high_i, low_i = span_i
            high_j, low_j = span_j
            numerator = (
                objective(((i, high_i), (j, high_j)))
                - objective(((i, high_i), (j, low_j)))
                - objective(((i, low_i), (j, high_j)))
                + objective(((i, low_i), (j, low_j)))
            )
            hessian[i, j] = (
                numerator
                / ((high_i - low_i) * (high_j - low_j))
                * item.scale
                * other.scale
            )
            hessian[j, i] = hessian[i, j]
    return hessian


def frozen_coupling_diagnostics(
    problem: LoopTuningProblem, values: Array, *, max_evaluations: int = 256
) -> FrozenCouplingDiagnostics:
    """Scaled finite-difference response; no reoptimization or causal claim."""
    point = np.asarray(values, dtype=float)
    if point.shape != (len(problem.parameters),) or not np.isfinite(point).all():
        raise ValueError("frozen coupling point must match finite parameter vector")
    rollout_cost = (2 * len(point) ** 2 + 3 * len(point)) * len(problem.train_trials)
    if max_evaluations <= 0 or rollout_cost > max_evaluations:
        raise ValueError("frozen coupling diagnostic budget is insufficient")
    if any(
        point[i] < item.lower or point[i] > item.upper
        for i, item in enumerate(problem.parameters)
    ):
        raise ValueError("frozen coupling point lies outside bounds")
    jacobian = np.empty((len(problem.phases), len(problem.groups), len(point)))
    for i, item in enumerate(problem.parameters):
        plus = point.copy()
        minus = point.copy()
        plus[i] = min(item.upper, point[i] + 1e-4 * item.scale)
        minus[i] = max(item.lower, point[i] - 1e-4 * item.scale)
        if plus[i] == minus[i]:
            raise ValueError("parameter has no finite-difference span")
        derivative = (
            _trial_phase_mean(problem, plus) - _trial_phase_mean(problem, minus)
        ) / (plus[i] - minus[i])
        jacobian[:, :, i] = derivative * item.scale
    flat = jacobian.reshape(-1, len(point))
    singular = np.asarray(np.linalg.svd(flat, compute_uv=False), dtype=np.float64)
    rank = np.linalg.matrix_rank(flat)
    objective_hessian = _objective_hessian(problem, point)
    cross_norms = np.zeros((len(problem.groups), len(problem.groups)))
    cross_hessian = np.zeros_like(cross_norms)
    for source, group in enumerate(problem.groups):
        source_ids = [
            i for i, item in enumerate(problem.parameters) if item.group == group
        ]
        for target, other in enumerate(problem.groups):
            target_ids = [
                i for i, item in enumerate(problem.parameters) if item.group == other
            ]
            cross_norms[source, target] = float(
                np.linalg.norm(jacobian[:, source, :][:, target_ids])
            )
            cross_hessian[source, target] = float(
                np.linalg.norm(objective_hessian[np.ix_(source_ids, target_ids)])
            )
    return FrozenCouplingDiagnostics(
        jacobian, cross_norms, cross_hessian, rank < len(point), singular
    )


def descriptive_phase_covariance(
    observations: Array, phases: tuple[str, ...], groups: tuple[str, ...]
) -> PhaseCovariance:
    """Separate pooled phase shifts from within-phase covariance, descriptively."""
    values = np.asarray(observations, dtype=float)
    if (
        values.ndim != 3
        or values.shape[0] < 2
        or values.shape[1:] != (len(phases), len(groups))
        or not np.isfinite(values).all()
        or not phases
        or not groups
    ):
        raise ValueError("covariance needs finite repeated trials and named axes")
    flattened = values.reshape(-1, len(groups))
    centred = values - values.mean(axis=0, keepdims=True)
    within = centred.reshape(-1, len(groups))
    return PhaseCovariance(
        np.cov(flattened, rowvar=False),
        np.cov(within, rowvar=False),
        phases,
        groups,
    )


def _candidate_reason(
    prior: TuningScore, candidate: TuningScore, config: LoopTuningConfig, success: bool
) -> str:
    if not success:
        return "nonconverged"
    if candidate.constraint_violation > config.max_constraint_violation:
        return "constraint_regression"
    if np.any(
        candidate.phase_group_losses - prior.phase_group_losses
        > config.max_phase_group_regression
    ):
        return "phase_group_regression"
    if np.any(
        candidate.group_losses - prior.group_losses > config.max_cross_group_regression
    ):
        return "cross_group_regression"
    if candidate.objective >= prior.objective - 1e-10:
        return "no_improvement"
    return "accepted"


class _BudgetExhausted(Exception):
    """Private sentinel that stops SciPy before another plant rollout."""


def _stage(
    problem: LoopTuningProblem,
    config: LoopTuningConfig,
    current: Array,
    prior: TuningScore,
    indices: tuple[int, ...],
    stage: str,
    pass_index: int,
    budget: int,
) -> tuple[Array, TuningScore, TuningCheckpoint]:
    bounds = tuple(
        (
            max(
                problem.parameters[i].lower,
                current[i] - config.trust_radius_scaled * problem.parameters[i].scale,
            ),
            min(
                problem.parameters[i].upper,
                current[i] + config.trust_radius_scaled * problem.parameters[i].scale,
            ),
        )
        for i in indices
    )

    def proposed(subvector: Array) -> Array:
        values = current.copy()
        values[list(indices)] = subvector
        return values

    evaluations = 0
    last_values = current
    last_score = prior

    def objective(subvector: Array) -> float:
        nonlocal evaluations, last_values, last_score
        if evaluations >= budget:
            raise _BudgetExhausted
        last_values = proposed(subvector)
        last_score = _score(problem, last_values, problem.train_trials, problem.groups)
        evaluations += 1
        return last_score.objective

    try:
        solved = minimize(
            objective,
            current[list(indices)],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxfun": budget, "maxiter": budget, "ftol": 1e-10},
        )
        candidate_values = proposed(np.asarray(solved.x, dtype=float))
        if not np.array_equal(candidate_values, last_values):
            objective(np.asarray(solved.x, dtype=float))
        candidate_score = last_score
    except _BudgetExhausted:
        checkpoint = TuningCheckpoint(
            pass_index,
            stage,
            False,
            "budget_exhausted",
            tuple(float(value) for value in last_values),
            last_score.objective,
            tuple(float(value) for value in last_score.group_losses),
            last_score.constraint_violation,
            last_score.saturation_fraction,
            False,
            evaluations,
        )
        return current, prior, checkpoint
    reason = _candidate_reason(prior, candidate_score, config, bool(solved.success))
    checkpoint = TuningCheckpoint(
        pass_index,
        stage,
        reason == "accepted",
        reason,
        tuple(float(value) for value in candidate_values),
        candidate_score.objective,
        tuple(float(value) for value in candidate_score.group_losses),
        candidate_score.constraint_violation,
        candidate_score.saturation_fraction,
        bool(solved.success),
        evaluations,
    )
    if reason == "accepted":
        return candidate_values, candidate_score, checkpoint
    return current, prior, checkpoint


def tune_control_loops(
    problem: LoopTuningProblem,
    config: LoopTuningConfig,
    *,
    reduced_groupings: tuple[tuple[str, ...], ...] | None = None,
) -> LoopTuningResult:
    """Block-coordinate initialize, jointly refine and retain every decision."""
    groupings = reduced_groupings or tuple((group,) for group in problem.groups)
    if any(not group or not set(group) <= set(problem.groups) for group in groupings):
        raise ValueError("reduced task grouping must use declared nonempty groups")
    current = problem.initial
    first = _score(problem, current, problem.train_trials, problem.groups)
    if first.constraint_violation > config.max_constraint_violation:
        raise ValueError("initial controller violates frozen constraint gate")
    prior = first
    checkpoints: list[TuningCheckpoint] = []
    for pass_index in range(config.max_passes):
        accepted_in_pass = False
        for group in problem.groups:
            indices = tuple(
                i for i, item in enumerate(problem.parameters) if item.group == group
            )
            if not indices:
                continue
            current, prior, checkpoint = _stage(
                problem,
                config,
                current,
                prior,
                indices,
                group,
                pass_index,
                config.block_max_evaluations,
            )
            checkpoints.append(checkpoint)
            accepted_in_pass |= checkpoint.accepted
        if not accepted_in_pass:
            break
    current, prior, checkpoint = _stage(
        problem,
        config,
        current,
        prior,
        tuple(range(len(current))),
        "joint",
        len(checkpoints),
        config.joint_max_evaluations,
    )
    checkpoints.append(checkpoint)
    coupling = frozen_coupling_diagnostics(
        problem, current, max_evaluations=config.max_diagnostic_evaluations
    )
    holdout_full = _score(problem, current, problem.holdout_trials, problem.groups)
    reduced = tuple(
        _score(problem, current, problem.holdout_trials, groups) for groups in groupings
    )
    covariance = None
    if len(problem.train_trials) > 1:
        observations = np.stack(
            [
                problem.evaluate(current.copy(), trial, problem.groups).phase_losses
                for trial in problem.train_trials
            ]
        )
        covariance = descriptive_phase_covariance(
            observations, problem.phases, problem.groups
        )
    encoded = json.dumps(
        {
            "config": asdict(config),
            "checkpoints": [asdict(checkpoint) for checkpoint in checkpoints],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return LoopTuningResult(
        current.copy(),
        tuple(
            (item.name, item.group, item.kind, item.scale)
            for item in problem.parameters
        ),
        first.objective,
        prior.objective,
        tuple(checkpoints),
        hashlib.sha256(encoded).hexdigest(),
        coupling,
        holdout_full,
        reduced,
        groupings,
        covariance,
        "multi_trial_descriptive"
        if len(problem.train_trials) > 1
        else "single_training_trial_insufficient",
    )


def compare_perturbation_compensation(
    baseline: LoopTuningResult,
    perturbed: LoopTuningProblem,
    config: LoopTuningConfig,
) -> PerturbationComparison:
    """Score a frozen policy before fitting again on separate perturbation trials."""
    values = np.asarray(baseline.parameters, dtype=float)
    identity = tuple(
        (item.name, item.group, item.kind, item.scale) for item in perturbed.parameters
    )
    if identity != baseline.parameter_identity:
        raise ValueError("perturbation comparison requires exact parameter identity")
    if values.shape != (len(perturbed.parameters),) or not np.isfinite(values).all():
        raise ValueError("perturbation comparison requires matching finite parameters")
    if any(
        values[i] < item.lower or values[i] > item.upper
        for i, item in enumerate(perturbed.parameters)
    ):
        raise ValueError("baseline controller lies outside perturbation bounds")
    frozen = _score(perturbed, values, perturbed.holdout_trials, perturbed.groups)
    warm = replace(
        perturbed,
        parameters=tuple(
            replace(item, initial=float(values[i]))
            for i, item in enumerate(perturbed.parameters)
        ),
    )
    refit = tune_control_loops(warm, config)
    scales = np.array([item.scale for item in perturbed.parameters])
    return PerturbationComparison(
        frozen,
        refit.holdout_full,
        refit.parameters,
        float(np.linalg.norm((refit.parameters - values) / scales)),
    )
