"""Bounded one-step muscle recruitment for quasistatic tendon equilibrium.

Coordinate efforts are N for translations and N*m for rotations. Moment arms
are -d(length)/d(coordinate), in m/m or m/rad respectively. This is not an
OpenSim-parity or tendon-internal-dynamics implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Literal, Sequence, cast

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics
from src.shared.python.biomechanics.hill_muscle import HillMuscleModel
from src.shared.python.biomechanics.muscle_equilibrium import EquilibriumSolver

from .muscle_force_evaluation import evaluate_quasistatic_muscle_force

EffortUnit = Literal["N", "N*m"]


@dataclass(frozen=True)
class MuscleRecruitmentPath:
    """One muscle path and immutable prior state for a single control step."""

    name: str
    muscle: HillMuscleModel
    activation_dynamics: ActivationDynamics
    muscle_tendon_length_m: float
    moment_arms_m_or_rad: dict[str, float]
    activation: float
    previous_excitation: float
    excitation_bounds: tuple[float, float]
    max_excitation_rate_per_s: float


@dataclass(frozen=True)
class MuscleAllocationResult:
    """Best bounded candidate, complete coordinate residual, and status."""

    status: Literal[
        "feasible",
        "infeasible",
        "budget_exhausted",
        "solver_failed",
        "evaluation_failed",
    ]
    feasible: bool
    # Raw commanded values obey the declared box and slew limit.
    excitations: dict[str, float]
    # ActivationDynamics internally floors its input; report that separately.
    effective_excitations: dict[str, float]
    activations: dict[str, float]
    effective_bounds: dict[str, tuple[float, float]]
    muscle_forces_n: dict[str, dict[str, float]]
    generalized_effort: dict[str, float]
    residual_by_coordinate: dict[str, float]
    nfev: int
    message: str


@dataclass(frozen=True)
class _Evaluation:
    activation: float
    effective_excitation: float
    components: dict[str, float]
    tendon_force: float
    equilibrium_residual_n: float


class _EvaluationBudgetExhausted(RuntimeError):
    """Raised before an objective call would exceed the explicit budget."""


def _number(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _bounds(path: MuscleRecruitmentPath, dt_s: float) -> tuple[float, float]:
    low, high = path.excitation_bounds
    return (
        max(low, path.previous_excitation - path.max_excitation_rate_per_s * dt_s),
        min(high, path.previous_excitation + path.max_excitation_rate_per_s * dt_s),
    )


def _evaluate(
    path: MuscleRecruitmentPath, excitation: float, dt_s: float
) -> _Evaluation:
    effective_excitation = max(
        excitation, float(path.activation_dynamics.min_activation)
    )
    activation = float(
        path.activation_dynamics.update(excitation, path.activation, dt_s)
    )
    force = evaluate_quasistatic_muscle_force(
        path.muscle, path.muscle_tendon_length_m, activation
    )
    return _Evaluation(
        activation=activation,
        effective_excitation=effective_excitation,
        components={
            "active_fiber_along_fiber_n": force.active_fiber_along_fiber_n,
            "passive_fiber_along_fiber_n": force.passive_fiber_along_fiber_n,
            "damping_fiber_along_fiber_n": force.damping_fiber_along_fiber_n,
            "active_tendon_axis_n": force.active_tendon_axis_n,
            "passive_tendon_axis_n": force.passive_tendon_axis_n,
            "damping_tendon_axis_n": force.damping_tendon_axis_n,
            "fiber_tendon_axis_n": force.fiber_tendon_axis_n,
            "tendon_n": force.tendon_force_n,
            "fiber_length_m": force.fiber_length_m,
            "force_equilibrium_residual_n": force.force_equilibrium_residual_n,
            "activation_input_excitation": effective_excitation,
        },
        tendon_force=force.tendon_force_n,
        equilibrium_residual_n=force.force_equilibrium_residual_n,
    )


@dataclass(frozen=True)
class _ParsedAllocationRequest:
    """Validated numeric inputs for muscle excitation allocation."""

    target: np.ndarray
    tolerances: np.ndarray
    dt: float
    force_tolerance: float
    max_nfev: int
    size: int


@dataclass
class _SearchOutcome:
    best_candidate: np.ndarray | None
    best_observation: tuple[np.ndarray, list[_Evaluation]] | None
    evaluation_count: int
    status_message: str
    budget_exhausted: bool
    optimizer_success: bool
    evaluation_error: str | None


def _parse_allocation_request(
    coordinate_names: Sequence[str],
    desired_generalized_effort: Sequence[float],
    coordinate_effort_units: Sequence[EffortUnit],
    effort_tolerances: Sequence[float],
    dt_s: float,
    force_equilibrium_tolerance_n: float,
    max_nfev: int,
) -> _ParsedAllocationRequest:
    if not coordinate_names or len(set(coordinate_names)) != len(coordinate_names):
        raise ValueError("coordinate names must be non-empty and unique")
    size = len(coordinate_names)
    if (
        len(desired_generalized_effort) != size
        or len(coordinate_effort_units) != size
        or len(effort_tolerances) != size
    ):
        raise ValueError(
            "effort values, units, and tolerances must match coordinate order"
        )
    if any(unit not in ("N", "N*m") for unit in coordinate_effort_units):
        raise ValueError("effort units must be explicitly 'N' or 'N*m'")
    target = np.asarray(
        [_number(v, "target effort") for v in desired_generalized_effort]
    )
    tolerances = np.asarray([_number(v, "effort tolerance") for v in effort_tolerances])
    if np.any(tolerances <= 0):
        raise ValueError("effort tolerances must be positive")
    dt = _number(dt_s, "dt")
    force_tolerance = _number(force_equilibrium_tolerance_n, "force tolerance")
    if dt <= 0 or force_tolerance <= 0:
        raise ValueError("dt and force tolerance must be positive")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    return _ParsedAllocationRequest(
        target=target,
        tolerances=tolerances,
        dt=dt,
        force_tolerance=force_tolerance,
        max_nfev=max_nfev,
        size=size,
    )


def _validate_recruitment_paths(
    paths: Sequence[MuscleRecruitmentPath],
    coordinate_names: Sequence[str],
    dt: float,
) -> None:
    if not paths or len({path.name for path in paths}) != len(paths):
        raise ValueError("paths must be non-empty with unique names")
    for path in paths:
        if not path.name or path.muscle is None or path.activation_dynamics is None:
            raise ValueError(
                "each path needs a name, Hill model, and activation dynamics"
            )
        if _number(path.muscle_tendon_length_m, "muscle-tendon length") <= 0:
            raise ValueError("muscle-tendon length must be positive")
        activation = _number(path.activation, "prior activation")
        previous = _number(path.previous_excitation, "previous excitation")
        rate = _number(path.max_excitation_rate_per_s, "excitation rate")
        low, high = (_number(v, "excitation bound") for v in path.excitation_bounds)
        floor = _number(path.activation_dynamics.min_activation, "activation floor")
        if not 0 <= activation <= 1 or not 0 <= previous <= 1:
            raise ValueError("activation and previous excitation must lie in [0, 1]")
        if not 0 < floor < 1 or rate < 0 or low < 0 or high > 1 or low > high:
            raise ValueError("invalid activation floor, rate, or excitation bounds")
        if any(name not in coordinate_names for name in path.moment_arms_m_or_rad):
            raise ValueError(
                "moment-arm coordinate is absent from the ordered coordinates"
            )
        if any(
            not math.isfinite(float(value))
            for value in path.moment_arms_m_or_rad.values()
        ):
            raise ValueError("moment arms must be finite")
        if _bounds(path, dt)[0] > _bounds(path, dt)[1]:
            raise ValueError(f"empty excitation/rate intersection for {path.name}")


def _seed_floor_plateaus(
    candidate: np.ndarray,
    free: Sequence[int],
    boxes: Sequence[tuple[float, float]],
    paths: Sequence[MuscleRecruitmentPath],
) -> np.ndarray:
    """Seed free controls above a flat activation-floor start when possible."""
    seeded = candidate.copy()
    for index in free:
        lower, upper = boxes[index]
        floor = float(paths[index].activation_dynamics.min_activation)
        if candidate[index] <= floor < upper:
            active_lower = max(lower, floor)
            seeded[index] = active_lower + 0.5 * (upper - active_lower)
    return seeded


class _BoundedPathEvaluator:
    """Invocation-local LRU cache of at most two observations for one path."""

    __slots__ = ("_path", "_dt_s", "_max_entries", "_cache")

    def __init__(
        self,
        path: MuscleRecruitmentPath,
        dt_s: float,
        max_entries: int = 2,
    ) -> None:
        self._path = path
        self._dt_s = dt_s
        self._max_entries = max_entries
        self._cache: dict[float, _Evaluation] = {}

    def __call__(self, excitation: float) -> _Evaluation:
        raw = float(excitation)
        if not math.isfinite(raw):
            return _evaluate(self._path, raw, self._dt_s)
        if raw in self._cache:
            entry = self._cache.pop(raw)
            self._cache[raw] = entry
            return _Evaluation(
                activation=entry.activation,
                effective_excitation=entry.effective_excitation,
                components=dict(entry.components),
                tendon_force=entry.tendon_force,
                equilibrium_residual_n=entry.equilibrium_residual_n,
            )
        result = _evaluate(self._path, raw, self._dt_s)
        if len(self._cache) >= self._max_entries:
            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[raw] = result
        return _Evaluation(
            activation=result.activation,
            effective_excitation=result.effective_excitation,
            components=dict(result.components),
            tendon_force=result.tendon_force,
            equilibrium_residual_n=result.equilibrium_residual_n,
        )


def _default_evaluator_factory(
    path: MuscleRecruitmentPath, dt_s: float
) -> Callable[[float], _Evaluation]:
    """Default invocation-local evaluator factory with a bounded cache."""
    return _BoundedPathEvaluator(path, dt_s)


def _uncached_path_evaluator_factory(
    path: MuscleRecruitmentPath, dt_s: float
) -> Callable[[float], _Evaluation]:
    """Uncached evaluator factory used for reference checks in unit tests."""
    return lambda excitation: _evaluate(path, float(excitation), dt_s)


class _ExcitationSearcher:
    """Bounded least-squares search over free excitation channels."""

    def __init__(
        self,
        paths: Sequence[MuscleRecruitmentPath],
        coordinate_names: Sequence[str],
        request: _ParsedAllocationRequest,
        boxes: list[tuple[float, float]],
        evaluator_factory: (
            Callable[[MuscleRecruitmentPath, float], Callable[[float], _Evaluation]]
            | None
        ) = None,
    ) -> None:
        self._paths = paths
        self._coordinate_names = coordinate_names
        self._request = request
        self._boxes = boxes
        factory = evaluator_factory or _default_evaluator_factory
        self._evaluators = [factory(path, request.dt) for path in paths]
        self._fixed = {i: low for i, (low, high) in enumerate(boxes) if low == high}
        self._free = [i for i, (low, high) in enumerate(boxes) if low < high]
        self._candidate0 = np.asarray(
            [
                min(max(path.previous_excitation, boxes[i][0]), boxes[i][1])
                for i, path in enumerate(paths)
            ]
        )
        self.evaluation_count = 0
        self.best_candidate: np.ndarray | None = None
        self.best_observation: tuple[np.ndarray, list[_Evaluation]] | None = None
        self.best_score = math.inf

    def _evaluate(self, values: np.ndarray) -> tuple[np.ndarray, list[_Evaluation]]:
        if self.evaluation_count >= self._request.max_nfev:
            raise _EvaluationBudgetExhausted(
                f"maximum candidate-evaluation budget {self._request.max_nfev} exhausted"
            )
        self.evaluation_count += 1
        details = [self._evaluators[i](float(value)) for i, value in enumerate(values)]
        effort = np.zeros(self._request.size, dtype=float)
        for path, detail in zip(self._paths, details, strict=True):
            for i, coordinate in enumerate(self._coordinate_names):
                effort[i] += (
                    path.moment_arms_m_or_rad.get(coordinate, 0.0) * detail.tendon_force
                )
        residual_vector = np.concatenate(
            (
                (effort - self._request.target) / self._request.tolerances,
                np.asarray(
                    [
                        d.equilibrium_residual_n / self._request.force_tolerance
                        for d in details
                    ]
                ),
            )
        )
        score = float(np.linalg.norm(residual_vector))
        if math.isfinite(score) and score < self.best_score:
            self.best_candidate = values.copy()
            self.best_observation = (effort, details)
            self.best_score = score
        return effort, details

    def _expand(self, values: np.ndarray) -> np.ndarray:
        result = self._candidate0.copy()
        for i, value in self._fixed.items():
            result[i] = value
        for i, value in zip(self._free, values, strict=True):
            result[i] = value
        return result

    def run(self) -> _SearchOutcome:
        status_message = "all channels fixed"
        budget_exhausted = False
        optimizer_success = True
        evaluation_error: str | None = None
        try:
            self._evaluate(self._candidate0)
            optimizer_start = _seed_floor_plateaus(
                self._candidate0, self._free, self._boxes, self._paths
            )
            if self._free:

                def objective(values: np.ndarray) -> np.ndarray:
                    effort, details = self._evaluate(self._expand(values))
                    return np.concatenate(
                        (
                            (effort - self._request.target) / self._request.tolerances,
                            np.asarray(
                                [
                                    d.equilibrium_residual_n
                                    / self._request.force_tolerance
                                    for d in details
                                ]
                            ),
                        )
                    )

                result = cast(Callable[..., OptimizeResult], least_squares)(
                    objective,
                    np.asarray([optimizer_start[i] for i in self._free]),
                    bounds=(
                        np.asarray([self._boxes[i][0] for i in self._free]),
                        np.asarray([self._boxes[i][1] for i in self._free]),
                    ),
                    max_nfev=self._request.max_nfev,
                    method="trf",
                    x_scale="jac",
                )
                status_message = f"{result.message}; optimizer_nfev={result.nfev}"
                optimizer_success = bool(result.success)
                budget_exhausted = not optimizer_success and (
                    getattr(result, "status", None) == 0
                    or "maximum" in str(result.message).lower()
                )
        except _EvaluationBudgetExhausted as exc:
            budget_exhausted = True
            status_message = str(exc)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
            evaluation_error = str(exc)
            status_message = f"candidate evaluation failed: {exc}"
        return _SearchOutcome(
            best_candidate=self.best_candidate,
            best_observation=self.best_observation,
            evaluation_count=self.evaluation_count,
            status_message=status_message,
            budget_exhausted=budget_exhausted,
            optimizer_success=optimizer_success,
            evaluation_error=evaluation_error,
        )


@dataclass(frozen=True)
class _AllocationPayload:
    excitations: dict[str, float]
    effective_excitations: dict[str, float]
    activations: dict[str, float]
    effective_bounds: dict[str, tuple[float, float]]
    muscle_forces_n: dict[str, dict[str, float]]
    generalized_effort: dict[str, float]
    residual_by_coordinate: dict[str, float]


def _allocation_payload(
    paths: Sequence[MuscleRecruitmentPath],
    coordinate_names: Sequence[str],
    best_candidate: np.ndarray,
    achieved: np.ndarray,
    details: list[_Evaluation],
    boxes: list[tuple[float, float]],
    target: np.ndarray,
) -> _AllocationPayload:
    residual = achieved - target
    return _AllocationPayload(
        excitations={p.name: float(best_candidate[i]) for i, p in enumerate(paths)},
        effective_excitations={
            p.name: details[i].effective_excitation for i, p in enumerate(paths)
        },
        activations={p.name: details[i].activation for i, p in enumerate(paths)},
        effective_bounds={p.name: boxes[i] for i, p in enumerate(paths)},
        muscle_forces_n={p.name: details[i].components for i, p in enumerate(paths)},
        generalized_effort={
            name: float(achieved[i]) for i, name in enumerate(coordinate_names)
        },
        residual_by_coordinate={
            name: float(residual[i]) for i, name in enumerate(coordinate_names)
        },
    )


def _finalize_allocation_result(
    paths: Sequence[MuscleRecruitmentPath],
    coordinate_names: Sequence[str],
    request: _ParsedAllocationRequest,
    boxes: list[tuple[float, float]],
    outcome: _SearchOutcome,
) -> MuscleAllocationResult:
    if outcome.best_candidate is None or outcome.best_observation is None:
        return MuscleAllocationResult(
            status="evaluation_failed",
            feasible=False,
            excitations={},
            effective_excitations={},
            activations={},
            effective_bounds={p.name: boxes[i] for i, p in enumerate(paths)},
            muscle_forces_n={},
            generalized_effort={},
            residual_by_coordinate={},
            nfev=outcome.evaluation_count,
            message=outcome.status_message,
        )
    achieved, details = outcome.best_observation
    payload = _allocation_payload(
        paths,
        coordinate_names,
        outcome.best_candidate,
        achieved,
        details,
        boxes,
        request.target,
    )
    if outcome.evaluation_error is not None:
        return MuscleAllocationResult(
            status="evaluation_failed",
            feasible=False,
            nfev=outcome.evaluation_count,
            message=outcome.status_message,
            excitations=payload.excitations,
            effective_excitations=payload.effective_excitations,
            activations=payload.activations,
            effective_bounds=payload.effective_bounds,
            muscle_forces_n=payload.muscle_forces_n,
            generalized_effort=payload.generalized_effort,
            residual_by_coordinate=payload.residual_by_coordinate,
        )
    residual = achieved - request.target
    feasible = bool(
        np.all(np.abs(residual) <= request.tolerances)
        and all(
            abs(item.equilibrium_residual_n) <= request.force_tolerance
            for item in details
        )
        and not outcome.budget_exhausted
        and outcome.optimizer_success
    )
    solver_failed = not outcome.optimizer_success and not outcome.budget_exhausted
    status: Literal[
        "feasible",
        "infeasible",
        "budget_exhausted",
        "solver_failed",
        "evaluation_failed",
    ] = (
        "budget_exhausted"
        if outcome.budget_exhausted
        else "solver_failed"
        if solver_failed
        else "feasible"
        if feasible
        else "infeasible"
    )
    return MuscleAllocationResult(
        status=status,
        feasible=feasible,
        nfev=outcome.evaluation_count,
        message=outcome.status_message,
        excitations=payload.excitations,
        effective_excitations=payload.effective_excitations,
        activations=payload.activations,
        effective_bounds=payload.effective_bounds,
        muscle_forces_n=payload.muscle_forces_n,
        generalized_effort=payload.generalized_effort,
        residual_by_coordinate=payload.residual_by_coordinate,
    )


def allocate_muscle_excitations(
    paths: Sequence[MuscleRecruitmentPath],
    coordinate_names: Sequence[str],
    desired_generalized_effort: Sequence[float],
    *,
    coordinate_effort_units: Sequence[EffortUnit],
    dt_s: float,
    effort_tolerances: Sequence[float],
    force_equilibrium_tolerance_n: float = 1e-4,
    max_nfev: int = 80,
) -> MuscleAllocationResult:
    """Allocate a bounded single-step excitation vector without applying it.

    A returned infeasible candidate is diagnostic only. Activation updates are
    computed from the immutable input state; callers decide whether to apply.
    Fixed excitation channels remain fixed and are evaluated without epsilon
    widening of the box.
    `max_nfev` caps complete candidate evaluations, including the initial
    candidate and finite-difference objective probes; cached final reporting
    does not perform an uncounted equilibrium solve.
    """
    request = _parse_allocation_request(
        coordinate_names,
        desired_generalized_effort,
        coordinate_effort_units,
        effort_tolerances,
        dt_s,
        force_equilibrium_tolerance_n,
        max_nfev,
    )
    _validate_recruitment_paths(paths, coordinate_names, request.dt)
    boxes = [_bounds(path, request.dt) for path in paths]
    outcome = _ExcitationSearcher(paths, coordinate_names, request, boxes).run()
    return _finalize_allocation_result(paths, coordinate_names, request, boxes, outcome)
