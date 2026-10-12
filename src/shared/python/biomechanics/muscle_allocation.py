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
from src.shared.python.biomechanics.muscle_force_evaluation import (
    evaluate_quasistatic_muscle_force,
)

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
    if not paths or len({path.name for path in paths}) != len(paths):
        raise ValueError("paths must be non-empty with unique names")
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

    boxes = [_bounds(path, dt) for path in paths]
    fixed = {i: low for i, (low, high) in enumerate(boxes) if low == high}
    free = [i for i, (low, high) in enumerate(boxes) if low < high]
    candidate0 = np.asarray(
        [
            min(max(path.previous_excitation, boxes[i][0]), boxes[i][1])
            for i, path in enumerate(paths)
        ]
    )
    evaluation_count = 0
    best_candidate: np.ndarray | None = None
    best_observation: tuple[np.ndarray, list[_Evaluation]] | None = None
    best_score = math.inf

    def evaluate(values: np.ndarray) -> tuple[np.ndarray, list[_Evaluation]]:
        nonlocal evaluation_count, best_candidate, best_observation, best_score
        if evaluation_count >= max_nfev:
            raise _EvaluationBudgetExhausted(
                f"maximum candidate-evaluation budget {max_nfev} exhausted"
            )
        evaluation_count += 1
        details = [
            _evaluate(path, float(value), dt)
            for path, value in zip(paths, values, strict=True)
        ]
        effort = np.zeros(size, dtype=float)
        for path, detail in zip(paths, details, strict=True):
            for i, coordinate in enumerate(coordinate_names):
                effort[i] += (
                    path.moment_arms_m_or_rad.get(coordinate, 0.0) * detail.tendon_force
                )
        residual_vector = np.concatenate(
            (
                (effort - target) / tolerances,
                np.asarray(
                    [d.equilibrium_residual_n / force_tolerance for d in details]
                ),
            )
        )
        score = float(np.linalg.norm(residual_vector))
        if math.isfinite(score) and score < best_score:
            best_candidate = values.copy()
            best_observation = (effort, details)
            best_score = score
        return effort, details

    def expand(values: np.ndarray) -> np.ndarray:
        result = candidate0.copy()
        for i, value in fixed.items():
            result[i] = value
        for i, value in zip(free, values, strict=True):
            result[i] = value
        return result

    status_message = "all channels fixed"
    budget_exhausted = False
    optimizer_success = True
    evaluation_error: str | None = None
    try:
        evaluate(candidate0)
        if free:

            def objective(values: np.ndarray) -> np.ndarray:
                effort, details = evaluate(expand(values))
                return np.concatenate(
                    (
                        (effort - target) / tolerances,
                        np.asarray(
                            [
                                d.equilibrium_residual_n / force_tolerance
                                for d in details
                            ]
                        ),
                    )
                )

            # SciPy 1.15 exposes this public function but mypy resolves the
            # same-named submodule instead of the package re-export.
            result = cast(Callable[..., OptimizeResult], least_squares)(
                objective,
                np.asarray([candidate0[i] for i in free]),
                bounds=(
                    np.asarray([boxes[i][0] for i in free]),
                    np.asarray([boxes[i][1] for i in free]),
                ),
                max_nfev=max_nfev,
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

    if best_candidate is None or best_observation is None:
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
            nfev=evaluation_count,
            message=status_message,
        )
    if evaluation_error is not None:
        return MuscleAllocationResult(
            status="evaluation_failed",
            feasible=False,
            excitations={p.name: float(best_candidate[i]) for i, p in enumerate(paths)},
            effective_excitations={
                p.name: best_observation[1][i].effective_excitation
                for i, p in enumerate(paths)
            },
            activations={
                p.name: best_observation[1][i].activation for i, p in enumerate(paths)
            },
            effective_bounds={p.name: boxes[i] for i, p in enumerate(paths)},
            muscle_forces_n={
                p.name: best_observation[1][i].components for i, p in enumerate(paths)
            },
            generalized_effort={
                name: float(best_observation[0][i])
                for i, name in enumerate(coordinate_names)
            },
            residual_by_coordinate={
                name: float(best_observation[0][i] - target[i])
                for i, name in enumerate(coordinate_names)
            },
            nfev=evaluation_count,
            message=status_message,
        )
    achieved, details = best_observation
    residual = achieved - target
    feasible = bool(
        np.all(np.abs(residual) <= tolerances)
        and all(abs(item.equilibrium_residual_n) <= force_tolerance for item in details)
        and not budget_exhausted
        and optimizer_success
    )
    solver_failed = not optimizer_success and not budget_exhausted
    status: Literal[
        "feasible",
        "infeasible",
        "budget_exhausted",
        "solver_failed",
        "evaluation_failed",
    ] = (
        "budget_exhausted"
        if budget_exhausted
        else "solver_failed"
        if solver_failed
        else "feasible"
        if feasible
        else "infeasible"
    )
    return MuscleAllocationResult(
        status=status,
        feasible=feasible,
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
        nfev=evaluation_count,
        message=status_message,
    )
