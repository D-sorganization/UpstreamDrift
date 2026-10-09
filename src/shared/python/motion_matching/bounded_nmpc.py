"""Optional bounded shooting NMPC with a verified actuator fallback (F05).

This controller plans with a caller-supplied discrete plant. It never advances
the execution plant; the caller must apply each returned post-limit command at
the native integration boundary and separately retain an independent replay.
The optimizer is cooperative-budgeted, so measured latency is evidence rather
than a hard real-time guarantee.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, OptimizeResult, minimize

Array: TypeAlias = NDArray[np.float64]
StepFunction: TypeAlias = Callable[[Array, Array], Array]
FallbackFunction: TypeAlias = Callable[[Array, float], Array]


def _frozen(values: Array) -> Array:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _vector(values: Array, length: int, name: str) -> Array:
    result = np.asarray(values, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} requires {length} finite entries")
    return _frozen(result)


@dataclass(frozen=True)
class MPCProblem:
    """Predeclared tangent-state task and robust discrete prediction models."""

    step: StepFunction
    time_step_s: float
    target_states: Array
    state_weights: Array
    terminal_weights: Array
    input_weights: Array
    input_lower: Array
    input_upper: Array
    state_lower: Array
    state_upper: Array
    state_component_ids: tuple[str, ...]
    state_units: tuple[str, ...]
    input_channel_ids: tuple[str, ...]
    input_units: tuple[str, ...]
    scenario_steps: tuple[StepFunction, ...] = ()
    max_input_change: Array | None = None
    state_identity: str = "euclidean_tangent/v1"

    def __post_init__(self) -> None:
        targets = np.asarray(self.target_states, dtype=float)
        if (
            targets.ndim != 2
            or targets.shape[0] < 2
            or targets.shape[1] < 1
            or not np.isfinite(targets).all()
        ):
            raise ValueError("target states require a finite time-by-state matrix")
        nx = targets.shape[1]
        nu = len(self.input_channel_ids)
        if (
            nu < 1
            or any(not name for name in self.input_channel_ids)
            or len(set(self.input_channel_ids)) != nu
        ):
            raise ValueError("input channel identities must be non-empty and unique")
        if (
            len(self.state_component_ids) != nx
            or len(set(self.state_component_ids)) != nx
            or any(not name for name in self.state_component_ids)
        ):
            raise ValueError("state component identities must be ordered and unique")
        if len(self.state_units) != nx or any(not unit for unit in self.state_units):
            raise ValueError("state units must be explicitly bound to every component")
        if self.input_units != ("N*m",) * nu:
            raise ValueError("input units require direct actuator torque in N*m")
        if self.state_identity != "euclidean_tangent/v1":
            raise ValueError("only explicit Euclidean tangent-state MPC is supported")
        if (
            not np.isfinite(self.time_step_s)
            or self.time_step_s <= 0.0
            or not callable(self.step)
            or any(not callable(model) for model in self.scenario_steps)
        ):
            raise ValueError("step, clock, model and state identity must be valid")
        object.__setattr__(self, "target_states", _frozen(targets))
        for name, size in (
            ("state_weights", nx),
            ("terminal_weights", nx),
            ("input_weights", nu),
            ("input_lower", nu),
            ("input_upper", nu),
            ("state_lower", nx),
            ("state_upper", nx),
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), size, name))
        if (
            np.any(self.state_weights < 0.0)
            or np.any(self.terminal_weights < 0.0)
            or np.any(self.input_weights < 0.0)
            or not np.any(self.state_weights > 0.0)
            or np.any(self.input_lower >= self.input_upper)
            or np.any(self.state_lower >= self.state_upper)
        ):
            raise ValueError("weights and hard state/input bounds are invalid")
        if self.max_input_change is not None:
            change = _vector(self.max_input_change, nu, "max_input_change")
            if np.any(change <= 0.0):
                raise ValueError("input slew bounds must be positive")
            object.__setattr__(self, "max_input_change", change)

    @property
    def nx(self) -> int:
        return self.target_states.shape[1]

    @property
    def nu(self) -> int:
        return len(self.input_channel_ids)

    @property
    def models(self) -> tuple[StepFunction, ...]:
        return (self.step, *self.scenario_steps)


@dataclass(frozen=True)
class MPCConfig:
    """Fixed solve budget and stale-observation policy, selected before trials."""

    horizon_steps: int
    max_evaluations: int
    max_wall_s: float
    max_observation_age_s: float = 0.0
    solver_max_iterations: int = 80
    feasibility_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        if self.horizon_steps < 1 or self.max_evaluations < 1:
            raise ValueError("horizon and evaluation budget must be positive")
        if self.solver_max_iterations < 1:
            raise ValueError("solver iteration budget must be positive")
        if (
            not np.isfinite(self.max_wall_s)
            or self.max_wall_s <= 0.0
            or not np.isfinite(self.max_observation_age_s)
            or self.max_observation_age_s < 0.0
            or not np.isfinite(self.feasibility_tolerance)
            or self.feasibility_tolerance < 0.0
        ):
            raise ValueError("runtime, observation and feasibility policy invalid")


@dataclass(frozen=True)
class MPCCommandReceipt:
    """Executed command and failed-solve provenance for one native step."""

    status: str
    applied: Array
    fallback: Array
    objective: float | None
    fallback_objective: float | None
    evaluations: int
    elapsed_s: float
    warm_started: bool
    scenario_count: int
    input_boundary: str = "post_limit_actuator_command"
    information_pattern: str = "exact_simulated_state"

    def __post_init__(self) -> None:
        object.__setattr__(self, "applied", _frozen(self.applied))
        object.__setattr__(self, "fallback", _frozen(self.fallback))


class _BudgetExpired(Exception):
    pass


class _SolveCancelled(Exception):
    pass


class BoundedNMPC:
    """Warm-started robust direct-shooting MPC with one safe fallback path."""

    def __init__(
        self,
        problem: MPCProblem,
        config: MPCConfig,
        fallback: FallbackFunction,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if config.horizon_steps >= problem.target_states.shape[0]:
            raise ValueError("planning horizon exceeds target trajectory")
        if not callable(fallback) or not callable(clock):
            raise ValueError("fallback and monotonic clock must be callable")
        self.problem = problem
        self.config = config
        self.fallback = fallback
        self.clock = clock
        self._history: list[MPCCommandReceipt] = []
        self._prior_plan: Array | None = None

    @property
    def applied_history(self) -> tuple[MPCCommandReceipt, ...]:
        return tuple(self._history)

    def _next(self, model: StepFunction, state: Array, effort: Array) -> Array:
        result = np.asarray(model(state.copy(), effort.copy()), dtype=float)
        if result.shape != (self.problem.nx,) or not np.isfinite(result).all():
            raise ValueError("prediction model returned invalid finite state")
        return result

    def _input_margin(self, effort: Array, previous: Array | None) -> Array:
        margins = [
            effort - self.problem.input_lower,
            self.problem.input_upper - effort,
        ]
        if previous is not None and self.problem.max_input_change is not None:
            margins.extend(
                (
                    self.problem.max_input_change - (effort - previous),
                    self.problem.max_input_change + (effort - previous),
                )
            )
        return np.concatenate(margins)

    def _rollout(
        self, index: int, state: Array, plan: Array, previous: Array | None
    ) -> tuple[float, Array]:
        costs: list[float] = []
        margins: list[Array] = []
        for model in self.problem.models:
            predicted = state.copy()
            prior = previous
            cost = 0.0
            for offset, effort in enumerate(plan):
                margins.append(self._input_margin(effort, prior))
                predicted = self._next(model, predicted, effort)
                margins.extend(
                    (
                        predicted - self.problem.state_lower,
                        self.problem.state_upper - predicted,
                    )
                )
                error = predicted - self.problem.target_states[index + offset + 1]
                cost += float(
                    np.dot(self.problem.state_weights, error**2)
                    + np.dot(self.problem.input_weights, effort**2)
                )
                prior = effort
            terminal_error = predicted - self.problem.target_states[index + len(plan)]
            cost += float(np.dot(self.problem.terminal_weights, terminal_error**2))
            costs.append(cost)
        return max(costs), np.concatenate(margins)

    def _safe_fallback(
        self, state: Array, time_s: float, previous: Array | None
    ) -> Array:
        command = np.asarray(self.fallback(state.copy(), time_s), dtype=float)
        if command.shape != (self.problem.nu,) or not np.isfinite(command).all():
            raise ValueError("fallback command has invalid actuator shape or values")
        tol = self.config.feasibility_tolerance
        if np.min(self._input_margin(command, previous)) < -tol:
            raise ValueError("fallback violates actuator bounds or slew policy")
        for model in self.problem.models:
            next_state = self._next(model, state, command)
            if (
                np.min(next_state - self.problem.state_lower) < -tol
                or np.min(self.problem.state_upper - next_state) < -tol
            ):
                raise ValueError("fallback violates next-state bounds")
        return command

    def _validate_step_clock(
        self, index: int, observation_time_s: float, current_time_s: float
    ) -> None:
        targets = self.problem.target_states
        if (
            index < 0
            or index + self.config.horizon_steps >= targets.shape[0]
            or not np.isfinite(observation_time_s)
            or not np.isfinite(current_time_s)
            or current_time_s < observation_time_s
            or abs(current_time_s - index * self.problem.time_step_s) > 1e-9
        ):
            raise ValueError("step index, state clock or observation clock invalid")

    def _candidate_outcome(
        self,
        index: int,
        state: Array,
        previous: Array | None,
        result: OptimizeResult,
        horizon: int,
        fallback_objective: float,
    ) -> tuple[str, float | None, Array | None]:
        candidate = np.asarray(result.x, dtype=float).reshape(horizon, -1)
        if (
            candidate.shape != (horizon, self.problem.nu)
            or not np.isfinite(candidate).all()
        ):
            return "fallback_solver_failure", None, None
        objective, margins = self._rollout(index, state, candidate, previous)
        if np.min(margins) < -self.config.feasibility_tolerance:
            return "fallback_infeasible", objective, None
        if not result.success:
            return "fallback_solver_failure", objective, None
        if objective >= fallback_objective - 1e-9:
            return "fallback_no_benefit", objective, None
        return "optimized", objective, candidate

    def command_for_step(
        self,
        index: int,
        observed_state: Array,
        *,
        observation_time_s: float,
        current_time_s: float,
        cancelled: bool = False,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> MPCCommandReceipt:
        """Optimize one step; return only a verified post-limit actuator input."""
        started = float(self.clock())
        state = _vector(observed_state, self.problem.nx, "observed_state")
        self._validate_step_clock(index, observation_time_s, current_time_s)
        previous = self._history[-1].applied if self._history else None
        safe = self._safe_fallback(state, current_time_s, previous)
        warm = self._prior_plan is not None
        evaluations = 0
        objective: float | None = None
        fallback_objective: float | None = None
        applied = safe
        status: str
        age = current_time_s - observation_time_s
        if age > self.config.max_observation_age_s:
            status = "fallback_stale_observation"
        elif cancelled or (cancel_requested is not None and cancel_requested()):
            status = "fallback_cancelled"
        else:
            horizon = self.config.horizon_steps
            initial = (
                np.vstack((self._prior_plan[1:], self._prior_plan[-1]))
                if warm and self._prior_plan is not None
                else np.tile(safe, (horizon, 1))
            )
            fallback_objective, _ = self._rollout(
                index, state, np.tile(safe, (horizon, 1)), previous
            )

            def check_budget() -> None:
                if cancel_requested is not None and cancel_requested():
                    raise _SolveCancelled
                if (
                    evaluations >= self.config.max_evaluations
                    or self.clock() - started > self.config.max_wall_s
                ):
                    raise _BudgetExpired

            def evaluate(flat: Array) -> tuple[float, Array]:
                nonlocal evaluations
                check_budget()
                evaluations += 1
                return self._rollout(
                    index, state, np.asarray(flat).reshape(horizon, -1), previous
                )

            try:
                lower = np.tile(self.problem.input_lower, horizon)
                upper = np.tile(self.problem.input_upper, horizon)
                result = minimize(
                    lambda flat: evaluate(flat)[0],
                    initial.ravel(),
                    method="SLSQP",
                    bounds=Bounds(lower, upper),
                    constraints=[
                        {"type": "ineq", "fun": lambda flat: evaluate(flat)[1]}
                    ],
                    options={
                        "maxiter": self.config.solver_max_iterations,
                        "ftol": 1e-7,
                    },
                )
                check_budget()
                status, objective, accepted = self._candidate_outcome(
                    index, state, previous, result, horizon, fallback_objective
                )
                if accepted is not None:
                    applied = accepted[0]
                    self._prior_plan = _frozen(accepted)
            except _BudgetExpired:
                status = "fallback_timeout"
            except _SolveCancelled:
                status = "fallback_cancelled"
        if status != "optimized":
            self._prior_plan = None
        elapsed = max(0.0, float(self.clock()) - started)
        receipt = MPCCommandReceipt(
            status=status,
            applied=applied,
            fallback=safe,
            objective=objective,
            fallback_objective=fallback_objective,
            evaluations=evaluations,
            elapsed_s=elapsed,
            warm_started=warm,
            scenario_count=len(self.problem.models),
        )
        self._history.append(receipt)
        return receipt
