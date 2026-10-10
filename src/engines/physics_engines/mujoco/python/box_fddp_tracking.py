"""Bounded BoxFDDP hinge controller with independent native-input boundary."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.native_box_fddp_hinge import (
    HingeDiscreteModel,
    linearize_native_hinge,
)
from src.engines.physics_engines.mujoco.python.native_nmpc_tracking import (
    NativeNMPCTracking,
    run_native_nmpc_tracking,
)
from src.shared.python.motion_matching.bounded_nmpc import (
    MPCCommandReceipt,
    MPCProblem,
)

Array: TypeAlias = NDArray[np.float64]


def classify_native_box_candidate(
    *,
    elapsed_s: float,
    budget_s: float,
    solved: bool,
    objective: float | None,
    fallback_objective: float,
) -> str:
    """Share the native solver's timing, feasibility and benefit decision."""
    if elapsed_s > budget_s:
        return "fallback_timeout"
    if not solved:
        return "fallback_solver_failure"
    if objective is None or not np.isfinite(objective):
        return "fallback_infeasible"
    if objective >= fallback_objective - 1e-9:
        return "fallback_no_benefit"
    return "optimized"


@dataclass(frozen=True)
class BoxFDDPConfig:
    """Predeclared horizon, iteration limit and cooperative wall budget."""

    horizon_steps: int
    max_iterations: int
    max_wall_s: float
    max_observation_age_s: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.horizon_steps < 1
            or self.max_iterations < 1
            or not np.isfinite(self.max_wall_s)
            or self.max_wall_s <= 0
            or not np.isfinite(self.max_observation_age_s)
            or self.max_observation_age_s < 0
        ):
            raise ValueError("BoxFDDP horizon or solve/observation budget invalid")


@dataclass(frozen=True)
class BoxFDDPCommandReceipt(MPCCommandReceipt):
    """Post-limit command plus separately timed compiled-solver evidence."""

    solver_iterations: int = 0
    preparation_s: float = 0.0
    solve_s: float = 0.0


def _combined_matrices(
    models: tuple[HingeDiscreteModel, HingeDiscreteModel],
) -> tuple[Array, Array]:
    A = np.zeros((4, 4), dtype=np.float64)
    B = np.empty((4, 1), dtype=np.float64)
    for index, model in enumerate(models):
        span = slice(index * 2, index * 2 + 2)
        A[span, span] = model.A
        B[span, :] = model.B
    return A, B


def _running_action(
    crocoddyl: Any,
    A: Array,
    B: Array,
    problem: MPCProblem,
    target: Array,
) -> Any:
    weight = np.diag(np.tile(problem.state_weights / 2, 2))
    target_pair = np.tile(target, 2)
    Q = 2 * A.T @ weight @ A
    R = 2 * (B.T @ weight @ B + np.diag(problem.input_weights))
    N = 2 * A.T @ weight @ B
    action = crocoddyl.ActionModelLQR(
        A,
        B,
        Q,
        R,
        N,
        np.zeros(4),
        -2 * A.T @ weight @ target_pair,
        -2 * B.T @ weight @ target_pair,
    )
    action.u_lb = problem.input_lower
    action.u_ub = problem.input_upper
    return action


def _terminal_action(crocoddyl: Any, problem: MPCProblem, target: Array) -> Any:
    weight = np.diag(np.tile(problem.terminal_weights / 2, 2))
    return crocoddyl.ActionModelLQR(
        np.eye(4),
        np.zeros((4, 0)),
        2 * weight,
        np.zeros((0, 0)),
        np.zeros((4, 0)),
        np.zeros(4),
        -2 * weight @ np.tile(target, 2),
        np.zeros(0),
    )


class BoxFDDPHingeController:
    """Receding-horizon BoxFDDP with robust post-solve admission/fallback."""

    def __init__(
        self,
        problem: MPCProblem,
        nominal: HingeDiscreteModel,
        scenario: HingeDiscreteModel,
        config: BoxFDDPConfig,
        fallback: Callable[[Array, float], Array],
    ) -> None:
        if (
            problem.nx != 2
            or problem.nu != 1
            or len(problem.scenario_steps) != 1
            or problem.max_input_change is not None
            or problem.state_units != ("rad", "rad/s")
            or problem.input_units != ("N*m",)
            or config.horizon_steps >= len(problem.target_states)
            or not np.isclose(problem.time_step_s, nominal.time_step_s, atol=1e-12)
            or not np.isclose(problem.time_step_s, scenario.time_step_s, atol=1e-12)
            or not callable(fallback)
        ):
            raise ValueError("BoxFDDP requires matched one-hinge scenario contract")
        probes = (
            (np.array([0.4, -0.2]), np.array([0.7])),
            (np.array([-0.5, 0.3]), np.array([-1.1])),
        )
        for model, step in zip((nominal, scenario), problem.models, strict=True):
            for state, effort in probes:
                if not np.allclose(
                    step(state, effort),
                    model.A @ state + model.B @ effort,
                    atol=1e-11,
                    rtol=0,
                ):
                    raise ValueError(
                        "declared step callback differs from native derivative"
                    )
        self.problem = problem
        self.models = (nominal, scenario)
        self.config = config
        self.fallback = fallback
        self._history: list[BoxFDDPCommandReceipt] = []
        self._prior_plan: Array | None = None

    @property
    def applied_history(self) -> tuple[BoxFDDPCommandReceipt, ...]:
        return tuple(self._history)

    def _score(self, index: int, state: Array, plan: Array) -> float:
        if plan.shape != (self.config.horizon_steps, 1) or not np.isfinite(plan).all():
            return float("inf")
        if np.any(plan < self.problem.input_lower - 1e-9) or np.any(
            plan > self.problem.input_upper + 1e-9
        ):
            return float("inf")
        costs: list[float] = []
        for model in self.models:
            predicted = state.copy()
            cost = 0.0
            for offset, effort in enumerate(plan):
                predicted = model.A @ predicted + model.B @ effort
                if np.any(predicted < self.problem.state_lower - 1e-9) or np.any(
                    predicted > self.problem.state_upper + 1e-9
                ):
                    return float("inf")
                error = predicted - self.problem.target_states[index + offset + 1]
                cost += float(
                    np.dot(self.problem.state_weights, error**2)
                    + np.dot(self.problem.input_weights, effort**2)
                )
            terminal = predicted - self.problem.target_states[index + len(plan)]
            costs.append(
                cost + float(np.dot(self.problem.terminal_weights, terminal**2))
            )
        return max(costs)

    def _safe_fallback(self, state: Array, time_s: float) -> Array:
        safe = np.asarray(self.fallback(state.copy(), time_s), dtype=np.float64)
        if safe.shape != (1,) or not np.isfinite(safe).all():
            raise ValueError("BoxFDDP fallback input must be finite direct torque")
        if np.any(safe < self.problem.input_lower) or np.any(
            safe > self.problem.input_upper
        ):
            raise ValueError("BoxFDDP fallback violates native torque limits")
        for model in self.models:
            next_state = model.A @ state + model.B @ safe
            if np.any(next_state < self.problem.state_lower) or np.any(
                next_state > self.problem.state_upper
            ):
                raise ValueError("BoxFDDP fallback violates next-state bounds")
        return safe

    def _shooting_problem(
        self, index: int, state: Array, warm: Array
    ) -> tuple[Any, list[Array]]:
        import crocoddyl

        A, B = _combined_matrices(self.models)
        x0 = np.tile(state, 2)
        running = [
            _running_action(
                crocoddyl,
                A,
                B,
                self.problem,
                self.problem.target_states[index + offset + 1],
            )
            for offset in range(self.config.horizon_steps)
        ]
        terminal = _terminal_action(
            crocoddyl,
            self.problem,
            self.problem.target_states[index + self.config.horizon_steps],
        )
        shooting = crocoddyl.ShootingProblem(x0, running, terminal)
        xs = [x0]
        for effort in warm:
            xs.append(A @ xs[-1] + B @ effort)
        return shooting, xs

    def command_for_step(
        self,
        index: int,
        observed_state: Array,
        *,
        observation_time_s: float,
        current_time_s: float,
        cancelled: bool = False,
    ) -> BoxFDDPCommandReceipt:
        """Return a feasible post-limit motor command with total-call timing."""
        import crocoddyl

        started = time.perf_counter()
        state = np.asarray(observed_state, dtype=np.float64)
        if (
            state.shape != (2,)
            or not np.isfinite(state).all()
            or index < 0
            or index + self.config.horizon_steps >= len(self.problem.target_states)
            or current_time_s < observation_time_s
            or not np.isfinite([current_time_s, observation_time_s]).all()
            or abs(current_time_s - index * self.problem.time_step_s) > 1e-9
        ):
            raise ValueError("BoxFDDP state or observation clock invalid")
        safe = self._safe_fallback(state, current_time_s)
        warm_started = self._prior_plan is not None
        warm = (
            np.vstack((self._prior_plan[1:], self._prior_plan[-1]))
            if self._prior_plan is not None
            else np.tile(safe, (self.config.horizon_steps, 1))
        )
        fallback_objective = self._score(
            index, state, np.tile(safe, (self.config.horizon_steps, 1))
        )
        applied = safe
        objective: float | None = None
        iterations = 0
        solve_s = 0.0
        preparation_s = 0.0
        status = (
            "fallback_cancelled"
            if cancelled
            else "fallback_stale_observation"
            if current_time_s - observation_time_s > self.config.max_observation_age_s
            else "pending"
        )
        if status == "pending":
            try:
                shooting, xs = self._shooting_problem(index, state, warm)
                preparation_s = time.perf_counter() - started
                solver = crocoddyl.SolverBoxFDDP(shooting)
                solve_started = time.perf_counter()
                solved = solver.solve(xs, list(warm), self.config.max_iterations, True)
                solve_s = time.perf_counter() - solve_started
                iterations = int(solver.iter)
                candidate = np.asarray(solver.us, dtype=np.float64).reshape(
                    self.config.horizon_steps, 1
                )
                objective = self._score(index, state, candidate)
            except (RuntimeError, ValueError, TypeError, OverflowError):
                status = "fallback_solver_exception"
            else:
                status = classify_native_box_candidate(
                    elapsed_s=time.perf_counter() - started,
                    budget_s=self.config.max_wall_s,
                    solved=bool(solved),
                    objective=objective,
                    fallback_objective=fallback_objective,
                )
                if status == "optimized":
                    applied = candidate[0]
                    self._prior_plan = np.array(candidate, copy=True)
        if status != "optimized":
            self._prior_plan = None
        receipt = BoxFDDPCommandReceipt(
            status=status,
            applied=applied,
            fallback=safe,
            objective=objective,
            fallback_objective=fallback_objective,
            evaluations=0,  # Native solver exposes iterations; no function-evaluation count.
            elapsed_s=time.perf_counter() - started,
            warm_started=warm_started,
            scenario_count=2,
            solver_iterations=iterations,
            preparation_s=preparation_s,
            solve_s=solve_s,
        )
        self._history.append(receipt)
        return receipt


def run_native_box_fddp_tracking(
    nominal_path: str | Path,
    execution_path: str | Path,
    initial_integration_state: Array,
    controller: BoxFDDPHingeController,
    *,
    steps: int,
    experiment_id: str,
) -> NativeNMPCTracking:
    """Re-admit both loaded native models before controlled and frozen replay."""
    import mujoco as mj

    for path, declared in zip(
        (nominal_path, execution_path), controller.models, strict=True
    ):
        loaded = linearize_native_hinge(mj.MjModel.from_xml_path(str(path)))
        if not all(
            np.allclose(
                getattr(loaded, name), getattr(declared, name), atol=1e-12, rtol=0
            )
            for name in ("A", "B", "inertia_kg_m2", "damping_nm_s_rad", "time_step_s")
        ):
            raise ValueError("loaded native model derivative identity differs")
    return run_native_nmpc_tracking(
        execution_path,
        initial_integration_state,
        controller,
        steps=steps,
        experiment_id=experiment_id,
    )
