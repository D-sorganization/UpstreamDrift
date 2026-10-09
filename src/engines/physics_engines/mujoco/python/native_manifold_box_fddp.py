"""Bounded native floating-root BoxFDDP candidate on a MuJoCo manifold.

This F05d fixture is contact-free and has two direct hinge motors. Crocoddyl
uses the physical ``nx=17`` state with a ``ndx=16`` MuJoCo tangent; all
dynamics and Jacobians come from the same native Euler step. A candidate is
accepted only after an independent nonlinear native rollout beats fallback.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import crocoddyl
import mujoco as mj
import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.mujoco.python.box_fddp_tracking import (
    classify_native_box_candidate,
)
from src.engines.physics_engines.mujoco.python.native_manifold_calculus import (
    difference_jacobians,
    integration_jacobians,
    set_tracking_cost_derivatives,
)
from src.engines.physics_engines.mujoco.python.native_nmpc_tracking import (
    NativeNMPCTracking,
    run_native_direct_torque_tracking,
)
from src.engines.physics_engines.mujoco.python.native_tangent_derivative import (
    linearize_native_tangent_step,
)
from src.engines.physics_engines.mujoco.python.native_torque_replay import _CALLBACKS
from src.shared.python.motion_matching.bounded_nmpc import MPCCommandReceipt

Array = NDArray[np.float64]
_EPSILON = 1e-6
_BOUND_MARGIN = 1e-4


def _vector(values: Array, length: int, label: str) -> Array:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{label} requires {length} finite entries")
    return result


class NativeManifoldState(crocoddyl.StateAbstract):
    """Physical ``[qpos(9), qvel(8)]`` with MuJoCo tangent operations."""

    def __init__(self, model: Any) -> None:
        if (model.nq, model.nv, model.nu) != (9, 8, 2):
            raise ValueError("native manifold state requires nq=9/nv=8/nu=2")
        super().__init__(17, 16)
        self.model = model

    def _checked(self, x: Array) -> Array:
        state = _vector(x, self.nx, "native physical state")
        if not np.isclose(np.linalg.norm(state[3:7]), 1.0, atol=1e-12, rtol=0):
            raise ValueError("native quaternion must be normalized")
        return state

    def diff(self, x0: Array, x1: Array) -> Array:
        """Return configuration difference in native tangent plus velocity delta."""
        before = self._checked(x0)
        after = self._checked(x1)
        dq = np.empty(self.model.nv)
        mj.mj_differentiatePos(self.model, dq, 1.0, before[:9], after[:9])
        return np.r_[dq, after[9:] - before[9:]]

    def integrate(self, x: Array, dx: Array) -> Array:
        """Retract a 16-vector without Euclidean quaternion addition."""
        base = self._checked(x)
        tangent = _vector(dx, self.ndx, "native tangent")
        qpos = base[:9].copy()
        mj.mj_integratePos(self.model, qpos, tangent[:8], 1.0)
        return np.r_[qpos, base[9:] + tangent[8:]]

    def _jacobian(self, function: Any, size: int = 16) -> Array:
        result = np.empty((self.ndx, size))
        for column in range(size):
            direction = np.eye(size)[column] * _EPSILON
            result[:, column] = (function(direction) - function(-direction)) / (
                2 * _EPSILON
            )
        return result

    def Jdiff(
        self, x0: Array, x1: Array, firstsecond: Any = crocoddyl.Jcomponent.both
    ) -> list[Array]:
        before = self._checked(x0)
        after = self._checked(x1)

        return difference_jacobians(self, before, after, firstsecond)

    def Jintegrate(
        self, x: Array, dx: Array, firstsecond: Any = crocoddyl.Jcomponent.both
    ) -> list[Array]:
        base = self._checked(x)
        tangent = _vector(dx, self.ndx, "native tangent")
        return integration_jacobians(self, base, tangent, firstsecond)

    def JintegrateTransport(
        self, x: Array, dx: Array, Jin: Array, firstsecond: Any
    ) -> Array:
        jacobian = self.Jintegrate(x, dx, firstsecond)[0]
        return np.asarray(
            np.linalg.solve(jacobian, np.asarray(Jin, dtype=np.float64)),
            dtype=np.float64,
        )

    def zero(self) -> Array:
        result = np.zeros(self.nx)
        result[3] = 1.0
        return result

    def rand(self) -> Array:
        return self.integrate(self.zero(), np.random.default_rng().normal(size=16))


def _step(model: Any, physical: Array, command: Array) -> Array:
    data = mj.MjData(model)
    data.qpos[:] = physical[:9]
    data.qvel[:] = physical[9:]
    data.ctrl[:] = command
    mj.mj_step(model, data)
    if (
        data.ncon
        or not np.isfinite(data.qpos).all()
        or not np.isfinite(data.qvel).all()
    ):
        raise ValueError("native candidate contacted or diverged")
    return np.r_[data.qpos, data.qvel]


def _full_state(model: Any, physical: Array) -> Array:
    data = mj.MjData(model)
    data.qpos[:] = physical[:9]
    data.qvel[:] = physical[9:]
    values = np.empty(mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(model, data, values, mj.mjtState.mjSTATE_INTEGRATION)
    return values


def _compiled_model_sha256(model: Any) -> str:
    binary = np.empty(mj.mj_sizeModel(model), dtype=np.uint8)
    mj.mj_saveModel(model, buffer=binary)
    return hashlib.sha256(binary.tobytes()).hexdigest()


def _require_no_native_callbacks() -> None:
    if any(getattr(mj, "get_mjcb_" + name)() is not None for name in _CALLBACKS):
        raise ValueError("native callbacks are forbidden during control admission")


class NativeManifoldAction(crocoddyl.ActionModelAbstract):
    """One actual native Euler step with native discrete tangent Jacobians."""

    def __init__(
        self,
        state: NativeManifoldState,
        reference: Array,
        state_weights: Array,
        input_weights: Array,
    ) -> None:
        super().__init__(state, 2, 16)
        self.native_state = state
        self.reference = state._checked(reference).copy()
        self.state_weights = _vector(state_weights, 16, "state weights").copy()
        self.input_weights = _vector(input_weights, 2, "input weights").copy()
        if np.any(self.state_weights <= 0) or np.any(self.input_weights <= 0):
            raise ValueError("native weights must be positive")
        bounds = state.model.actuator_ctrlrange
        self.u_lb = bounds[:, 0] + _BOUND_MARGIN
        self.u_ub = bounds[:, 1] - _BOUND_MARGIN
        if np.any(self.u_lb >= self.u_ub):
            raise ValueError("native motor bounds need derivative interior")

    def calc(self, data: Any, x: Array, u: Array) -> None:
        physical = self.native_state._checked(x)
        command = _vector(u, 2, "native motor command")
        if np.any(command < self.u_lb) or np.any(command > self.u_ub):
            raise ValueError("native motor command exceeds admitted interior")
        data.xnext = _step(self.native_state.model, physical, command)
        error = self.native_state.diff(self.reference, data.xnext)
        data.cost = float(
            np.dot(self.state_weights, error**2)
            + np.dot(self.input_weights, command**2)
        )

    def calcDiff(self, data: Any, x: Array, u: Array) -> None:
        physical = self.native_state._checked(x)
        command = _vector(u, 2, "native motor command")
        discrete = linearize_native_tangent_step(
            self.native_state.model,
            _full_state(self.native_state.model, physical),
            command,
        )
        predicted = np.r_[discrete.next_qpos, discrete.next_qvel]
        if np.max(np.abs(self.native_state.diff(predicted, data.xnext))) > 1e-10:
            raise ValueError("native action and discrete derivative disagree")
        data.Fx = discrete.A
        data.Fu = discrete.B
        set_tracking_cost_derivatives(
            data,
            self.native_state,
            self.reference,
            self.state_weights,
            command,
            self.input_weights,
        )


class NativeManifoldTerminal(crocoddyl.ActionModelAbstract):
    """Terminal native manifold tracking cost without an actuator input."""

    def __init__(self, state: NativeManifoldState, reference: Array, weights: Array):
        super().__init__(state, 0, 16)
        self.native_state = state
        self.reference = state._checked(reference).copy()
        self.weights = _vector(weights, 16, "terminal weights").copy()

    def calc(self, data: Any, x: Array, u: Array | None = None) -> None:
        physical = self.native_state._checked(x)
        data.xnext = physical.copy()
        error = self.native_state.diff(self.reference, physical)
        data.cost = float(np.dot(self.weights, error**2))

    def calcDiff(self, data: Any, x: Array, u: Array | None = None) -> None:
        physical = self.native_state._checked(x)
        error = self.native_state.diff(self.reference, physical)
        jacobian = self.native_state.Jdiff(
            self.reference, physical, crocoddyl.Jcomponent.second
        )[0]
        data.Lx = 2 * jacobian.T @ (self.weights * error)
        data.Lxx = 2 * jacobian.T @ np.diag(self.weights) @ jacobian


@dataclass(frozen=True)
class NativeManifoldReceipt(MPCCommandReceipt):
    """Accepted or fallback native command with explicit solver timing."""

    solver_iterations: int = 0
    preparation_s: float = 0.0
    solve_s: float = 0.0
    solver_id: str = "crocoddyl-boxfddp"


class NativeManifoldBoxFDDP:
    """Bounded receding-horizon candidate with nonlinear native admission."""

    solver_id = "crocoddyl-boxfddp"

    def __init__(
        self,
        model: Any,
        references: Array,
        *,
        horizon_steps: int,
        max_iterations: int,
        max_wall_s: float,
    ) -> None:
        self.state = NativeManifoldState(model)
        self.references = np.asarray(references, dtype=np.float64)
        if self.references.ndim != 2 or self.references.shape[1] != 17:
            raise ValueError("native references must have physical nx=17")
        for reference in self.references:
            self.state._checked(reference)
        if (
            horizon_steps < 1
            or max_iterations < 1
            or not np.isfinite(max_wall_s)
            or max_wall_s <= 0
            or horizon_steps >= len(self.references)
        ):
            raise ValueError("native solve horizon or cooperative wall budget invalid")
        self.model = model
        self.horizon_steps = horizon_steps
        self.max_iterations = max_iterations
        self.max_wall_s = max_wall_s
        self.state_weights = np.r_[np.ones(6), 20.0, 20.0, np.full(8, 0.1)]
        self.terminal_weights = self.state_weights * 3
        self.input_weights = np.array([0.01, 0.01])
        _require_no_native_callbacks()
        linearize_native_tangent_step(
            model, _full_state(model, self.references[0]), np.zeros(2)
        )
        self.compiled_model_sha256 = _compiled_model_sha256(model)
        self._history: list[NativeManifoldReceipt] = []
        self._prior_plan: Array | None = None

    @property
    def applied_history(self) -> tuple[NativeManifoldReceipt, ...]:
        return tuple(self._history)

    def _score(self, index: int, initial: Array, plan: Array) -> float:
        if plan.shape != (self.horizon_steps, 2) or not np.isfinite(plan).all():
            return float("inf")
        ranges = self.model.actuator_ctrlrange
        if np.any(plan < ranges[:, 0]) or np.any(plan > ranges[:, 1]):
            return float("inf")
        state = initial.copy()
        total = 0.0
        try:
            for offset, command in enumerate(plan):
                state = _step(self.model, state, command)
                error = self.state.diff(self.references[index + offset + 1], state)
                total += float(
                    np.dot(self.state_weights, error**2)
                    + np.dot(self.input_weights, command**2)
                )
            terminal = self.state.diff(self.references[index + len(plan)], state)
            return total + float(np.dot(self.terminal_weights, terminal**2))
        except ValueError:
            return float("inf")

    def _problem(self, index: int, x0: Array, warm: Array) -> tuple[Any, list[Array]]:
        running = [
            NativeManifoldAction(
                self.state,
                self.references[index + offset + 1],
                self.state_weights,
                self.input_weights,
            )
            for offset in range(self.horizon_steps)
        ]
        terminal = NativeManifoldTerminal(
            self.state,
            self.references[index + self.horizon_steps],
            self.terminal_weights,
        )
        shooting = crocoddyl.ShootingProblem(x0, running, terminal)
        states = [x0.copy()]
        for command in warm:
            states.append(_step(self.model, states[-1], command))
        return shooting, states

    def _solve_candidate(
        self, index: int, state: Array, warm: Array, started: float
    ) -> tuple[bool, Array, int, float, float]:
        shooting, states = self._problem(index, state, warm)
        preparation_s = time.perf_counter() - started
        solver = crocoddyl.SolverBoxFDDP(shooting)
        solve_started = time.perf_counter()
        solved = solver.solve(states, list(warm), self.max_iterations, True)
        solve_s = time.perf_counter() - solve_started
        candidate = np.asarray(solver.us, dtype=np.float64).reshape(
            self.horizon_steps, 2
        )
        return bool(solved), candidate, int(solver.iter), preparation_s, solve_s

    def command_for_step(
        self,
        index: int,
        observed_state: Array,
        *,
        observation_time_s: float,
        current_time_s: float,
        cancelled: bool = False,
    ) -> NativeManifoldReceipt:
        """Apply only a post-limit native-verified first motor command."""
        started = time.perf_counter()
        _require_no_native_callbacks()
        state = self.state._checked(observed_state)
        if _compiled_model_sha256(self.model) != self.compiled_model_sha256:
            raise ValueError("native compiled model identity changed")
        model = self.model
        dt = float(model.opt.timestep)
        if (
            index < 0
            or index + self.horizon_steps >= len(self.references)
            or not np.isfinite([observation_time_s, current_time_s]).all()
            or current_time_s < observation_time_s
            or abs(current_time_s - index * dt) > 1e-9
        ):
            raise ValueError("native state or observation clock invalid")
        fallback = np.zeros(2)
        fallback_plan = np.zeros((self.horizon_steps, 2))
        fallback_objective = self._score(index, state, fallback_plan)
        if not np.isfinite(fallback_objective):
            raise ValueError("native fallback is infeasible")
        prior = self._prior_plan
        warm_started = prior is not None
        warm = np.vstack((prior[1:], prior[-1])) if prior is not None else fallback_plan
        status = "fallback_cancelled" if cancelled else "pending"
        if current_time_s - observation_time_s > 0:
            status = "fallback_stale_observation"
        applied = fallback
        objective: float | None = None
        iterations = 0
        preparation_s = 0.0
        solve_s = 0.0
        if status == "pending":
            try:
                solved, candidate, iterations, preparation_s, solve_s = (
                    self._solve_candidate(index, state, warm, started)
                )
                objective = self._score(index, state, candidate)
            except (RuntimeError, ValueError, TypeError, OverflowError):
                status = "fallback_solver_exception"
            else:
                status = classify_native_box_candidate(
                    elapsed_s=time.perf_counter() - started,
                    budget_s=self.max_wall_s,
                    solved=solved,
                    objective=objective,
                    fallback_objective=fallback_objective,
                )
                if status == "optimized":
                    applied = candidate[0]
                    self._prior_plan = np.array(candidate, copy=True)
        if status != "optimized":
            self._prior_plan = None
        receipt = NativeManifoldReceipt(
            status=status,
            applied=applied,
            fallback=fallback,
            objective=objective,
            fallback_objective=fallback_objective,
            evaluations=0,
            elapsed_s=time.perf_counter() - started,
            warm_started=warm_started,
            scenario_count=1,
            solver_iterations=iterations,
            preparation_s=preparation_s,
            solve_s=solve_s,
            solver_id=self.solver_id,
        )
        self._history.append(receipt)
        return receipt


class NativeManifoldSciPyShooting(NativeManifoldBoxFDDP):
    """Matched native-shooting comparator; shares the exact admission path."""

    solver_id = "scipy-slsqp-native-shooting"

    def _solve_candidate(
        self, index: int, state: Array, warm: Array, started: float
    ) -> tuple[bool, Array, int, float, float]:
        from scipy.optimize import Bounds, minimize

        lower = np.tile(
            self.model.actuator_ctrlrange[:, 0] + _BOUND_MARGIN, self.horizon_steps
        )
        upper = np.tile(
            self.model.actuator_ctrlrange[:, 1] - _BOUND_MARGIN, self.horizon_steps
        )
        bounds = Bounds(lower, upper)
        preparation_s = time.perf_counter() - started
        solve_started = time.perf_counter()
        # scipy-stubs 1.17.1.4 omits this valid SLSQP + Bounds overload;
        # native provider tests exercise the exact call on supported SciPy.
        result = minimize(  # type: ignore[call-overload]
            lambda flat: self._score(
                index, state, np.asarray(flat).reshape(self.horizon_steps, 2)
            ),
            warm.ravel(),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": self.max_iterations, "ftol": 1e-7},
        )
        solve_s = time.perf_counter() - solve_started
        candidate = np.asarray(result.x, dtype=np.float64).reshape(
            self.horizon_steps, 2
        )
        return bool(result.success), candidate, int(result.nit), preparation_s, solve_s


def run_native_manifold_box_fddp_tracking(
    model_path: str | Path,
    initial_integration_state: Array,
    controller: NativeManifoldBoxFDDP,
    *,
    steps: int,
    experiment_id: str,
) -> NativeNMPCTracking:
    """Drive the native plant, export applied torque and independently replay."""
    path = Path(model_path)
    model = mj.MjModel.from_xml_path(str(path))
    model.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_AUTORESET)
    if _compiled_model_sha256(model) != controller.compiled_model_sha256:
        raise ValueError("native execution model identity differs from controller")
    if steps + controller.horizon_steps > len(controller.references):
        raise ValueError("native reference does not cover the requested horizon")
    return run_native_direct_torque_tracking(
        path,
        model,
        initial_integration_state,
        controller,
        lambda data: np.r_[data.qpos, data.qvel],
        steps=steps,
        experiment_id=experiment_id,
    )
