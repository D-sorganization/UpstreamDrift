"""Crocoddyl actions for one global full-body Bernstein control vector."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral, Real
from typing import Any, NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray

from src.shared.python.optimization.crocoddyl_backend import (
    CrocoddylNotAvailableError,
    crocoddyl_stack_healthy,
    require_crocoddyl,
)

from .full_body_step import NativeFullBodyStep

Array = NDArray[np.float64]
_SYMMETRY_RELATIVE_TOLERANCE = 1e-10


def _readonly(values: Any, label: str) -> Array:
    result = np.array(values, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PolynomialCostQuadratic:
    """A scalar cost and local quadratic model in canonical ``[x, p]`` order.

    Hessian symmetry uses a numerical relative tolerance of ``1e-10`` times
    ``max(1, max(abs(hessian)))``. This is input sanitation, not a physical
    acceptance threshold. Indefinite symmetric Hessians remain valid.
    """

    value: float
    gradient: Array
    hessian: Array

    def __post_init__(self) -> None:
        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, Real)
            or not np.isfinite(self.value)
        ):
            raise ValueError("quadratic value must be finite")
        gradient = _readonly(self.gradient, "quadratic gradient")
        hessian = _readonly(self.hessian, "quadratic hessian")
        if gradient.ndim != 1 or hessian.shape != (gradient.size, gradient.size):
            raise ValueError("quadratic gradient and hessian shape are inconsistent")
        scale = max(1.0, float(np.max(np.abs(hessian), initial=0.0)))
        asymmetry = float(np.max(np.abs(hessian - hessian.T), initial=0.0))
        if asymmetry > _SYMMETRY_RELATIVE_TOLERANCE * scale:
            raise ValueError("quadratic hessian must be symmetric")
        symmetric = _readonly(0.5 * (hessian + hessian.T), "quadratic hessian")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "hessian", symmetric)


class PolynomialCost(Protocol):
    """Injected scalar cost on physical state and global coefficients."""

    def value(self, x: Array, p: Array, time_s: float) -> float: ...

    def quadraticize(
        self, x: Array, p: Array, time_s: float
    ) -> PolynomialCostQuadratic: ...


class CrocoddylSmoothnessError(RuntimeError):
    """Raised when an RK4 action reaches a nondifferentiable contact branch."""


class PolynomialWarmStart(NamedTuple):
    """Crocoddyl-compatible, variable-control-dimension initial trajectories."""

    xs: list[Array]
    us: list[Array]


@dataclass(frozen=True)
class PolynomialShootingDiagnostics:
    """Raw defects and independent replay values without acceptance policy."""

    coefficients: Array
    initial_physical_defect: Array
    lift_physical_defect: Array
    lift_coefficient_defect: Array
    flow_physical_defects: tuple[Array, ...]
    coefficient_continuity_defects: tuple[Array, ...]
    coefficient_node_mismatches: tuple[Array, ...]
    lower_bound_violation: Array
    upper_bound_violation: Array
    replay_states: tuple[Array, ...]
    replay_physical_gaps: tuple[Array, ...]
    node_cost: float
    replay_cost: float
    cost_gap: float
    max_physical_defect: float
    max_coefficient_defect: float
    max_coefficient_node_mismatch: float
    max_bound_violation: float
    max_replay_gap: float


class _DiagnosticParts(NamedTuple):
    coefficients: Array
    initial_physical: Array
    lift_physical: Array
    lift_coefficients: Array
    flow_physical: Sequence[Array]
    continuity: Sequence[Array]
    mismatches: Sequence[Array]
    lower_violation: Array
    upper_violation: Array
    replay: Sequence[Array]
    replay_gaps: Sequence[Array]
    node_cost: float
    replay_cost: float


class _StepBlocks(NamedTuple):
    next_state: Array
    dstate: Array
    dcoefficients: Array
    differentiable: bool


@lru_cache(maxsize=1)
def _qualified_crocoddyl() -> Any:
    crocoddyl = require_crocoddyl()
    healthy, reason = crocoddyl_stack_healthy()
    if not healthy:
        raise CrocoddylNotAvailableError(reason)
    return crocoddyl


def _vector(value: Any, size: int, label: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},)")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


def _matrix(value: Any, shape: tuple[int, int], label: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


def _step_value(
    stepper: NativeFullBodyStep,
    state: Array,
    parameters: Array,
    time_s: float,
    dt_s: float,
    physical: int,
) -> Array:
    result = stepper.step(state, parameters, time_s=time_s, dt_s=dt_s)
    return _vector(result, physical, "RK4 next_state")


def _step_blocks(
    stepper: NativeFullBodyStep,
    state: Array,
    parameters: Array,
    time_s: float,
    dt_s: float,
    physical: int,
    coefficient_count: int,
) -> _StepBlocks:
    result = stepper.linearize(state, parameters, time_s=time_s, dt_s=dt_s)
    return _StepBlocks(
        _vector(result.next_state, physical, "RK4 next_state"),
        _matrix(result.dnext_dstate, (physical, physical), "RK4 state Jacobian"),
        _matrix(
            result.dnext_dcoefficients,
            (physical, coefficient_count),
            "RK4 coefficient Jacobian",
        ),
        bool(result.differentiable),
    )


def _empty_control(value: Any) -> None:
    if value is None:
        return
    _vector(value, 0, "autonomous control")


def _cost_value(cost: PolynomialCost, x: Array, p: Array, time_s: float) -> float:
    value = cost.value(x.copy(), p.copy(), time_s)
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError("cost value must be finite")
    return float(value)


def _finite_cost(value: float, label: str) -> float:
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _quadratic(
    cost: PolynomialCost, x: Array, p: Array, time_s: float, dimension: int
) -> PolynomialCostQuadratic:
    result = cost.quadraticize(x.copy(), p.copy(), time_s)
    if not isinstance(result, PolynomialCostQuadratic):
        raise TypeError("quadraticize must return PolynomialCostQuadratic")
    if result.gradient.shape != (dimension,):
        raise ValueError(f"quadratic shape must match augmented dimension {dimension}")
    return result


def _clear_derivatives(data: Any) -> None:
    for name in ("Fx", "Fu", "Lx", "Lu", "Lxx", "Lxu", "Luu"):
        getattr(data, name).fill(0.0)


def _lift_action_type(crocoddyl: Any) -> type[Any]:
    class LiftAction(crocoddyl.ActionModelAbstract):
        def __init__(self, state: Any, physical: int, parameters: int) -> None:
            super().__init__(state, parameters, 0)
            self.physical = physical
            self.parameters = parameters

        def calc(self, data: Any, state: Any, control: Any = None) -> None:
            z = _vector(state, self.state.nx, "augmented state")
            p = _vector(control, self.parameters, "lift control")
            data.xnext[:] = z
            data.xnext[self.physical :] = p
            data.cost = 0.0

        def calcDiff(self, data: Any, state: Any, control: Any = None) -> None:
            self.calc(data, state, control)
            _clear_derivatives(data)
            data.Fx[: self.physical, : self.physical] = np.eye(self.physical)
            data.Fu[self.physical :, :] = np.eye(self.parameters)

    return LiftAction


def _flow_action_type(crocoddyl: Any) -> type[Any]:
    class FlowAction(crocoddyl.ActionModelAbstract):
        def __init__(
            self,
            state: Any,
            stepper: NativeFullBodyStep,
            cost: PolynomialCost,
            time_s: float,
            dt_s: float,
            physical: int,
            parameters: int,
        ) -> None:
            super().__init__(state, 0, 0)
            self.stepper = stepper
            self.cost_model = cost
            self.time_s = time_s
            self.dt_s = dt_s
            self.physical = physical
            self.parameters = parameters

        def calc(self, data: Any, state: Any, control: Any = None) -> None:
            _empty_control(control)
            z = _vector(state, self.state.nx, "augmented state")
            x, p = z[: self.physical], z[self.physical :]
            data.xnext[: self.physical] = _step_value(
                self.stepper, x, p, self.time_s, self.dt_s, self.physical
            )
            data.xnext[self.physical :] = p
            data.cost = self.dt_s * _cost_value(self.cost_model, x, p, self.time_s)

        def calcDiff(self, data: Any, state: Any, control: Any = None) -> None:
            _empty_control(control)
            z = _vector(state, self.state.nx, "augmented state")
            x, p = z[: self.physical], z[self.physical :]
            linearization = _step_blocks(
                self.stepper,
                x,
                p,
                self.time_s,
                self.dt_s,
                self.physical,
                self.parameters,
            )
            if not linearization.differentiable:
                raise CrocoddylSmoothnessError(
                    "RK4 action is not differentiable at a contact branch"
                )
            quadratic = _quadratic(self.cost_model, x, p, self.time_s, self.state.nx)
            data.xnext[: self.physical] = linearization.next_state
            data.xnext[self.physical :] = p
            data.cost = self.dt_s * quadratic.value
            _clear_derivatives(data)
            data.Fx[: self.physical, : self.physical] = linearization.dstate
            data.Fx[: self.physical, self.physical :] = linearization.dcoefficients
            data.Fx[self.physical :, self.physical :] = np.eye(self.parameters)
            data.Lx[:] = self.dt_s * quadratic.gradient
            data.Lxx[:] = self.dt_s * quadratic.hessian

    return FlowAction


def _terminal_action_type(crocoddyl: Any) -> type[Any]:
    class TerminalAction(crocoddyl.ActionModelAbstract):
        def __init__(
            self,
            state: Any,
            cost: PolynomialCost,
            time_s: float,
            physical: int,
            parameters: int,
        ) -> None:
            super().__init__(state, 0, 0)
            self.cost_model = cost
            self.time_s = time_s
            self.physical = physical
            self.parameters = parameters

        def calc(self, data: Any, state: Any, control: Any = None) -> None:
            _empty_control(control)
            z = _vector(state, self.state.nx, "augmented state")
            data.xnext[:] = z
            data.cost = _cost_value(
                self.cost_model,
                z[: self.physical],
                z[self.physical :],
                self.time_s,
            )

        def calcDiff(self, data: Any, state: Any, control: Any = None) -> None:
            _empty_control(control)
            z = _vector(state, self.state.nx, "augmented state")
            quadratic = _quadratic(
                self.cost_model,
                z[: self.physical],
                z[self.physical :],
                self.time_s,
                self.state.nx,
            )
            data.xnext[:] = z
            data.cost = quadratic.value
            _clear_derivatives(data)
            data.Lx[:] = quadratic.gradient
            data.Lxx[:] = quadratic.hessian

    return TerminalAction


def _maximum(arrays: Sequence[Array]) -> float:
    return max(
        (float(np.max(np.abs(array), initial=0.0)) for array in arrays),
        default=0.0,
    )


@dataclass(frozen=True)
class PolynomialShootingProblem:
    """Assembled Crocoddyl problem with replay-based public utilities."""

    problem: Any
    time_grid: Array
    initial_state: Array
    coefficient_bounds: tuple[Array, Array] | None
    _stepper: NativeFullBodyStep
    _running_cost: PolynomialCost
    _terminal_cost: PolynomialCost
    _physical: int
    _parameters: int

    def _parameter_vector(self, parameters: Any, *, enforce_bounds: bool) -> Array:
        result = _vector(parameters, self._parameters, "parameters").copy()
        if enforce_bounds and self.coefficient_bounds is not None:
            lower, upper = self.coefficient_bounds
            if np.any(result < lower) or np.any(result > upper):
                raise ValueError("parameters must satisfy coefficient bounds")
        return result

    def warm_start(self, parameters: Any) -> PolynomialWarmStart:
        """Roll out the actual RK4 step from the fixed original physical state."""
        coefficients = self._parameter_vector(parameters, enforce_bounds=True)
        physical = self.initial_state.copy()
        xs = [
            np.r_[physical, np.zeros(self._parameters)],
            np.r_[physical, coefficients],
        ]
        us = [coefficients.copy()]
        for time_s, end_s in zip(self.time_grid[:-1], self.time_grid[1:], strict=True):
            physical = _step_value(
                self._stepper,
                physical,
                coefficients,
                float(time_s),
                float(end_s - time_s),
                self._physical,
            )
            xs.append(np.r_[physical, coefficients])
            us.append(np.zeros(0))
        return PolynomialWarmStart(xs, us)

    def _trajectories(
        self, xs: Sequence[Any], us: Sequence[Any]
    ) -> tuple[list[Array], list[Array]]:
        intervals = self.time_grid.size - 1
        if len(xs) != intervals + 2:
            raise ValueError("state trajectory has the wrong length")
        if len(us) != intervals + 1:
            raise ValueError("control trajectory has the wrong length")
        states = [
            _vector(value, self._physical + self._parameters, "state trajectory")
            for value in xs
        ]
        controls = [self._parameter_vector(us[0], enforce_bounds=False)]
        for value in us[1:]:
            _empty_control(value)
            controls.append(np.asarray(value, dtype=float))
        return states, controls

    def _independent_replay(self, coefficients: Array) -> tuple[list[Array], float]:
        physical = self.initial_state.copy()
        replay = [physical.copy()]
        cost = 0.0
        for time_s, end_s in zip(self.time_grid[:-1], self.time_grid[1:], strict=True):
            dt_s = float(end_s - time_s)
            cost += dt_s * _cost_value(
                self._running_cost, physical, coefficients, float(time_s)
            )
            physical = _step_value(
                self._stepper,
                physical,
                coefficients,
                float(time_s),
                dt_s,
                self._physical,
            )
            replay.append(physical.copy())
        cost += _cost_value(
            self._terminal_cost,
            physical,
            coefficients,
            float(self.time_grid[-1]),
        )
        return replay, _finite_cost(cost, "replay cost")

    def _node_cost(self, states: Sequence[Array]) -> float:
        cost = 0.0
        for state, time_s, end_s in zip(
            states[1:-1], self.time_grid[:-1], self.time_grid[1:], strict=True
        ):
            cost += float(end_s - time_s) * _cost_value(
                self._running_cost,
                state[: self._physical],
                state[self._physical :],
                float(time_s),
            )
        terminal = states[-1]
        cost += _cost_value(
            self._terminal_cost,
            terminal[: self._physical],
            terminal[self._physical :],
            float(self.time_grid[-1]),
        )
        return _finite_cost(cost, "shooting node cost")

    def diagnose(
        self, xs: Sequence[Any], us: Sequence[Any]
    ) -> PolynomialShootingDiagnostics:
        """Measure raw shooting defects and replay the extracted lift control."""
        states, controls = self._trajectories(xs, us)
        coefficients = controls[0]
        initial_physical = states[0][: self._physical] - self.initial_state
        lift_physical = states[1][: self._physical] - states[0][: self._physical]
        lift_coefficients = states[1][self._physical :] - coefficients
        flow_physical: list[Array] = []
        continuity: list[Array] = []
        for index, (time_s, end_s) in enumerate(
            zip(self.time_grid[:-1], self.time_grid[1:], strict=True)
        ):
            current, following = states[index + 1], states[index + 2]
            predicted = _step_value(
                self._stepper,
                current[: self._physical],
                current[self._physical :],
                float(time_s),
                float(end_s - time_s),
                self._physical,
            )
            flow_physical.append(following[: self._physical] - predicted)
            continuity.append(following[self._physical :] - current[self._physical :])
        mismatches = [state[self._physical :] - coefficients for state in states[1:]]
        replay, replay_cost = self._independent_replay(coefficients)
        node_cost = self._node_cost(states)
        replay_gaps = [
            state[: self._physical] - expected
            for state, expected in zip(states[1:], replay, strict=True)
        ]
        zeros: Array = np.zeros(self._parameters, dtype=float)
        lower_violation: Array = zeros
        upper_violation: Array = zeros.copy()
        if self.coefficient_bounds is not None:
            lower, upper = self.coefficient_bounds
            lower_violation = np.asarray(
                np.maximum(lower - coefficients, 0.0), dtype=float
            )
            upper_violation = np.asarray(
                np.maximum(coefficients - upper, 0.0), dtype=float
            )
        return _diagnostics(
            _DiagnosticParts(
                coefficients,
                initial_physical,
                lift_physical,
                lift_coefficients,
                flow_physical,
                continuity,
                mismatches,
                lower_violation,
                upper_violation,
                replay,
                replay_gaps,
                node_cost,
                replay_cost,
            )
        )


def _diagnostics(parts: _DiagnosticParts) -> PolynomialShootingDiagnostics:
    owned_flow = tuple(_readonly(value, "flow defect") for value in parts.flow_physical)
    owned_continuity = tuple(
        _readonly(value, "coefficient defect") for value in parts.continuity
    )
    owned_mismatches = tuple(
        _readonly(value, "coefficient mismatch") for value in parts.mismatches
    )
    owned_replay = tuple(_readonly(value, "replay state") for value in parts.replay)
    owned_gaps = tuple(_readonly(value, "replay gap") for value in parts.replay_gaps)
    owned_lift_x = _readonly(parts.lift_physical, "lift physical defect")
    owned_initial_x = _readonly(parts.initial_physical, "initial physical defect")
    owned_lift_p = _readonly(parts.lift_coefficients, "lift coefficient defect")
    owned_lower = _readonly(parts.lower_violation, "lower bound violation")
    owned_upper = _readonly(parts.upper_violation, "upper bound violation")
    cost_gap = _finite_cost(parts.node_cost - parts.replay_cost, "cost gap")
    return PolynomialShootingDiagnostics(
        _readonly(parts.coefficients, "coefficients"),
        owned_initial_x,
        owned_lift_x,
        owned_lift_p,
        owned_flow,
        owned_continuity,
        owned_mismatches,
        owned_lower,
        owned_upper,
        owned_replay,
        owned_gaps,
        _finite_cost(parts.node_cost, "shooting node cost"),
        float(parts.replay_cost),
        cost_gap,
        _maximum((owned_initial_x, owned_lift_x, *owned_flow)),
        _maximum((owned_lift_p, *owned_continuity)),
        _maximum(owned_mismatches),
        _maximum((owned_lower, owned_upper)),
        _maximum(owned_gaps),
    )


def _dimensions(stepper: NativeFullBodyStep) -> tuple[int, int, float]:
    control = stepper.control
    coordinates = control.n_coordinates
    parameters = control.n_parameters
    duration = control.duration_s
    if (
        isinstance(coordinates, bool)
        or not isinstance(coordinates, Integral)
        or coordinates <= 0
        or isinstance(parameters, bool)
        or not isinstance(parameters, Integral)
        or parameters <= 0
        or isinstance(duration, bool)
        or not isinstance(duration, Real)
        or not np.isfinite(duration)
        or duration <= 0.0
    ):
        raise ValueError("stepper control dimensions and duration must be positive")
    return 2 * int(coordinates), int(parameters), float(duration)


def _grid(value: Any, duration_s: float) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size < 2 or not np.isfinite(result).all():
        raise ValueError("time_grid must contain at least two finite times")
    if result[0] < 0.0:
        raise ValueError("time_grid must be nonnegative")
    if np.any(np.diff(result) <= 0.0):
        raise ValueError("time_grid must be strictly increasing")
    tolerance = 8.0 * np.finfo(float).eps * max(1.0, duration_s)
    if result[-1] > duration_s + tolerance:
        raise ValueError("time_grid exceeds the polynomial horizon")
    return _readonly(result, "time_grid")


def _bounds(value: Any, parameters: int) -> tuple[Array, Array] | None:
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("coefficient_bounds must be a lower/upper tuple")
    lower = _readonly(value[0], "coefficient lower bounds")
    upper = _readonly(value[1], "coefficient upper bounds")
    if lower.shape != (parameters,) or upper.shape != (parameters,):
        raise ValueError(f"coefficient bounds must have shape ({parameters},)")
    if np.any(lower > upper):
        raise ValueError("coefficient bounds must be ordered")
    return lower, upper


def build_polynomial_shooting_problem(
    stepper: NativeFullBodyStep,
    time_grid: Any,
    initial_state: Any,
    running_cost: PolynomialCost,
    terminal_cost: PolynomialCost,
    *,
    coefficient_bounds: tuple[Any, Any] | None = None,
) -> PolynomialShootingProblem:
    """Build a zero-time lift, autonomous RK4 flows, and x-only terminal."""
    physical, parameters, duration = _dimensions(stepper)
    grid = _grid(time_grid, duration)
    initial = _readonly(
        _vector(initial_state, physical, "initial_state"), "initial_state"
    )
    bounds = _bounds(coefficient_bounds, parameters)
    for cost in (running_cost, terminal_cost):
        if not callable(getattr(cost, "value", None)) or not callable(
            getattr(cost, "quadraticize", None)
        ):
            raise TypeError("costs must implement value and quadraticize")
    crocoddyl = _qualified_crocoddyl()
    state = crocoddyl.StateVector(physical + parameters)
    LiftAction = _lift_action_type(crocoddyl)
    FlowAction = _flow_action_type(crocoddyl)
    TerminalAction = _terminal_action_type(crocoddyl)
    lift = LiftAction(state, physical, parameters)
    if bounds is not None:
        lift.u_lb = bounds[0].copy()
        lift.u_ub = bounds[1].copy()
    flows = [
        FlowAction(
            state,
            stepper,
            running_cost,
            float(time_s),
            float(end_s - time_s),
            physical,
            parameters,
        )
        for time_s, end_s in zip(grid[:-1], grid[1:], strict=True)
    ]
    terminal = TerminalAction(
        state, terminal_cost, float(grid[-1]), physical, parameters
    )
    augmented = np.r_[initial, np.zeros(parameters)]
    problem = crocoddyl.ShootingProblem(augmented, [lift, *flows], terminal)
    return PolynomialShootingProblem(
        problem,
        grid,
        initial,
        bounds,
        stepper,
        running_cost,
        terminal_cost,
        physical,
        parameters,
    )
