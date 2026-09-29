"""Bounded nonlinear-controller kernels for manufactured qualification."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .nonlinear_controller_numerics import (
    BoxBounds,
    Dynamics,
    DynamicsFailure,
    FloatArray,
    QuadraticTrackingCost,
    SolverResult,
    central_dynamics_jacobians,
    checked_step,
    monotonic,
    rollout,
    trajectory_cost,
    validated_problem,
)


def solve_projected_ilqr(
    dynamics: Dynamics,
    initial_state: npt.ArrayLike,
    *,
    horizon: int,
    cost: QuadraticTrackingCost,
    bounds: BoxBounds,
    initial_controls: npt.ArrayLike,
) -> SolverResult:
    """Solve first-order iLQR with box projection inside every rollout."""
    state, controls, lower, upper = validated_problem(
        initial_state, horizon, cost, bounds, initial_controls
    )
    controls = np.clip(controls, lower, upper)
    try:
        states = rollout(dynamics, state, controls)
    except DynamicsFailure:
        return _failure("dynamics_failure")
    accepted = [trajectory_cost(states, controls, cost)]
    for iteration in range(1, 61):
        try:
            feedback, feedforward = _ilqr_backward(dynamics, states, controls, cost)
        except (ValueError, np.linalg.LinAlgError, DynamicsFailure):
            return _failure("solver_failure", tuple(accepted), iteration)
        candidate = _ilqr_descent(
            dynamics,
            states,
            controls,
            feedback,
            feedforward,
            cost,
            lower,
            upper,
        )
        if candidate is None:
            break
        states, controls, candidate_cost = candidate
        accepted.append(candidate_cost)
        if accepted[-2] - accepted[-1] <= 1.0e-10:
            break
    if len(accepted) < 2 or not monotonic(accepted) or accepted[-1] >= accepted[0]:
        return _failure("solver_failure", tuple(accepted), len(accepted) - 1)
    return SolverResult(
        True,
        "qualified_solution",
        states,
        controls,
        accepted[-1],
        tuple(accepted),
        len(accepted) - 1,
    )


def _ilqr_backward(
    dynamics: Dynamics,
    states: FloatArray,
    controls: FloatArray,
    cost: QuadraticTrackingCost,
) -> tuple[list[FloatArray], list[FloatArray]]:
    value_gradient = 2.0 * cost.terminal_weight @ (states[-1] - cost.reference_state)
    value_hessian = 2.0 * cost.terminal_weight
    feedback: list[FloatArray] = []
    feedforward: list[FloatArray] = []
    for index in range(controls.shape[0] - 1, -1, -1):
        state_map, control_map = _local_jacobians(
            dynamics, states[index], controls[index]
        )
        state_error = states[index] - cost.reference_state
        control_error = controls[index] - cost.reference_control
        q_state = 2.0 * cost.state_weight @ state_error + state_map.T @ value_gradient
        q_control = (
            2.0 * cost.control_weight @ control_error + control_map.T @ value_gradient
        )
        q_xx = 2.0 * cost.state_weight + state_map.T @ value_hessian @ state_map
        q_uu = 2.0 * cost.control_weight + control_map.T @ value_hessian @ control_map
        q_ux = control_map.T @ value_hessian @ state_map
        regularized = q_uu + np.eye(q_uu.shape[0]) * 1.0e-8
        gain = -np.linalg.solve(regularized, q_ux)
        shift = -np.linalg.solve(regularized, q_control)
        feedback.insert(0, gain)
        feedforward.insert(0, shift)
        value_gradient = _value_gradient(q_state, q_control, q_uu, q_ux, gain, shift)
        value_hessian = _value_hessian(q_xx, q_uu, q_ux, gain)
    return feedback, feedforward


def _local_jacobians(
    dynamics: Dynamics, state: FloatArray, control: FloatArray
) -> tuple[FloatArray, FloatArray]:
    return central_dynamics_jacobians(
        dynamics,
        state,
        control,
        state_steps=np.full(state.size, 1.0e-5),
        control_steps=np.full(control.size, 1.0e-5),
    )


def _value_gradient(
    q_state: FloatArray,
    q_control: FloatArray,
    q_uu: FloatArray,
    q_ux: FloatArray,
    gain: FloatArray,
    shift: FloatArray,
) -> FloatArray:
    return q_state + gain.T @ q_uu @ shift + gain.T @ q_control + q_ux.T @ shift


def _value_hessian(
    q_xx: FloatArray,
    q_uu: FloatArray,
    q_ux: FloatArray,
    gain: FloatArray,
) -> FloatArray:
    value = q_xx + gain.T @ q_uu @ gain + gain.T @ q_ux + q_ux.T @ gain
    return 0.5 * (value + value.T)


def _ilqr_descent(
    dynamics: Dynamics,
    states: FloatArray,
    controls: FloatArray,
    feedback: list[FloatArray],
    feedforward: list[FloatArray],
    cost: QuadraticTrackingCost,
    lower: FloatArray,
    upper: FloatArray,
) -> tuple[FloatArray, FloatArray, float] | None:
    baseline = trajectory_cost(states, controls, cost)
    for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        candidate = _projected_rollout(
            dynamics,
            states,
            controls,
            feedback,
            feedforward,
            lower,
            upper,
            alpha,
        )
        if candidate is None:
            continue
        candidate_states, candidate_controls = candidate
        candidate_cost = trajectory_cost(candidate_states, candidate_controls, cost)
        if candidate_cost < baseline - 1.0e-12:
            return candidate_states, candidate_controls, candidate_cost
    return None


def _projected_rollout(
    dynamics: Dynamics,
    states: FloatArray,
    controls: FloatArray,
    feedback: list[FloatArray],
    feedforward: list[FloatArray],
    lower: FloatArray,
    upper: FloatArray,
    alpha: float,
) -> tuple[FloatArray, FloatArray] | None:
    candidate_states = np.empty_like(states)
    candidate_controls = np.empty_like(controls)
    candidate_states[0] = states[0]
    try:
        for index in range(controls.shape[0]):
            correction = feedback[index] @ (candidate_states[index] - states[index])
            raw = controls[index] + alpha * feedforward[index] + correction
            candidate_controls[index] = np.clip(raw, lower, upper)
            candidate_states[index + 1] = checked_step(
                dynamics, candidate_states[index], candidate_controls[index]
            )
    except DynamicsFailure:
        return None
    return candidate_states, candidate_controls


def _failure(
    status: str, accepted: tuple[float, ...] = (), iterations: int = 0
) -> SolverResult:
    return SolverResult(False, status, None, None, None, accepted, iterations)
