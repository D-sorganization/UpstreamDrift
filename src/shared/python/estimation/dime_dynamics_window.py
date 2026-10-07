"""Coupled State-Control Full-Dynamics Window Factors (DIME-05, #11426).

Provides:
- DefectMode: exact vs soft transcription for dynamic transition defects.
- ModelDiscrepancyBounds: bounded process-noise / model-discrepancy slack validation.
- DimeDynamicsWindowFactor: transition defects, Jacobians, and exclusivity factor interface.
- DimeDynamicsWindowProblem: coupled window problem configuration and DbC validation.
- DimeDynamicsWindowResult: structured receipt exporting all cost and residual components.
- solve_dime_dynamics_window: coupled state-control window solver with fail-closed dynamics.

Transcription (#11554): multiple shooting.  The decision vector holds the controls
``u_0..u_{N-1}`` and the knot states ``x_1..x_N`` (``x = [q, v]``; ``x_0`` is the given
initial state).  Each interval contributes the transition defect
``d_k = x_{k+1} - Phi(x_k, u_k, dt)``, where ``Phi`` is the provider's own step, so
states and controls are coupled only through the defects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from src.shared.python.contracts import PreconditionError, ensure, require
from src.shared.python.estimation.dime_contracts import (
    ContactPolicy,
    DimeCompleteState,
    DimeFullStepRequest,
    DimeFullStepResult,
    DimeDynamicsProvider,
    EstimationIntervalFactor,
)


class DefectMode(str, Enum):
    """Transcription mode for integrated dynamic transition defects."""

    EXACT = "exact"
    SOFT = "soft"


@dataclass(frozen=True)
class ModelDiscrepancyBounds:
    """Bounded model discrepancy slack specifications."""

    max_slack_norm: float = 0.05
    slack_weight: float = 100.0

    def __post_init__(self) -> None:
        require(
            self.max_slack_norm > 0.0,
            "max_slack_norm must be positive",
            self.max_slack_norm,
        )
        require(
            self.slack_weight >= 0.0,
            "slack_weight must be non-negative",
            self.slack_weight,
        )

    def validate_slack(self, slack: np.ndarray) -> None:
        """Validate that model discrepancy slack stays within declared bounds."""
        raw_slack = np.asarray(slack, dtype=np.float64)
        norm = float(np.linalg.norm(raw_slack))
        if norm > self.max_slack_norm:
            raise PreconditionError(
                f"Model discrepancy slack norm {norm:.6f} exceeds bound {self.max_slack_norm:.6f}"
            )

    def evaluate_slack_cost(self, slack: np.ndarray) -> float:
        """Evaluate quadratic cost for discrepancy slack."""
        raw_slack = np.asarray(slack, dtype=np.float64)
        self.validate_slack(raw_slack)
        return 0.5 * self.slack_weight * float(np.sum(raw_slack**2))


CONTROL_EFFORT_RESIDUAL_SCALE = 1e-4
"""Scale of the control-effort regulariser residual ``scale * u`` (cost ``0.5 scale^2 |u|^2``)."""


class DimeDynamicsWindowFactor:
    """Coupled window factor computing transition defects and Jacobians."""

    def __init__(
        self,
        provider: DimeDynamicsProvider,
        initial_state: DimeCompleteState,
        horizon_steps: int,
        dt_s: float,
        defect_mode: DefectMode = DefectMode.EXACT,
        discrepancy_bounds: ModelDiscrepancyBounds | None = None,
        process_noise_cov: np.ndarray | None = None,
    ) -> None:
        require(provider is not None, "provider must not be None")
        require(initial_state is not None, "initial_state must not be None")
        require(horizon_steps >= 1, "horizon_steps must be at least 1", horizon_steps)
        require(dt_s > 0.0, "dt_s must be positive", dt_s)

        self.provider = provider
        self.initial_state = initial_state
        self.horizon_steps = horizon_steps
        self.dt_s = dt_s
        self.defect_mode = defect_mode
        self.discrepancy_bounds = discrepancy_bounds
        self.process_noise_cov = process_noise_cov

    def compute_single_step_defect(
        self,
        state_k: DimeCompleteState,
        state_k1: DimeCompleteState,
        control: np.ndarray,
        slack: np.ndarray | None = None,
    ) -> np.ndarray:
        """Evaluate integrated transition defect d = local_diff(state_k1, f(state_k, u)) - slack."""
        step_req = DimeFullStepRequest(
            state=state_k,
            controls=np.asarray(control, dtype=np.float64),
            dt=self.dt_s,
            model_hash=self.provider.model_hash,
        )
        step_res = self.provider.step(step_req)
        pred = step_res.next_state

        capability = self.provider.capability
        nv = capability.n_v
        # Tangent coordinate difference
        diff_q = np.asarray(state_k1.q[:nv], dtype=np.float64) - np.asarray(
            pred.q[:nv], dtype=np.float64
        )
        diff_v = np.asarray(state_k1.v[:nv], dtype=np.float64) - np.asarray(
            pred.v[:nv], dtype=np.float64
        )
        defect = np.concatenate([diff_q, diff_v])

        if slack is not None:
            raw_slack = np.asarray(slack, dtype=np.float64)
            if self.discrepancy_bounds is not None:
                self.discrepancy_bounds.validate_slack(raw_slack)
            defect = defect - raw_slack

        return defect

    def compute_defect_jacobians(
        self,
        state_k: DimeCompleteState,
        state_k1: DimeCompleteState,
        control: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Jacobians J_x0, J_x1, J_u of the defect via central finite differences."""
        capability = self.provider.capability
        nv = capability.n_v
        dim_x = 2 * nv
        nu = len(capability.control_channels)

        # J_x1 is identity
        J_x1 = np.eye(dim_x, dtype=np.float64)

        eps = 1e-6
        # J_u: derivative w.r.t control
        raw_u = np.asarray(control, dtype=np.float64)
        J_u = np.zeros((dim_x, nu), dtype=np.float64)
        for i in range(nu):
            u_plus = raw_u.copy()
            u_minus = raw_u.copy()
            u_plus[i] += eps
            u_minus[i] -= eps
            d_plus = self.compute_single_step_defect(state_k, state_k1, u_plus)
            d_minus = self.compute_single_step_defect(state_k, state_k1, u_minus)
            J_u[:, i] = (d_plus - d_minus) / (2.0 * eps)

        # J_x0: derivative w.r.t state_k
        J_x0 = np.zeros((dim_x, dim_x), dtype=np.float64)
        # Position perturbations
        for i in range(nv):
            q_plus = np.array(state_k.q, copy=True)
            q_minus = np.array(state_k.q, copy=True)
            q_plus[i] += eps
            q_minus[i] -= eps
            sk_plus = DimeCompleteState(
                t=state_k.t,
                q=q_plus,
                v=state_k.v,
                model_hash=state_k.model_hash,
                units=dict(state_k.units),
            )
            sk_minus = DimeCompleteState(
                t=state_k.t,
                q=q_minus,
                v=state_k.v,
                model_hash=state_k.model_hash,
                units=dict(state_k.units),
            )
            d_plus = self.compute_single_step_defect(sk_plus, state_k1, raw_u)
            d_minus = self.compute_single_step_defect(sk_minus, state_k1, raw_u)
            J_x0[:, i] = (d_plus - d_minus) / (2.0 * eps)

        # Velocity perturbations
        for i in range(nv):
            v_plus = np.array(state_k.v, copy=True)
            v_minus = np.array(state_k.v, copy=True)
            v_plus[i] += eps
            v_minus[i] -= eps
            sk_plus = DimeCompleteState(
                t=state_k.t,
                q=state_k.q,
                v=v_plus,
                model_hash=state_k.model_hash,
                units=dict(state_k.units),
            )
            sk_minus = DimeCompleteState(
                t=state_k.t,
                q=state_k.q,
                v=v_minus,
                model_hash=state_k.model_hash,
                units=dict(state_k.units),
            )
            d_plus = self.compute_single_step_defect(sk_plus, state_k1, raw_u)
            d_minus = self.compute_single_step_defect(sk_minus, state_k1, raw_u)
            J_x0[:, nv + i] = (d_plus - d_minus) / (2.0 * eps)

        return J_x0, J_x1, J_u

    def as_interval_factor(self) -> EstimationIntervalFactor:
        """Export as an explicit-input interval factor for runtime exclusivity validation."""
        return EstimationIntervalFactor(
            name="dime_dynamics_window",
            factor_type="explicit_input_likelihood",
            t_start=float(self.initial_state.t),
            t_end=float(self.initial_state.t + self.horizon_steps * self.dt_s),
            contributes_to_objective=True,
            metadata={
                "defect_mode": self.defect_mode.value,
                "horizon_steps": self.horizon_steps,
                "dt_s": self.dt_s,
            },
        )


@dataclass(frozen=True)
class DimeDynamicsWindowOptions:
    """Optional configuration and weights for dynamics window estimation."""

    defect_mode: DefectMode = DefectMode.EXACT
    enforce_root_constraints: bool = True
    actuator_bounds: tuple[float, float] | None = None
    control_rate_weight: float = 1.0
    observation_weight: float = 1.0
    transition_weight: float = 1000.0
    discrepancy_bounds: ModelDiscrepancyBounds | None = None
    max_iterations: int = 50


class DimeDynamicsWindowProblem:
    """Problem specification for coupled state-control window estimation."""

    def __init__(
        self,
        provider: DimeDynamicsProvider,
        initial_state: DimeCompleteState,
        horizon_steps: int,
        dt_s: float,
        target_positions: Sequence[np.ndarray] | None = None,
        options: DimeDynamicsWindowOptions | None = None,
        **kwargs: Any,
    ) -> None:
        require(provider is not None, "provider must not be None")
        require(initial_state is not None, "initial_state must not be None")
        require(horizon_steps >= 1, "horizon_steps must be at least 1", horizon_steps)
        require(dt_s > 0.0, "dt_s must be positive", dt_s)

        opts = options or DimeDynamicsWindowOptions()
        defect_mode = kwargs.get("defect_mode", opts.defect_mode)
        enforce_root_constraints = kwargs.get(
            "enforce_root_constraints", opts.enforce_root_constraints
        )
        actuator_bounds = kwargs.get("actuator_bounds", opts.actuator_bounds)
        control_rate_weight = kwargs.get(
            "control_rate_weight", opts.control_rate_weight
        )
        observation_weight = kwargs.get("observation_weight", opts.observation_weight)
        transition_weight = kwargs.get("transition_weight", opts.transition_weight)
        discrepancy_bounds = kwargs.get("discrepancy_bounds", opts.discrepancy_bounds)
        max_iterations = kwargs.get("max_iterations", opts.max_iterations)

        require(
            control_rate_weight >= 0.0,
            "control_rate_weight must be non-negative",
            control_rate_weight,
        )
        require(
            transition_weight > 0.0,
            "transition_weight must be positive",
            transition_weight,
        )
        capability = provider.capability
        require(
            capability.n_q == capability.n_v,
            "multiple-shooting window requires a vector-space state (n_q == n_v); "
            "manifold knot states need a retraction",
            (capability.n_q, capability.n_v),
        )

        self.provider = provider
        self.initial_state = initial_state
        self.horizon_steps = horizon_steps
        self.dt_s = dt_s
        self.target_positions = (
            tuple(np.asarray(p, dtype=np.float64) for p in target_positions)
            if target_positions is not None
            else None
        )
        self.defect_mode = defect_mode
        self.enforce_root_constraints = enforce_root_constraints
        self.actuator_bounds = actuator_bounds
        self.control_rate_weight = control_rate_weight
        self.observation_weight = observation_weight
        self.transition_weight = transition_weight
        self.discrepancy_bounds = discrepancy_bounds
        self.max_iterations = max_iterations

    def validate_controls(self, controls: np.ndarray) -> None:
        """Validate control array shape, underactuated root constraints, and actuator bounds."""
        raw_u = np.asarray(controls, dtype=np.float64)
        require(raw_u.ndim == 2, "controls must be a 2D array of shape (N, n_u)")
        require(
            raw_u.shape[0] == self.horizon_steps,
            f"controls row count {raw_u.shape[0]} must match horizon_steps {self.horizon_steps}",
        )

        capability = self.provider.capability
        nu_declared = len(capability.control_channels)
        nv = capability.n_v

        # Check for underactuated root constraint shortcuts
        if self.enforce_root_constraints:
            if raw_u.shape[1] > nu_declared:
                # User passed controls for all DOFs including passive root
                # Check if passive DOFs have non-zero torque
                passive_dofs = set(range(nv))
                for ch in capability.control_channels:
                    passive_dofs.difference_update(ch.selection_map)
                for dof in passive_dofs:
                    if dof < raw_u.shape[1] and np.any(np.abs(raw_u[:, dof]) > 1e-9):
                        raise PreconditionError(
                            f"Root joint {dof} is passive/underactuated; cannot apply arbitrary control shortcut."
                        )

        require(
            raw_u.shape[1] == nu_declared,
            f"controls channel count {raw_u.shape[1]} must match declared channels {nu_declared}",
        )

        if self.actuator_bounds is not None:
            low, high = self.actuator_bounds
            if np.any(raw_u < low - 1e-9) or np.any(raw_u > high + 1e-9):
                raise PreconditionError(
                    f"Controls violate actuator bounds [{low}, {high}]"
                )

    def evaluate_control_variation_cost(self, controls: np.ndarray) -> float:
        """Evaluate control variation regularizer cost: 0.5 * weight * sum(||(u_{k+1}-u_k)/dt||^2) * dt."""
        raw_u = np.asarray(controls, dtype=np.float64)
        if len(raw_u) <= 1:
            return 0.0
        delta_u = np.diff(raw_u, axis=0) / self.dt_s
        rate_sq = np.sum(delta_u**2)
        return 0.5 * self.control_rate_weight * float(rate_sq) * self.dt_s


@dataclass(frozen=True)
class DimeDynamicsWindowResult:
    """Result receipt of a coupled state-control window solve."""

    success: bool
    states: tuple[DimeCompleteState, ...]
    controls: np.ndarray
    transition_defects: np.ndarray
    actuator_bound_residuals: np.ndarray
    control_variation_residuals: np.ndarray
    root_constraint_residuals: np.ndarray
    cost_breakdown: Mapping[str, float]
    status: str
    qualification_status: str
    n_iterations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "states": [s.to_dict() for s in self.states],
            "controls": self.controls.tolist(),
            "transition_defects": self.transition_defects.tolist(),
            "actuator_bound_residuals": self.actuator_bound_residuals.tolist(),
            "control_variation_residuals": self.control_variation_residuals.tolist(),
            "root_constraint_residuals": self.root_constraint_residuals.tolist(),
            "cost_breakdown": dict(self.cost_breakdown),
            "status": self.status,
            "qualification_status": self.qualification_status,
            "n_iterations": self.n_iterations,
        }


def _window_dims(problem: DimeDynamicsWindowProblem) -> tuple[int, int, int]:
    """Return ``(n_steps, n_u, n_x)`` with ``n_x = 2 n_v`` the knot-state size."""
    capability = problem.provider.capability
    return problem.horizon_steps, len(capability.control_channels), 2 * capability.n_v


def _failed_result(
    problem: DimeDynamicsWindowProblem, status: str
) -> DimeDynamicsWindowResult:
    """Fail-closed receipt: no trajectory was produced, so the cost is infinite."""
    n_steps, nu, nx = _window_dims(problem)
    return DimeDynamicsWindowResult(
        success=False,
        states=(problem.initial_state,),
        controls=np.zeros((n_steps, nu)),
        transition_defects=np.zeros((n_steps, nx)),
        actuator_bound_residuals=np.zeros(1),
        control_variation_residuals=np.zeros(1),
        root_constraint_residuals=np.zeros(1),
        cost_breakdown={"total_cost": float("inf")},
        status=status,
        qualification_status=problem.provider.capability.status,
        n_iterations=0,
    )


def _check_state_divergence(
    problem: DimeDynamicsWindowProblem,
) -> DimeDynamicsWindowResult | None:
    """Fail closed if initial conditions are divergent or unphysical."""
    q0 = np.asarray(problem.initial_state.q, dtype=np.float64)
    v0 = np.asarray(problem.initial_state.v, dtype=np.float64)
    if (
        not np.all(np.isfinite(q0))
        or not np.all(np.isfinite(v0))
        or np.any(np.abs(q0) > 1e4)
        or np.any(np.abs(v0) > 1e4)
    ):
        return _failed_result(problem, "failed_dynamics_divergence")
    return None


def _rollout_nominal_trajectory(
    problem: DimeDynamicsWindowProblem,
    controls: np.ndarray | None = None,
) -> tuple[bool, list[DimeCompleteState]]:
    """Roll the provider forward under ``controls`` (zero when omitted).

    Returns ``(ok, states)`` with ``states = [x_0, ..., x_N]`` on success and
    ``(False, [x_0])`` if any provider step fails.
    """
    n_steps, nu, _ = _window_dims(problem)
    u_arr = np.zeros((n_steps, nu)) if controls is None else controls
    curr_state = problem.initial_state
    states = [curr_state]
    try:
        for k in range(n_steps):
            req = DimeFullStepRequest(
                state=curr_state,
                controls=u_arr[k],
                dt=problem.dt_s,
                model_hash=problem.provider.model_hash,
            )
            curr_state = problem.provider.step(req).next_state
            states.append(curr_state)
        return True, states
    except Exception:
        return False, [problem.initial_state]


class DimeResidualEvaluationError(RuntimeError):
    """Raised when the dynamics provider fails while evaluating window residuals.

    The residual vector has a fixed size on every path, so a failed step cannot be
    represented by a placeholder residual; callers must handle this error.
    """


def pack_window_decision(controls: np.ndarray, knot_states: np.ndarray) -> np.ndarray:
    """Stack controls ``(N, n_u)`` and knot states ``(N, n_x)`` into ``z``.

    Layout: ``z = [u_0, ..., u_{N-1}, x_1, ..., x_N]`` with ``x_k = [q_k, v_k]``.
    """
    u_arr = np.asarray(controls, dtype=np.float64)
    x_arr = np.asarray(knot_states, dtype=np.float64)
    require(u_arr.ndim == 2, "controls must have shape (N, n_u)", u_arr.shape)
    require(x_arr.ndim == 2, "knot_states must have shape (N, n_x)", x_arr.shape)
    require(
        u_arr.shape[0] == x_arr.shape[0],
        "controls and knot_states must have one row per interval",
        (u_arr.shape, x_arr.shape),
    )
    return np.concatenate([u_arr.ravel(), x_arr.ravel()])


def unpack_window_decision(
    problem: DimeDynamicsWindowProblem, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split a decision vector into controls ``(N, n_u)`` and knots ``(N, n_x)``."""
    n_steps, nu, nx = _window_dims(problem)
    raw = np.asarray(z, dtype=np.float64)
    require(
        raw.shape == (n_steps * (nu + nx),),
        f"decision vector must have length N*(n_u+n_x) = {n_steps * (nu + nx)}",
        raw.shape,
    )
    split = n_steps * nu
    return raw[:split].reshape(n_steps, nu), raw[split:].reshape(n_steps, nx)


def _knot_states(
    problem: DimeDynamicsWindowProblem, knot_states: np.ndarray
) -> list[DimeCompleteState]:
    """Return ``[x_0, x_1, ..., x_N]`` as complete states (``x_0`` is fixed)."""
    init = problem.initial_state
    nv = problem.provider.capability.n_v
    states = [init]
    for k, x in enumerate(knot_states, start=1):
        states.append(
            DimeCompleteState(
                t=float(init.t + k * problem.dt_s),
                q=x[:nv],
                v=x[nv:],
                model_hash=init.model_hash,
                units=dict(init.units),
                frame=init.frame,
            )
        )
    return states


def _states_to_knots(states: Sequence[DimeCompleteState]) -> np.ndarray:
    """Knot array ``(N, n_x)`` from ``[x_0, ..., x_N]`` (``x_0`` dropped)."""
    return np.array([np.concatenate([s.q, s.v]) for s in states[1:]], dtype=np.float64)


def compute_window_transition_defects(
    problem: DimeDynamicsWindowProblem,
    knot_states: np.ndarray,
    controls: np.ndarray,
) -> np.ndarray:
    """Multiple-shooting defects ``d_k = x_{k+1} - Phi(x_k, u_k, dt)``, k = 0..N-1.

    ``Phi`` is the provider's step and ``x_0`` the problem's initial state.

    Preconditions: ``knot_states`` has shape ``(N, 2 n_v)``, ``controls`` has shape
    ``(N, n_u)``, both finite.
    Postcondition: returns a finite ``(N, 2 n_v)`` array.

    Raises:
        DimeResidualEvaluationError: if a provider step fails.
    """
    n_steps, nu, nx = _window_dims(problem)
    x_arr = np.asarray(knot_states, dtype=np.float64)
    u_arr = np.asarray(controls, dtype=np.float64)
    require(x_arr.shape == (n_steps, nx), "knot_states must be (N, 2 n_v)", x_arr.shape)
    require(u_arr.shape == (n_steps, nu), "controls must be (N, n_u)", u_arr.shape)
    require(bool(np.all(np.isfinite(x_arr))), "knot_states must be finite")
    require(bool(np.all(np.isfinite(u_arr))), "controls must be finite")

    factor = DimeDynamicsWindowFactor(
        provider=problem.provider,
        initial_state=problem.initial_state,
        horizon_steps=n_steps,
        dt_s=problem.dt_s,
        defect_mode=problem.defect_mode,
    )
    states = _knot_states(problem, x_arr)
    defects = np.empty((n_steps, nx), dtype=np.float64)
    for k in range(n_steps):
        try:
            defects[k] = factor.compute_single_step_defect(
                states[k], states[k + 1], u_arr[k]
            )
        except Exception as exc:
            raise DimeResidualEvaluationError(
                f"dynamics step {k} failed during residual evaluation: {exc}"
            ) from exc
    ensure(bool(np.all(np.isfinite(defects))), "transition defects must be finite")
    return defects


def _discrepancy_slack_bounds(
    problem: DimeDynamicsWindowProblem,
) -> ModelDiscrepancyBounds | None:
    """Bounds when SOFT defects are model-discrepancy slack, else ``None``."""
    if problem.defect_mode == DefectMode.SOFT:
        return problem.discrepancy_bounds
    return None


def _defect_weight(problem: DimeDynamicsWindowProblem) -> float:
    """Quadratic weight on the transition defects in the objective."""
    slack_bounds = _discrepancy_slack_bounds(problem)
    if slack_bounds is not None:
        return float(slack_bounds.slack_weight)
    return float(problem.transition_weight)


def _observation_targets(
    problem: DimeDynamicsWindowProblem,
) -> list[tuple[int, np.ndarray]]:
    """``(k, y_k)`` pairs for knots ``k <= N``, truncated to ``n_q`` entries."""
    if problem.target_positions is None:
        return []
    n_q = problem.provider.capability.n_q
    return [
        (k, targ[:n_q])
        for k, targ in enumerate(problem.target_positions)
        if k <= problem.horizon_steps
    ]


def expected_residual_size(problem: DimeDynamicsWindowProblem) -> int:
    """Return the fixed length of the window residual vector.

    Postcondition: every successful residual evaluation has exactly this length,
    independent of the decision-vector values.
    """
    n_steps, nu, nx = _window_dims(problem)
    size = sum(len(targ) for _, targ in _observation_targets(problem))
    size += n_steps * nx  # transition defects
    if problem.control_rate_weight > 0.0 and n_steps > 1:
        size += (n_steps - 1) * nu
    size += n_steps * nu  # control effort
    if problem.actuator_bounds is not None:
        size += n_steps * nu  # one bound-violation entry per control
    return size


def _observation_residuals(
    problem: DimeDynamicsWindowProblem, knot_states: np.ndarray
) -> list[np.ndarray]:
    """Unweighted position residuals ``q_k - y_k`` for every target."""
    n_q = problem.provider.capability.n_q
    q_all = np.vstack([problem.initial_state.q[None, :n_q], knot_states[:, :n_q]])
    return [q_all[k, : len(targ)] - targ for k, targ in _observation_targets(problem)]


def _build_residuals_evaluator(
    problem: DimeDynamicsWindowProblem,
) -> Callable[[np.ndarray], np.ndarray]:
    """Least-squares residual over ``z = [u, x_1..x_N]`` (see ``pack_window_decision``).

    Block order: observations, transition defects, control rate (when weighted),
    control effort, actuator-bound violations (when bounds are set).
    """
    expected_size = expected_residual_size(problem)
    dt = problem.dt_s
    w_obs = np.sqrt(problem.observation_weight)
    w_def = np.sqrt(_defect_weight(problem))
    w_rate = np.sqrt(problem.control_rate_weight * dt)
    use_rate = problem.control_rate_weight > 0.0 and problem.horizon_steps > 1

    def residuals_fn(z: np.ndarray) -> np.ndarray:
        u_arr, x_arr = unpack_window_decision(problem, z)
        defects = compute_window_transition_defects(problem, x_arr, u_arr)
        blocks = [w_obs * r for r in _observation_residuals(problem, x_arr)]
        blocks.append(w_def * defects.ravel())
        if use_rate:
            blocks.append(w_rate * (np.diff(u_arr, axis=0) / dt).ravel())
        blocks.append(CONTROL_EFFORT_RESIDUAL_SCALE * u_arr.ravel())
        if problem.actuator_bounds is not None:
            low, high = problem.actuator_bounds
            violation = np.maximum(0.0, low - u_arr) + np.maximum(0.0, u_arr - high)
            blocks.append(100.0 * violation.ravel())

        residual = np.concatenate(blocks)
        ensure(
            residual.shape == (expected_size,),
            "residual length must equal expected_residual_size",
            residual.shape,
        )
        return residual

    return residuals_fn


def _compute_window_cost_breakdown(
    problem: DimeDynamicsWindowProblem,
    knot_states: np.ndarray,
    u_opt: np.ndarray,
    defects: np.ndarray,
) -> dict[str, float]:
    """Cost components ``0.5 w |r|^2`` evaluated on the returned trajectory.

    Postcondition: ``total_cost`` equals the sum of the other components.
    """
    obs_cost = (
        0.5
        * problem.observation_weight
        * sum(float(np.sum(r**2)) for r in _observation_residuals(problem, knot_states))
    )
    defect_cost = 0.5 * _defect_weight(problem) * float(np.sum(defects**2))
    slack = _discrepancy_slack_bounds(problem) is not None
    transition_cost = 0.0 if slack else defect_cost
    discrepancy_cost = defect_cost if slack else 0.0
    rate_cost = problem.evaluate_control_variation_cost(u_opt)
    effort_cost = 0.5 * CONTROL_EFFORT_RESIDUAL_SCALE**2 * float(np.sum(u_opt**2))
    return {
        "observation_cost": obs_cost,
        "transition_cost": transition_cost,
        "control_effort_cost": effort_cost,
        "control_rate_cost": rate_cost,
        "discrepancy_cost": discrepancy_cost,
        "total_cost": obs_cost
        + transition_cost
        + effort_cost
        + rate_cost
        + discrepancy_cost,
    }


def _assemble_window_result(
    problem: DimeDynamicsWindowProblem,
    opt_res: Any,
) -> DimeDynamicsWindowResult:
    """Bundle the optimum into a receipt with defects and costs evaluated on it.

    SOFT returns the optimised knot states, whose defects are the residual model
    mismatch.  EXACT returns the forward rollout of the optimised controls (a
    feasibility projection), so its defects are zero by construction and are
    still evaluated rather than assumed.
    """
    n_steps, nu, _ = _window_dims(problem)
    u_opt, x_opt = unpack_window_decision(problem, opt_res.x)
    if problem.defect_mode == DefectMode.EXACT:
        ok, states = _rollout_nominal_trajectory(problem, u_opt)
        if not ok:
            return _failed_result(problem, "failed_dynamics_step")
        x_opt = _states_to_knots(states)
    else:
        states = _knot_states(problem, x_opt)
    defects = compute_window_transition_defects(problem, x_opt, u_opt)
    cost_breakdown = _compute_window_cost_breakdown(problem, x_opt, u_opt, defects)
    ctrl_var_res = (
        np.diff(u_opt, axis=0) / problem.dt_s if n_steps > 1 else np.zeros((1, nu))
    )

    if problem.actuator_bounds is not None:
        low, high = problem.actuator_bounds
        bound_res = np.maximum(0.0, low - u_opt) + np.maximum(0.0, u_opt - high)
    else:
        bound_res = np.zeros_like(u_opt)

    success = bool(opt_res.success)
    status = "converged" if success else "max_iterations_reached"
    slack_bounds = _discrepancy_slack_bounds(problem)
    if slack_bounds is not None:
        max_step_slack = float(np.max(np.linalg.norm(defects, axis=1)))
        if max_step_slack > slack_bounds.max_slack_norm:
            success, status = False, "failed_discrepancy_bound"

    return DimeDynamicsWindowResult(
        success=success,
        states=tuple(states),
        controls=u_opt,
        transition_defects=defects,
        actuator_bound_residuals=bound_res,
        control_variation_residuals=ctrl_var_res,
        root_constraint_residuals=np.zeros(n_steps, dtype=np.float64),
        cost_breakdown=cost_breakdown,
        status=status,
        qualification_status=problem.provider.capability.status,
        n_iterations=int(opt_res.nfev),
    )


def _initial_decision(
    problem: DimeDynamicsWindowProblem,
    initial_controls: np.ndarray | None,
    initial_knot_states: np.ndarray | None,
) -> np.ndarray | None:
    """Validated initial ``z``; knots default to the rollout of the controls.

    Returns ``None`` if that rollout fails (the dynamics diverge).
    """
    n_steps, nu, nx = _window_dims(problem)
    u0 = np.zeros((n_steps, nu)) if initial_controls is None else initial_controls
    u0 = np.asarray(u0, dtype=np.float64)
    require(u0.shape == (n_steps, nu), "initial_controls must be (N, n_u)", u0.shape)
    require(bool(np.all(np.isfinite(u0))), "initial_controls must be finite")
    if initial_knot_states is None:
        ok, states = _rollout_nominal_trajectory(problem, u0)
        if not ok:
            return None
        x0 = _states_to_knots(states)
    else:
        x0 = np.asarray(initial_knot_states, dtype=np.float64)
        require(
            x0.shape == (n_steps, nx),
            "initial_knot_states must be (N, 2 n_v)",
            x0.shape,
        )
        require(bool(np.all(np.isfinite(x0))), "initial_knot_states must be finite")
    return pack_window_decision(u0, x0)


def solve_dime_dynamics_window(
    problem: DimeDynamicsWindowProblem,
    *,
    initial_controls: np.ndarray | None = None,
    initial_knot_states: np.ndarray | None = None,
) -> DimeDynamicsWindowResult:
    """Solve the coupled state-control window by multiple shooting.

    Decision variables are the controls and the knot states ``x_1..x_N``; the
    objective is ``0.5 |r(z)|^2`` with ``r`` from ``_build_residuals_evaluator``.

    Args:
        problem: Window problem.
        initial_controls: Optional ``(N, n_u)`` warm start (default zeros).
        initial_knot_states: Optional ``(N, 2 n_v)`` warm start; it need not be
            dynamically consistent (default: rollout of ``initial_controls``).

    Postcondition: ``result.transition_defects`` are evaluated on the returned
    ``(states, controls)``; failures return ``success=False``.
    """
    if problem.target_positions is not None:
        for targ in problem.target_positions:
            require(bool(np.all(np.isfinite(targ))), "target_positions must be finite")
    divergence_res = _check_state_divergence(problem)
    if divergence_res is not None:
        return divergence_res

    nominal_ok, _ = _rollout_nominal_trajectory(problem)
    if not nominal_ok:
        return _failed_result(problem, "failed_dynamics_divergence")
    z_init = _initial_decision(problem, initial_controls, initial_knot_states)
    if z_init is None:
        return _failed_result(problem, "failed_dynamics_divergence")

    residuals_fn = _build_residuals_evaluator(problem)
    n_steps, nu, nx = _window_dims(problem)
    n_u_vars = n_steps * nu
    n_vars = z_init.size

    bounds: tuple[Any, Any] = (-np.inf, np.inf)
    if problem.actuator_bounds is not None:
        low, high = problem.actuator_bounds
        lower = np.full(n_vars, -np.inf)
        upper = np.full(n_vars, np.inf)
        lower[:n_u_vars] = low
        upper[:n_u_vars] = high
        bounds = (lower, upper)

    try:
        opt_res = least_squares(
            residuals_fn,
            z_init,
            bounds=bounds,
            method="trf",
            max_nfev=problem.max_iterations * (n_vars + 1),
            ftol=1e-7,
            xtol=1e-7,
        )
    except DimeResidualEvaluationError:
        return _failed_result(problem, "failed_dynamics_step")
    except Exception as exc:
        return _failed_result(problem, f"solver_exception_{exc}")

    return _assemble_window_result(problem, opt_res)
