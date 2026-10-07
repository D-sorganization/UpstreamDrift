"""Coupled State-Control Full-Dynamics Window Factors (DIME-05, #11426).

Provides:
- DefectMode: exact vs soft transcription for dynamic transition defects.
- ModelDiscrepancyBounds: bounded process-noise / model-discrepancy slack validation.
- DimeDynamicsWindowFactor: transition defects, Jacobians, and exclusivity factor interface.
- DimeDynamicsWindowProblem: coupled window problem configuration and DbC validation.
- DimeDynamicsWindowResult: structured receipt exporting all cost and residual components.
- solve_dime_dynamics_window: coupled state-control window solver with fail-closed dynamics.
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
    DynamicsProvider,
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


class DimeDynamicsWindowFactor:
    """Coupled window factor computing transition defects and Jacobians."""

    def __init__(
        self,
        provider: DynamicsProvider,
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
        provider: DynamicsProvider,
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


def _check_state_divergence(
    problem: DimeDynamicsWindowProblem,
) -> DimeDynamicsWindowResult | None:
    """Fail closed if initial conditions are divergent or unphysical."""
    q0 = np.asarray(problem.initial_state.q, dtype=np.float64)
    v0 = np.asarray(problem.initial_state.v, dtype=np.float64)
    capability = problem.provider.capability
    if (
        not np.all(np.isfinite(q0))
        or not np.all(np.isfinite(v0))
        or np.any(np.abs(q0) > 1e4)
        or np.any(np.abs(v0) > 1e4)
    ):
        return DimeDynamicsWindowResult(
            success=False,
            states=(problem.initial_state,),
            controls=np.zeros(
                (
                    problem.horizon_steps,
                    len(capability.control_channels),
                )
            ),
            transition_defects=np.zeros((problem.horizon_steps, 2 * capability.n_v)),
            actuator_bound_residuals=np.zeros(1),
            control_variation_residuals=np.zeros(1),
            root_constraint_residuals=np.zeros(1),
            cost_breakdown={"total_cost": float("inf")},
            status="failed_dynamics_divergence",
            qualification_status=capability.status,
            n_iterations=0,
        )
    return None


def _rollout_nominal_trajectory(
    problem: DimeDynamicsWindowProblem,
) -> tuple[bool, list[DimeCompleteState]]:
    """Perform zero-control initial rollout to verify dynamics provider stability."""
    curr_state = problem.initial_state
    states = [curr_state]
    capability = problem.provider.capability
    nu = len(capability.control_channels)
    zeros = np.zeros(nu, dtype=np.float64)
    try:
        for _ in range(problem.horizon_steps):
            req = DimeFullStepRequest(
                state=curr_state,
                controls=zeros,
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


def expected_residual_size(problem: DimeDynamicsWindowProblem) -> int:
    """Return the fixed length of the window residual vector.

    Postcondition: every successful residual evaluation has exactly this length,
    independent of the control values.
    """
    n_steps = problem.horizon_steps
    nu = len(problem.provider.capability.control_channels)
    n_q = problem.provider.capability.n_q
    size = n_steps * nu  # control effort
    if problem.target_positions is not None:
        size += sum(
            min(len(targ), n_q)
            for k, targ in enumerate(problem.target_positions)
            if k <= n_steps
        )
    if problem.control_rate_weight > 0.0 and n_steps > 1:
        size += (n_steps - 1) * nu
    if problem.actuator_bounds is not None:
        size += n_steps * nu  # one bound-violation entry per control
    return size


def _build_residuals_evaluator(
    problem: DimeDynamicsWindowProblem,
) -> Callable[[np.ndarray], np.ndarray]:
    """Construct least-squares residual function for state-control window estimation."""
    provider = problem.provider
    initial_state = problem.initial_state
    n_steps = problem.horizon_steps
    dt = problem.dt_s
    nu = len(provider.capability.control_channels)
    expected_size = expected_residual_size(problem)

    def residuals_fn(u_vec: np.ndarray) -> np.ndarray:
        res_list: list[float] = []
        u_arr = u_vec.reshape((n_steps, nu))
        st = initial_state
        traj = [st]
        for k in range(n_steps):
            try:
                s_req = DimeFullStepRequest(
                    state=st,
                    controls=u_arr[k],
                    dt=dt,
                    model_hash=provider.model_hash,
                )
                st = provider.step(s_req).next_state
                traj.append(st)
            except Exception as exc:
                raise DimeResidualEvaluationError(
                    f"dynamics step {k} failed during residual evaluation: {exc}"
                ) from exc

        if problem.target_positions is not None:
            w_obs = np.sqrt(problem.observation_weight)
            for k, targ in enumerate(problem.target_positions):
                if k < len(traj):
                    pos_diff = (traj[k].q[: len(targ)] - targ) * w_obs
                    res_list.extend(pos_diff.tolist())

        if problem.control_rate_weight > 0.0 and n_steps > 1:
            w_rate = np.sqrt(problem.control_rate_weight * dt)
            for k in range(n_steps - 1):
                diff_u = (u_arr[k + 1] - u_arr[k]) / dt
                res_list.extend((diff_u * w_rate).tolist())

        res_list.extend((u_vec * 1e-4).tolist())

        if problem.actuator_bounds is not None:
            low, high = problem.actuator_bounds
            for u_val in u_vec:
                # Always one entry per control so the residual length is fixed.
                res_list.append(
                    100.0 * (max(0.0, low - u_val) + max(0.0, u_val - high))
                )

        residual = np.asarray(res_list, dtype=np.float64)
        ensure(
            residual.shape == (expected_size,),
            "residual length must equal expected_residual_size",
            residual.shape,
        )
        return residual

    return residuals_fn


def _compute_window_cost_breakdown(
    problem: DimeDynamicsWindowProblem,
    final_states: Sequence[DimeCompleteState],
    u_opt: np.ndarray,
) -> dict[str, float]:
    """Compute observation, rate, and effort components of window cost."""
    obs_cost = 0.0
    if problem.target_positions is not None:
        for k, targ in enumerate(problem.target_positions):
            if k < len(final_states):
                obs_cost += (
                    0.5
                    * problem.observation_weight
                    * float(np.sum((final_states[k].q[: len(targ)] - targ) ** 2))
                )

    rate_cost = problem.evaluate_control_variation_cost(u_opt)
    effort_cost = 0.5 * 1e-4 * float(np.sum(u_opt**2))
    return {
        "observation_cost": obs_cost,
        "transition_cost": 0.0,
        "control_effort_cost": effort_cost,
        "control_rate_cost": rate_cost,
        "discrepancy_cost": 0.0,
        "total_cost": obs_cost + rate_cost + effort_cost,
    }


def _assemble_window_result(
    problem: DimeDynamicsWindowProblem,
    opt_res: Any,
    u_opt: np.ndarray,
) -> DimeDynamicsWindowResult:
    """Roll out optimal trajectory and bundle result dataclass."""
    n_steps = problem.horizon_steps
    dt = problem.dt_s
    capability = problem.provider.capability
    nu = len(capability.control_channels)
    nv = capability.n_v
    curr_state = problem.initial_state
    final_states = [curr_state]
    transition_defects = []
    for k in range(n_steps):
        step_req = DimeFullStepRequest(
            state=curr_state,
            controls=u_opt[k],
            dt=dt,
            model_hash=problem.provider.model_hash,
        )
        curr_state = problem.provider.step(step_req).next_state
        final_states.append(curr_state)
        transition_defects.append(np.zeros(2 * nv, dtype=np.float64))

    cost_breakdown = _compute_window_cost_breakdown(problem, final_states, u_opt)
    ctrl_var_res = np.diff(u_opt, axis=0) / dt if n_steps > 1 else np.zeros((1, nu))

    if problem.actuator_bounds is not None:
        low, high = problem.actuator_bounds
        bound_res = np.maximum(0.0, low - u_opt) + np.maximum(0.0, u_opt - high)
    else:
        bound_res = np.zeros_like(u_opt)

    return DimeDynamicsWindowResult(
        success=bool(opt_res.success),
        states=tuple(final_states),
        controls=u_opt,
        transition_defects=np.asarray(transition_defects, dtype=np.float64),
        actuator_bound_residuals=bound_res,
        control_variation_residuals=ctrl_var_res,
        root_constraint_residuals=np.zeros(n_steps, dtype=np.float64),
        cost_breakdown=cost_breakdown,
        status="converged" if opt_res.success else "max_iterations_reached",
        qualification_status=capability.status,
        n_iterations=int(opt_res.nfev),
    )


def solve_dime_dynamics_window(
    problem: DimeDynamicsWindowProblem,
) -> DimeDynamicsWindowResult:
    """Solve coupled state-control window estimation problem with fail-closed dynamics."""
    divergence_res = _check_state_divergence(problem)
    if divergence_res is not None:
        return divergence_res

    nominal_ok, _ = _rollout_nominal_trajectory(problem)
    capability = problem.provider.capability
    if not nominal_ok:
        nu = len(capability.control_channels)
        nv = capability.n_v
        return DimeDynamicsWindowResult(
            success=False,
            states=(problem.initial_state,),
            controls=np.zeros((problem.horizon_steps, nu)),
            transition_defects=np.zeros((problem.horizon_steps, 2 * nv)),
            actuator_bound_residuals=np.zeros(1),
            control_variation_residuals=np.zeros(1),
            root_constraint_residuals=np.zeros(1),
            cost_breakdown={"total_cost": float("inf")},
            status="failed_dynamics_divergence",
            qualification_status=capability.status,
            n_iterations=0,
        )

    residuals_fn = _build_residuals_evaluator(problem)
    n_steps = problem.horizon_steps
    nu = len(capability.control_channels)
    u_init = np.zeros(n_steps * nu, dtype=np.float64)

    bounds: tuple[Any, Any] = (-np.inf, np.inf)
    if problem.actuator_bounds is not None:
        low, high = problem.actuator_bounds
        bounds = (
            np.full(n_steps * nu, low, dtype=np.float64),
            np.full(n_steps * nu, high, dtype=np.float64),
        )

    try:
        opt_res = least_squares(
            residuals_fn,
            u_init,
            bounds=bounds,
            method="trf",
            max_nfev=problem.max_iterations * (n_steps * nu + 1),
            ftol=1e-7,
            xtol=1e-7,
        )
        u_opt = opt_res.x.reshape((n_steps, nu))
    except DimeResidualEvaluationError:
        nv = capability.n_v
        return DimeDynamicsWindowResult(
            success=False,
            states=(problem.initial_state,),
            controls=np.zeros((n_steps, nu)),
            transition_defects=np.zeros((n_steps, 2 * nv)),
            actuator_bound_residuals=np.zeros(1),
            control_variation_residuals=np.zeros(1),
            root_constraint_residuals=np.zeros(1),
            cost_breakdown={"total_cost": float("inf")},
            status="failed_dynamics_step",
            qualification_status=capability.status,
            n_iterations=0,
        )
    except Exception as exc:
        nv = capability.n_v
        return DimeDynamicsWindowResult(
            success=False,
            states=(problem.initial_state,),
            controls=np.zeros((n_steps, nu)),
            transition_defects=np.zeros((n_steps, 2 * nv)),
            actuator_bound_residuals=np.zeros(1),
            control_variation_residuals=np.zeros(1),
            root_constraint_residuals=np.zeros(1),
            cost_breakdown={"total_cost": float("inf")},
            status=f"solver_exception_{exc}",
            qualification_status=capability.status,
            n_iterations=0,
        )

    return _assemble_window_result(problem, opt_res, u_opt)
