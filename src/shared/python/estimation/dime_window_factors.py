"""Coupled State-Control Full-Dynamics Window Factors (#11421, #11426).

Provides decision vector layout, integration defects, bounded model discrepancy,
actuator limit penalties, underactuated root constraints, control variation
regularizers, and contact hooks for coupled state-control window estimation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal

import numpy as np
from scipy.optimize import least_squares

from src.shared.python.contracts import (
    ContractViolationError,
    PreconditionError,
    require,
)
from src.shared.python.estimation.dime_contracts import (
    DIME_CONTRACTS_VERSION,
    DimeCompleteState,
    DimeEstimationResult,
    DimeFullStepRequest,
    DynamicsProvider,
)
from src.shared.python.estimation.dime_observation_factors import (
    DimeObservationFactor,
)

DefectMode = Literal["exact", "soft"]
_VALID_DEFECT_MODES: Final[tuple[str, ...]] = ("exact", "soft")


@dataclass(frozen=True)
class DimeWindowFactorConfig:
    """Configuration parameters for coupled state-control window factor."""

    dt: float
    defect_mode: DefectMode = "soft"
    defect_weight: float = 1.0
    control_variation_weight: float = 0.05
    control_magnitude_weight: float = 0.001
    actuator_bound_penalty_weight: float = 50.0
    slack_bound: float | None = None
    slack_weight: float = 10.0
    underactuated_root_weight: float = 1000.0
    enable_model_discrepancy: bool = False

    def __post_init__(self) -> None:
        require(
            self.dt > 0.0 and np.isfinite(self.dt), "dt must be positive and finite"
        )
        require(self.defect_mode in _VALID_DEFECT_MODES, "Invalid defect_mode")
        require(self.defect_weight >= 0.0, "defect_weight must be non-negative")
        require(
            self.control_variation_weight >= 0.0,
            "control_variation_weight must be non-negative",
        )
        require(
            self.control_magnitude_weight >= 0.0,
            "control_magnitude_weight must be non-negative",
        )
        require(
            self.actuator_bound_penalty_weight >= 0.0,
            "actuator_bound_penalty_weight must be non-negative",
        )
        require(self.slack_weight >= 0.0, "slack_weight must be non-negative")
        require(
            self.underactuated_root_weight >= 0.0,
            "underactuated_root_weight must be non-negative",
        )
        if self.slack_bound is not None:
            require(
                self.slack_bound > 0.0 and np.isfinite(self.slack_bound),
                "slack_bound must be positive",
            )


@dataclass(frozen=True)
class DimeContactHook:
    """Exposes contact hooks without claiming stance completeness.

    Fail-closed: active contact asserted without a qualified contact provider
    raises PreconditionError. Stance completeness claimed on an analytic/unqualified
    contact provider raises PreconditionError.
    """

    contact_provider: Any | None = None
    assert_active_contact: bool = False
    contact_inputs: np.ndarray | None = None
    stance_completeness_claimed: bool = False

    def __post_init__(self) -> None:
        has_active_inputs = self.contact_inputs is not None and np.any(
            np.asarray(self.contact_inputs) != 0.0
        )
        if (
            self.assert_active_contact or has_active_inputs
        ) and self.contact_provider is None:
            raise PreconditionError(
                "Active contact asserted without a qualified contact provider. "
                "Stance completeness cannot be claimed."
            )
        if self.stance_completeness_claimed:
            raise PreconditionError(
                "DIME-05 validates fixed-base/flight analytic models with native no-contact "
                "semantics; stance completeness cannot be claimed by this factor."
            )

    @property
    def has_active_contact(self) -> bool:
        return self.assert_active_contact


@dataclass(frozen=True)
class DimeWindowResidualComponents:
    """Exported breakdown of all cost components and constraint residuals."""

    transition_defects: np.ndarray
    control_variation_residuals: np.ndarray
    control_magnitude_residuals: np.ndarray
    actuator_bound_residuals: np.ndarray
    underactuated_root_residuals: np.ndarray
    model_discrepancy_residuals: np.ndarray
    observation_residuals: np.ndarray | None
    slack_violations: tuple[float, ...]
    max_slack: float
    is_slack_bounded: bool
    total_cost: float


@dataclass(frozen=True)
class DimeDynamicsWindowResult:
    """Result of a coupled state-control window factor solve."""

    success: bool
    message: str
    trajectory_states: tuple[DimeCompleteState, ...]
    estimated_controls: np.ndarray
    estimated_slacks: np.ndarray | None
    components: DimeWindowResidualComponents
    defect_mode: DefectMode
    cost: float

    def to_estimation_result(
        self,
        model_hash: str,
        uncertainty_kind: str = "gaussian",
    ) -> DimeEstimationResult:
        """Convert result to standardized DIME estimation result receipt."""
        return DimeEstimationResult(
            contract_version=DIME_CONTRACTS_VERSION,
            trajectory_states=self.trajectory_states,
            estimated_controls=self.estimated_controls,
            residuals={
                "transition_defects": self.components.transition_defects.flatten(),
                "control_variation": self.components.control_variation_residuals.flatten(),
                "control_magnitude": self.components.control_magnitude_residuals.flatten(),
                "actuator_bounds": self.components.actuator_bound_residuals.flatten(),
                "underactuated_root": self.components.underactuated_root_residuals.flatten(),
            },
            uncertainty_kind="gaussian",
            uncertainty_summary={"cost": self.cost},
            model_hash=model_hash,
            qualification_status="implemented",
        )


class DimeWindowDecisionLayout:
    """Manages packing and unpacking between flat 1D parameter vectors and states/controls."""

    def __init__(
        self,
        n_times: int,
        n_q: int,
        n_v: int,
        n_u: int,
        enable_slack: bool = False,
    ) -> None:
        self.n_times = n_times
        self.n_q = n_q
        self.n_v = n_v
        self.n_u = n_u
        self.n_intervals = n_times - 1
        self.enable_slack = enable_slack

        self.state_dim = n_q + n_v
        self.states_total = self.n_times * self.state_dim
        self.controls_total = self.n_intervals * self.n_u
        self.slacks_total = self.n_intervals * self.n_v if enable_slack else 0
        self.total_dim = self.states_total + self.controls_total + self.slacks_total

    def pack(
        self,
        states: Sequence[DimeCompleteState],
        controls: np.ndarray,
        slacks: np.ndarray | None = None,
    ) -> np.ndarray:
        """Pack structured states, controls, and slacks into flat 1D parameter array."""
        state_parts = []
        for s in states:
            state_parts.extend(s.q.tolist())
            state_parts.extend(s.v.tolist())

        ctrl_part = np.asarray(controls, dtype=np.float64).flatten()
        parts = [np.array(state_parts, dtype=np.float64), ctrl_part]
        if self.enable_slack and slacks is not None:
            parts.append(np.asarray(slacks, dtype=np.float64).flatten())
        return np.concatenate(parts)

    def unpack(
        self,
        vec: np.ndarray,
        reference_states: Sequence[DimeCompleteState],
    ) -> tuple[tuple[DimeCompleteState, ...], np.ndarray, np.ndarray | None]:
        """Unpack flat 1D parameter array into structured states, controls, and slacks."""
        states = []
        offset = 0
        for i in range(self.n_times):
            ref = reference_states[i]
            q_i = vec[offset : offset + self.n_q]
            offset += self.n_q
            v_i = vec[offset : offset + self.n_v]
            offset += self.n_v
            states.append(
                DimeCompleteState(
                    t=ref.t,
                    q=q_i,
                    v=v_i,
                    model_hash=ref.model_hash,
                    units=dict(ref.units),
                )
            )

        controls = vec[offset : offset + self.controls_total].reshape(
            self.n_intervals, self.n_u
        )
        offset += self.controls_total

        slacks = None
        if self.enable_slack:
            slacks = vec[offset : offset + self.slacks_total].reshape(
                self.n_intervals, self.n_v
            )

        return tuple(states), controls, slacks


class DimeDynamicsWindowFactor:
    """Coupled state-control full-dynamics window factor."""

    def __init__(
        self,
        provider: DynamicsProvider,
        times: np.ndarray,
        config: DimeWindowFactorConfig,
        contact_hook: DimeContactHook | None = None,
        observation_factors: Sequence[DimeObservationFactor] = (),
    ) -> None:
        self.provider = provider
        self.times = np.asarray(times, dtype=np.float64)
        require(len(self.times) >= 2, "Window must contain at least 2 time nodes")
        require(
            bool(np.all(np.diff(self.times) > 0.0)), "times must be strictly monotonic"
        )

        self.config = config
        self.contact_hook = contact_hook or DimeContactHook()
        self.observation_factors = tuple(observation_factors)

        cap = self.provider.capability
        self.layout = DimeWindowDecisionLayout(
            n_times=len(self.times),
            n_q=cap.n_q,
            n_v=cap.n_v,
            n_u=len(cap.control_channels),
            enable_slack=self.config.enable_model_discrepancy,
        )

    def _evaluate_transition_defects(
        self,
        states: Sequence[DimeCompleteState],
        controls: np.ndarray,
        slacks: np.ndarray | None,
    ) -> np.ndarray:
        """Compute full integrated position and velocity defects across window intervals."""
        n_intervals = len(self.times) - 1
        nv = self.provider.capability.n_v
        defects = np.zeros((n_intervals, 2 * nv), dtype=np.float64)
        manifold = self.provider.capability.manifold

        for k in range(n_intervals):
            dt_k = float(self.times[k + 1] - self.times[k])
            req = DimeFullStepRequest(
                state=states[k],
                controls=controls[k],
                dt=dt_k,
                model_hash=self.provider.model_hash,
            )
            step_res = self.provider.step(req)
            pred_q = step_res.next_state.q
            pred_v = step_res.next_state.v

            if slacks is not None and self.config.enable_model_discrepancy:
                w_k = slacks[k]
                pred_v = pred_v + w_k * dt_k
                pred_q = manifold.retract(pred_q, 0.5 * w_k * (dt_k**2))

            # Tangent position difference and velocity difference
            dq_k = manifold.local_coordinates(pred_q, states[k + 1].q)
            dv_k = states[k + 1].v - pred_v
            defects[k, :nv] = dq_k
            defects[k, nv:] = dv_k

        return defects

    def _evaluate_control_residuals(
        self, controls: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute control variation (smoothness) and magnitude residuals."""
        n_intervals = len(self.times) - 1
        nu = self.layout.n_u

        # Control variation
        if n_intervals > 1:
            dt_steps = self.times[1:n_intervals] - self.times[0 : n_intervals - 1]
            diff_u = np.diff(controls, axis=0)
            rates = diff_u / dt_steps[:, None]
            var_res = rates * np.sqrt(self.config.control_variation_weight)
        else:
            var_res = np.empty((0, nu), dtype=np.float64)

        # Control magnitude
        mag_res = controls * np.sqrt(self.config.control_magnitude_weight)
        return var_res, mag_res

    def _evaluate_actuator_bounds(self, controls: np.ndarray) -> np.ndarray:
        """Compute penalty residuals for controls exceeding actuator channel limits."""
        channels = self.provider.capability.control_channels
        bound_res = np.zeros_like(controls)
        weight_sqrt = np.sqrt(self.config.actuator_bound_penalty_weight)

        for j, ch in enumerate(channels):
            u_min, u_max = ch.limits
            col = controls[:, j]
            excess_high = np.maximum(0.0, col - u_max)
            excess_low = np.maximum(0.0, u_min - col)
            bound_res[:, j] = (excess_high + excess_low) * weight_sqrt
        return bound_res

    def _evaluate_root_residuals(
        self, unactuated_root_wrenches: np.ndarray | None
    ) -> np.ndarray:
        """Compute underactuated root constraint residuals and reject arbitrary root shortcuts."""
        nv = self.provider.capability.n_v
        nu = self.layout.n_u
        n_unactuated = max(0, nv - nu)
        n_intervals = len(self.times) - 1

        if unactuated_root_wrenches is not None:
            raw_root = np.asarray(unactuated_root_wrenches, dtype=np.float64)
            if np.any(np.abs(raw_root) > 1e-6):
                raise PreconditionError(
                    "Arbitrary root wrench shortcut rejected: unactuated DOFs cannot be directly actuated."
                )
            return raw_root * np.sqrt(self.config.underactuated_root_weight)

        return np.zeros((n_intervals, n_unactuated), dtype=np.float64)

    def _evaluate_slack_residuals(
        self, slacks: np.ndarray | None
    ) -> tuple[np.ndarray, tuple[float, ...], float, bool]:
        """Compute model discrepancy slack residuals, bounds check, and violation tracking."""
        n_intervals = len(self.times) - 1
        nv = self.provider.capability.n_v

        if slacks is None or not self.config.enable_model_discrepancy:
            zero_res = np.zeros((n_intervals, nv), dtype=np.float64)
            return zero_res, (), 0.0, True

        slack_res = slacks * np.sqrt(self.config.slack_weight)
        slack_norms = np.linalg.norm(slacks, axis=1)
        max_slack = float(np.max(slack_norms)) if len(slack_norms) > 0 else 0.0

        violations = []
        is_bounded = True
        if self.config.slack_bound is not None:
            bound = self.config.slack_bound
            for s_norm in slack_norms:
                if s_norm > bound:
                    violations.append(float(s_norm))
            is_bounded = len(violations) == 0

        return slack_res, tuple(violations), max_slack, is_bounded

    def _evaluate_observation_residuals(
        self, states: Sequence[DimeCompleteState]
    ) -> np.ndarray | None:
        """Compute observation residuals across registered observation factors."""
        if not self.observation_factors:
            return None

        obs_residuals_list = []
        for factor in self.observation_factors:
            # Check if factor holds per-frame observations matching window times
            timing = getattr(factor, "_timing", None)
            obs_data = getattr(factor, "_obs_m", getattr(factor, "_obs_px", None))
            kinematics_fn = getattr(factor, "_kinematics_fn", None)
            operators = getattr(factor, "_operators", None)
            is_diagonal = getattr(factor, "_is_diagonal", True)

            if (
                timing is not None
                and obs_data is not None
                and kinematics_fn is not None
            ):
                # Per-node kinematics projection
                for k, s in enumerate(states):
                    pred = np.asarray(kinematics_fn(s.q), dtype=np.float64)
                    if pred.ndim == 2 and pred.shape[0] == 1:
                        target = obs_data[k]
                        diff = pred[0] - target
                        if operators is not None and is_diagonal:
                            diff = diff * operators[k]
                        obs_residuals_list.extend(diff.tolist())
                    else:
                        res = factor.evaluate_residuals(s.q)
                        obs_residuals_list.extend(res.tolist())
            else:
                for s in states:
                    res = factor.evaluate_residuals(s.q)
                    obs_residuals_list.extend(res.tolist())

        return np.array(obs_residuals_list, dtype=np.float64)

    def evaluate_components(
        self,
        states: Sequence[DimeCompleteState],
        controls: np.ndarray,
        slacks: np.ndarray | None = None,
        unactuated_root_wrenches: np.ndarray | None = None,
    ) -> DimeWindowResidualComponents:
        """Evaluate and export every cost component and constraint residual."""
        controls_arr = np.asarray(controls, dtype=np.float64)
        defects = self._evaluate_transition_defects(states, controls_arr, slacks)
        var_res, mag_res = self._evaluate_control_residuals(controls_arr)
        bound_res = self._evaluate_actuator_bounds(controls_arr)
        root_res = self._evaluate_root_residuals(unactuated_root_wrenches)
        slack_res, violations, max_slack, is_bounded = self._evaluate_slack_residuals(
            slacks
        )
        obs_res = self._evaluate_observation_residuals(states)

        # Scale defect in cost calculation
        defect_scale = (
            np.sqrt(self.config.defect_weight)
            if self.config.defect_mode == "soft"
            else 1.0
        )
        scaled_defects = defects * defect_scale

        cost = (
            0.5 * float(np.sum(scaled_defects**2))
            + 0.5 * float(np.sum(var_res**2))
            + 0.5 * float(np.sum(mag_res**2))
            + 0.5 * float(np.sum(bound_res**2))
            + 0.5 * float(np.sum(root_res**2))
            + 0.5 * float(np.sum(slack_res**2))
        )
        if obs_res is not None and len(obs_res) > 0:
            cost += 0.5 * float(np.sum(obs_res**2))

        return DimeWindowResidualComponents(
            transition_defects=defects,
            control_variation_residuals=var_res,
            control_magnitude_residuals=mag_res,
            actuator_bound_residuals=bound_res,
            underactuated_root_residuals=root_res,
            model_discrepancy_residuals=slack_res,
            observation_residuals=obs_res,
            slack_violations=violations,
            max_slack=max_slack,
            is_slack_bounded=is_bounded,
            total_cost=cost,
        )

    def _residual_fun(
        self,
        param_vec: np.ndarray,
        reference_states: Sequence[DimeCompleteState],
    ) -> np.ndarray:
        """Flattened residual function for scipy.optimize.least_squares."""
        try:
            states, controls, slacks = self.layout.unpack(param_vec, reference_states)
            comp = self.evaluate_components(states, controls, slacks)
            defect_scale = (
                np.sqrt(self.config.defect_weight)
                if self.config.defect_mode == "soft"
                else 1000.0
            )
            parts = [
                (comp.transition_defects * defect_scale).flatten(),
                comp.control_variation_residuals.flatten(),
                comp.control_magnitude_residuals.flatten(),
                comp.actuator_bound_residuals.flatten(),
                comp.underactuated_root_residuals.flatten(),
            ]
            if self.config.enable_model_discrepancy:
                parts.append(comp.model_discrepancy_residuals.flatten())
            if (
                comp.observation_residuals is not None
                and len(comp.observation_residuals) > 0
            ):
                parts.append(comp.observation_residuals.flatten())
            return np.concatenate(parts)
        except Exception:
            return np.full(100, 1.0e12, dtype=np.float64)

    def solve(
        self,
        initial_states: Sequence[DimeCompleteState],
        initial_controls: np.ndarray,
        initial_slacks: np.ndarray | None = None,
        max_iterations: int = 50,
    ) -> DimeDynamicsWindowResult:
        """Solve the coupled state-control window optimization problem."""
        try:
            # Pre-validate dynamics step to catch failing provider
            test_req = DimeFullStepRequest(
                state=initial_states[0],
                controls=initial_controls[0],
                dt=float(self.times[1] - self.times[0]),
                model_hash=self.provider.model_hash,
            )
            self.provider.step(test_req)
        except Exception as exc:
            comp_fail = DimeWindowResidualComponents(
                transition_defects=np.empty((0, 0)),
                control_variation_residuals=np.empty((0, 0)),
                control_magnitude_residuals=np.empty((0, 0)),
                actuator_bound_residuals=np.empty((0, 0)),
                underactuated_root_residuals=np.empty((0, 0)),
                model_discrepancy_residuals=np.empty((0, 0)),
                observation_residuals=None,
                slack_violations=(),
                max_slack=0.0,
                is_slack_bounded=True,
                total_cost=float("inf"),
            )
            return DimeDynamicsWindowResult(
                success=False,
                message=f"Dynamics provider failure: {exc}",
                trajectory_states=tuple(initial_states),
                estimated_controls=np.asarray(initial_controls),
                estimated_slacks=initial_slacks,
                components=comp_fail,
                defect_mode=self.config.defect_mode,
                cost=float("inf"),
            )

        x0 = self.layout.pack(initial_states, initial_controls, initial_slacks)
        res = least_squares(
            self._residual_fun,
            x0,
            args=(initial_states,),
            max_nfev=max_iterations * 20,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )

        opt_states, opt_controls, opt_slacks = self.layout.unpack(res.x, initial_states)
        comp = self.evaluate_components(opt_states, opt_controls, opt_slacks)

        return DimeDynamicsWindowResult(
            success=bool(res.success),
            message=str(res.message),
            trajectory_states=opt_states,
            estimated_controls=opt_controls,
            estimated_slacks=opt_slacks,
            components=comp,
            defect_mode=self.config.defect_mode,
            cost=comp.total_cost,
        )

    def compute_jacobian(self, param_vec: np.ndarray) -> np.ndarray:
        """Compute residual Jacobian using high-precision central differences."""
        eps = 1e-7
        states_ref, _, _ = self.layout.unpack(
            param_vec, [self.provider.get_state()] * len(self.times)
        )
        r0 = self._residual_fun(param_vec, states_ref)
        n_res = len(r0)
        n_vars = len(param_vec)
        jac = np.zeros((n_res, n_vars), dtype=np.float64)

        for col in range(n_vars):
            p_plus = param_vec.copy()
            p_minus = param_vec.copy()
            p_plus[col] += eps
            p_minus[col] -= eps
            r_plus = self._residual_fun(p_plus, states_ref)
            r_minus = self._residual_fun(p_minus, states_ref)
            jac[:, col] = (r_plus - r_minus) / (2.0 * eps)
        return jac

    def compute_finite_difference_jacobian(
        self, param_vec: np.ndarray, eps: float = 1e-6
    ) -> np.ndarray:
        """Compute residual Jacobian using standard forward finite differences."""
        states_ref, _, _ = self.layout.unpack(
            param_vec, [self.provider.get_state()] * len(self.times)
        )
        r0 = self._residual_fun(param_vec, states_ref)
        n_res = len(r0)
        n_vars = len(param_vec)
        jac = np.zeros((n_res, n_vars), dtype=np.float64)

        for col in range(n_vars):
            p_step = param_vec.copy()
            p_step[col] += eps
            r_step = self._residual_fun(p_step, states_ref)
            jac[:, col] = (r_step - r0) / eps
        return jac
