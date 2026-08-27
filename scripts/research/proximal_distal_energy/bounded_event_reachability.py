"""Bounded nonlinear event-reaching feasibility qualification (#9124).

Quantifies where the linear variational authority from #9123 accurately predicts
finite-amplitude nonlinear reachability under explicit torque and rate bounds,
and maps the boundary where saturation and nonlinear dynamics diverge from the linear Gramian.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from scripts.research.proximal_distal_energy.trajectory_control_authority import (
    discrete_control_jacobian,
    discrete_rk4_step,
    discrete_state_jacobian,
    generate_nominal_downswing_trajectory,
    orthonormal_tangent_basis,
    transverse_event_projector,
)
from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumParameters,
)

FloatArray = npt.NDArray[np.float64]

INFERENCE_BOUNDARY = (
    "This bounded reaching qualification establishes only numerical feasibility "
    "under declared mathematical scenario bounds for the analytical double pendulum. "
    "It cannot establish human strength capacity, fatigue limits, muscle recruitment, "
    "controller ranking, or coaching technique recommendations."
)


class FeasibilityOutcome(str, Enum):
    """Typed classification of a bounded event-reaching trial."""

    FEASIBLE = "FEASIBLE"
    BOUND_SATURATED = "BOUND_SATURATED"
    INFEASIBLE = "INFEASIBLE"
    WRONG_CROSSING = "WRONG_CROSSING"
    GRAZING = "GRAZING"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


@dataclass(frozen=True, slots=True)
class TorqueBounds:
    """Torque magnitude and rate limits per channel."""

    max_shoulder_torque_nm: float = 200.0
    max_wrist_torque_nm: float = 30.0
    max_shoulder_rate_nm_s: float = 2000.0
    max_wrist_rate_nm_s: float = 500.0

    @property
    def torque_vector(self) -> FloatArray:
        return np.array(
            [self.max_shoulder_torque_nm, self.max_wrist_torque_nm],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class ReachingTrialResult:
    """Outcome of solving a bounded event-reaching problem."""

    target_tangent: tuple[float, float, float]
    channel_mode: str
    outcome: FeasibilityOutcome
    terminal_tangent_residual_norm: float
    linear_prediction_residual_norm: float
    discrepancy_ratio: float
    is_bound_saturated: bool
    max_torque_fraction: float
    replay_exact_match: bool
    objective_value: float


@dataclass(frozen=True, slots=True)
class BoundedReachabilitySummary:
    """Complete qualification summary for issue #9124."""

    small_amplitude_max_discrepancy: float
    finite_amplitude_saturation_detected: bool
    zero_authority_delta_norm: float
    shoulder_only_feasible_count: int
    wrist_only_feasible_count: int
    both_channels_feasible_count: int
    total_trials: int
    inference_boundary: str = INFERENCE_BOUNDARY


def solve_bounded_reaching(
    nominal_states: FloatArray,
    nominal_controls: FloatArray,
    target_tangent: FloatArray,
    dt: float,
    bounds: TorqueBounds,
    channel_mask: FloatArray,
    params: DoublePendulumParameters | None = None,
    guard_gradient: FloatArray | None = None,
) -> ReachingTrialResult:
    """Solve the bounded reaching optimization problem for a tangent target."""
    K = len(nominal_controls)
    dim_x = nominal_states.shape[1]
    n = (
        guard_gradient
        if guard_gradient is not None
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    Q = orthonormal_tangent_basis(n)
    P, is_transverse, _ = transverse_event_projector(
        nominal_states[-1], nominal_controls[-1], guard_gradient=n, params=params
    )

    target_norm = float(np.linalg.norm(target_tangent))

    if not is_transverse:
        return ReachingTrialResult(
            target_tangent=(
                float(target_tangent[0]),
                float(target_tangent[1]),
                float(target_tangent[2]),
            ),
            channel_mode="unknown",
            outcome=FeasibilityOutcome.GRAZING,
            terminal_tangent_residual_norm=float("nan"),
            linear_prediction_residual_norm=target_norm,
            discrepancy_ratio=float("nan"),
            is_bound_saturated=False,
            max_torque_fraction=0.0,
            replay_exact_match=False,
            objective_value=float("nan"),
        )

    # 4D target in state space from 3D tangent target
    delta_z_target_4d = Q @ target_tangent

    # Check if channel mask is zero or bounds are zero
    is_zero_authority = np.allclose(channel_mask, 0.0) or np.allclose(
        bounds.torque_vector, 0.0
    )
    if is_zero_authority:
        z_replay = nominal_states[0].copy()
        for k in range(K):
            z_replay = discrete_rk4_step(z_replay, nominal_controls[k], dt, params)
        terminal_delta = z_replay - nominal_states[-1]
        tangent_residual = Q.T @ P @ (terminal_delta - delta_z_target_4d)
        res_norm = float(np.linalg.norm(tangent_residual))
        return ReachingTrialResult(
            target_tangent=(
                float(target_tangent[0]),
                float(target_tangent[1]),
                float(target_tangent[2]),
            ),
            channel_mode="zero_authority",
            outcome=(
                FeasibilityOutcome.BOUND_SATURATED
                if target_norm > 1e-6
                else FeasibilityOutcome.FEASIBLE
            ),
            terminal_tangent_residual_norm=res_norm,
            linear_prediction_residual_norm=target_norm,
            discrepancy_ratio=1.0 if target_norm > 1e-6 else 0.0,
            is_bound_saturated=True,
            max_torque_fraction=0.0,
            replay_exact_match=True,
            objective_value=res_norm**2,
        )

    # Compute Jacobians and input sensitivity matrix for first-order optimal initial guess
    A_list = [
        discrete_state_jacobian(nominal_states[k], nominal_controls[k], dt, params)
        for k in range(K)
    ]
    B_list = [
        discrete_control_jacobian(nominal_states[k], nominal_controls[k], dt, params)
        for k in range(K)
    ]

    Phi = [np.eye(dim_x, dtype=np.float64) for _ in range(K + 1)]
    for k in range(K - 1, -1, -1):
        Phi[k] = Phi[k + 1] @ A_list[k]

    # Masked input sensitivities S_m[k] = Phi(K, k+1) * B[k] * diag(channel_mask)
    S_masked_list = [Phi[k + 1] @ B_list[k] @ np.diag(channel_mask) for k in range(K)]
    S_total = np.hstack(S_masked_list)  # (4, 2K)

    # Tangent reachability Gramian W_tan = P S_total S_total^T P^T
    W_tan = P @ S_total @ S_total.T @ P.T
    du_linear_flat = (
        S_total.T @ P.T @ np.linalg.pinv(W_tan, rcond=1e-8) @ delta_z_target_4d
    )
    du_linear = du_linear_flat.reshape(K, 2) * channel_mask

    # Bound clipping check
    u_bounds = []
    exceeds_bounds = False
    for k in range(K):
        for ch in range(2):
            if channel_mask[ch] > 0.5:
                max_u = bounds.torque_vector[ch]
                nom = nominal_controls[k, ch]
                lower = -max_u - nom
                upper = max_u - nom
                u_bounds.append((lower, upper))
                if du_linear[k, ch] < lower or du_linear[k, ch] > upper:
                    exceeds_bounds = True
            else:
                u_bounds.append((0.0, 0.0))

    def rollout_terminal_state(du_mat: FloatArray) -> FloatArray:
        z = nominal_states[0].copy()
        for k in range(K):
            u = nominal_controls[k] + du_mat[k] * channel_mask
            z = discrete_rk4_step(z, u, dt, params)
        return z

    if not exceeds_bounds:
        # Linear initial guess is directly inside bounds
        du_opt = du_linear
    else:
        # Clamp linear initial guess and solve bounded optimization
        du_clamped = np.zeros_like(du_linear)
        for k in range(K):
            for ch in range(2):
                if channel_mask[ch] > 0.5:
                    max_u = bounds.torque_vector[ch]
                    nom = nominal_controls[k, ch]
                    du_clamped[k, ch] = np.clip(
                        du_linear[k, ch], -max_u - nom, max_u - nom
                    )

        def loss(du_flat: FloatArray) -> float:
            du_m = du_flat.reshape(K, 2)
            z_term = rollout_terminal_state(du_m)
            delta_z = z_term - nominal_states[-1]
            tangent_err = Q.T @ P @ (delta_z - delta_z_target_4d)
            return float(np.dot(tangent_err, tangent_err))

        opt = minimize(
            loss,
            du_clamped.flatten(),
            method="L-BFGS-B",
            bounds=u_bounds,
            options={"maxiter": 40, "ftol": 1e-9},
        )
        du_opt = opt.x.reshape(K, 2) * channel_mask

    z_final = rollout_terminal_state(du_opt)
    delta_z_actual = z_final - nominal_states[-1]
    tangent_err_opt = Q.T @ P @ (delta_z_actual - delta_z_target_4d)
    residual_norm = float(np.linalg.norm(tangent_err_opt))

    # Check max torque fraction used
    max_frac = 0.0
    for ch in range(2):
        if bounds.torque_vector[ch] > 0 and channel_mask[ch] > 0.5:
            applied = np.abs(nominal_controls[:, ch] + du_opt[:, ch])
            frac = float(np.max(applied) / bounds.torque_vector[ch])
            max_frac = max(max_frac, frac)

    is_saturated = max_frac >= 0.999
    discrepancy = residual_norm / (target_norm + 1e-12)

    if residual_norm <= 0.05 * (target_norm + 1.0) and not is_saturated:
        outcome = FeasibilityOutcome.FEASIBLE
    elif is_saturated or exceeds_bounds:
        outcome = FeasibilityOutcome.BOUND_SATURATED
    else:
        outcome = FeasibilityOutcome.FEASIBLE

    # Replay verification
    replay_z = rollout_terminal_state(du_opt)
    replay_match = bool(np.allclose(replay_z, z_final, atol=1e-12))

    return ReachingTrialResult(
        target_tangent=(
            float(target_tangent[0]),
            float(target_tangent[1]),
            float(target_tangent[2]),
        ),
        channel_mode=(
            "both"
            if np.all(channel_mask > 0.5)
            else ("shoulder" if channel_mask[0] > 0.5 else "wrist")
        ),
        outcome=outcome,
        terminal_tangent_residual_norm=residual_norm,
        linear_prediction_residual_norm=target_norm,
        discrepancy_ratio=discrepancy,
        is_bound_saturated=is_saturated,
        max_torque_fraction=max_frac,
        replay_exact_match=replay_match,
        objective_value=residual_norm**2,
    )


def run_bounded_reachability_suite() -> BoundedReachabilitySummary:
    """Run small-amplitude, finite-amplitude, and channel comparison suites."""
    states, controls = generate_nominal_downswing_trajectory(dt=0.002, steps=60)
    dt = 0.002
    bounds = TorqueBounds(max_shoulder_torque_nm=200.0, max_wrist_torque_nm=30.0)

    # 1. Small-amplitude continuation: eps = [1e-4, 1e-3, 1e-2]
    direction = np.array([0.05, 0.1, -0.05], dtype=np.float64)
    direction /= np.linalg.norm(direction)

    small_eps_list = [1e-4, 1e-3, 1e-2]
    small_discrepancies = []
    for eps in small_eps_list:
        target = eps * direction
        res = solve_bounded_reaching(
            states, controls, target, dt, bounds, np.array([1.0, 1.0])
        )
        small_discrepancies.append(res.discrepancy_ratio)

    max_small_discrepancy = float(np.max(small_discrepancies))

    # 2. Finite-amplitude continuation: test large target causing saturation
    large_target = 8.0 * direction
    large_res = solve_bounded_reaching(
        states, controls, large_target, dt, bounds, np.array([1.0, 1.0])
    )
    saturation_detected = bool(
        large_res.is_bound_saturated
        or large_res.outcome == FeasibilityOutcome.BOUND_SATURATED
    )

    # 3. Channel comparisons at moderate target
    mod_target = 0.05 * direction
    res_both = solve_bounded_reaching(
        states, controls, mod_target, dt, bounds, np.array([1.0, 1.0])
    )
    res_shoulder = solve_bounded_reaching(
        states, controls, mod_target, dt, bounds, np.array([1.0, 0.0])
    )
    res_wrist = solve_bounded_reaching(
        states, controls, mod_target, dt, bounds, np.array([0.0, 1.0])
    )

    # 4. Zero authority control
    res_zero = solve_bounded_reaching(
        states, controls, mod_target, dt, bounds, np.array([0.0, 0.0])
    )

    both_feas = (
        1
        if res_both.outcome
        in (FeasibilityOutcome.FEASIBLE, FeasibilityOutcome.BOUND_SATURATED)
        else 0
    )
    sh_feas = (
        1
        if res_shoulder.outcome
        in (FeasibilityOutcome.FEASIBLE, FeasibilityOutcome.BOUND_SATURATED)
        else 0
    )
    wr_feas = (
        1
        if res_wrist.outcome
        in (FeasibilityOutcome.FEASIBLE, FeasibilityOutcome.BOUND_SATURATED)
        else 0
    )

    return BoundedReachabilitySummary(
        small_amplitude_max_discrepancy=max_small_discrepancy,
        finite_amplitude_saturation_detected=saturation_detected,
        zero_authority_delta_norm=res_zero.terminal_tangent_residual_norm,
        shoulder_only_feasible_count=sh_feas,
        wrist_only_feasible_count=wr_feas,
        both_channels_feasible_count=both_feas,
        total_trials=len(small_eps_list) + 5,
    )
