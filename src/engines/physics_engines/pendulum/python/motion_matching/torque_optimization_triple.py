"""Bounded Bernstein torque parameterization and optimization for triple pendulum (TB-05 #10590).

Parameterizes hub, arm, and club joint torques using a degree-6 Bernstein basis (7 control points
per joint = 21 total parameters). Bounding control points strictly bounds the continuous torque
everywhere on [0, T] via the partition of unity. Includes curvature and effort regularization.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import numpy as np
from scipy.optimize import least_squares

from src.engines.pendulum_models.python.double_pendulum_model.physics.triple_pendulum import (
    TriplePendulumDynamics,
    TriplePendulumState,
)
from src.engines.physics_engines.pendulum.python.motion_matching.adapters_triple import (
    forward_kinematics_3dof,
)
from src.engines.physics_engines.pendulum.python.motion_matching.torque_optimization import (
    prepare_fit_target_arrays,
)

logger = logging.getLogger(__name__)

# Degree 6 Bernstein basis (7 control points per joint)
COEFFS_PER_JOINT: int = 7
POLY_DEGREE: int = 6

# Precomputed binomial coefficients for degree 6: comb(6, k) for k = 0..6
_BINOMIAL_6: np.ndarray = np.array(
    [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0], dtype=np.float64
)


@dataclass(frozen=True)
class BernsteinTripleTorqueProfile:
    """Continuous bounded degree-6 Bernstein torque trajectory for hub, arm, and wrist."""

    hub_controls: np.ndarray  # Shape (7,)
    arm_controls: np.ndarray  # Shape (7,)
    wrist_controls: np.ndarray  # Shape (7,)
    duration_s: float

    def __post_init__(self) -> None:
        if self.hub_controls.shape != (COEFFS_PER_JOINT,):
            raise ValueError(f"hub_controls must have shape ({COEFFS_PER_JOINT},)")
        if self.arm_controls.shape != (COEFFS_PER_JOINT,):
            raise ValueError(f"arm_controls must have shape ({COEFFS_PER_JOINT},)")
        if self.wrist_controls.shape != (COEFFS_PER_JOINT,):
            raise ValueError(f"wrist_controls must have shape ({COEFFS_PER_JOINT},)")
        if not (self.duration_s > 0.0 and math.isfinite(self.duration_s)):
            raise ValueError("duration_s must be finite and strictly positive")

    def evaluate(self, t: float) -> tuple[float, float, float]:
        """Evaluate continuous joint torques (tau_hub, tau_arm, tau_wrist) at time t.

        Strictly bounded: min(c) <= tau(t) <= max(c) for all t in [0, duration_s].
        """
        s = float(np.clip(t / self.duration_s, 0.0, 1.0))
        s_powers = s ** np.arange(COEFFS_PER_JOINT)
        om_s_powers = (1.0 - s) ** np.arange(COEFFS_PER_JOINT - 1, -1, -1)
        basis = _BINOMIAL_6 * s_powers * om_s_powers

        tau1 = float(np.dot(self.hub_controls, basis))
        tau2 = float(np.dot(self.arm_controls, basis))
        tau3 = float(np.dot(self.wrist_controls, basis))
        return tau1, tau2, tau3

    def curvature_penalty(self, weight: float = 0.05) -> np.ndarray:
        """Penalize second differences Delta^2 c_k = c_{k+2} - 2*c_{k+1} + c_k (torque jerk)."""
        if weight <= 0.0:
            return np.zeros(0, dtype=np.float64)
        rw = math.sqrt(weight)
        c1 = self.hub_controls
        c2 = self.arm_controls
        c3 = self.wrist_controls
        curv1 = c1[2:] - 2.0 * c1[1:-1] + c1[:-2]
        curv2 = c2[2:] - 2.0 * c2[1:-1] + c2[:-2]
        curv3 = c3[2:] - 2.0 * c3[1:-1] + c3[:-2]
        return rw * np.concatenate([curv1, curv2, curv3])

    def effort_penalty(self, weight: float = 0.001) -> np.ndarray:
        """Penalize L2 magnitude of control points."""
        if weight <= 0.0:
            return np.zeros(0, dtype=np.float64)
        rw = math.sqrt(weight)
        all_ctrl = np.concatenate(
            [self.hub_controls, self.arm_controls, self.wrist_controls]
        )
        return rw * (all_ctrl / 100.0)


@dataclass(frozen=True)
class TripleFitTrajectoryResult:
    """Outcome of bounded triple pendulum optimization."""

    profile: BernsteinTripleTorqueProfile
    q_traj: np.ndarray  # Shape (N, 3)
    v_traj: np.ndarray  # Shape (N, 3)
    final_cost: float
    final_rmse_m: float
    unforced_rmse_m: float
    converged: bool
    evaluations: int
    iterations: int
    message: str
    t0_evaluated_before_step: bool = True


@dataclass(frozen=True)
class TriplePendulumFitOptions:
    """Configurable options for bounded triple pendulum trajectory optimization."""

    pivot: np.ndarray | None = None
    max_nfev: int = 100
    tau1_bounds: tuple[float, float] = (-300.0, 300.0)
    tau2_bounds: tuple[float, float] = (-250.0, 250.0)
    tau3_bounds: tuple[float, float] = (-150.0, 150.0)
    curvature_weight: float = 0.05
    effort_weight: float = 0.001


@dataclass(frozen=True)
class TriplePendulumFitTarget:
    """Target 2D kinematics and initial conditions for triple pendulum fitting."""

    times: np.ndarray
    grip: np.ndarray
    head: np.ndarray
    l1: float
    l2: float
    l3: float
    q0: np.ndarray
    v0: np.ndarray
    shoulder: np.ndarray | None = None


def integrate_triple_pendulum_rollout(
    dynamics: TriplePendulumDynamics,
    q0: np.ndarray,
    v0: np.ndarray,
    times: np.ndarray,
    profile: BernsteinTripleTorqueProfile,
    *,
    substeps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate triple pendulum trajectory from q0, v0 across actual timestamps.

    Critical invariant: evaluates state at t0 before any integration step is taken.
    """
    n_frames = len(times)
    q_traj: np.ndarray = np.zeros((n_frames, 3), dtype=np.float64)
    v_traj: np.ndarray = np.zeros((n_frames, 3), dtype=np.float64)

    state = TriplePendulumState(
        theta1=float(q0[0]),
        theta2=float(q0[1]),
        theta3=float(q0[2]),
        omega1=float(v0[0]),
        omega2=float(v0[1]),
        omega3=float(v0[2]),
    )
    q_traj[0] = [state.theta1, state.theta2, state.theta3]
    v_traj[0] = [state.omega1, state.omega2, state.omega3]

    cur_time = float(times[0])
    for i in range(n_frames - 1):
        dt_frame = float(times[i + 1] - times[i])
        if dt_frame <= 0.0:
            dt_frame = 1e-4

        dt_sub = dt_frame / substeps
        for _ in range(substeps):
            control = profile.evaluate(cur_time)
            state = dynamics.step(cur_time, state, dt_sub, control)
            cur_time += dt_sub

        q_traj[i + 1] = [state.theta1, state.theta2, state.theta3]
        v_traj[i + 1] = [state.omega1, state.omega2, state.omega3]

    return q_traj, v_traj


def _compute_triple_tracking_errors(
    q_traj: np.ndarray,
    l1: float,
    l2: float,
    l3: float,
    pivot: np.ndarray,
    target_grip: np.ndarray,
    target_head: np.ndarray,
    observed_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute Euclidean marker tracking residuals and RMS distance error."""
    n_frames = len(q_traj)
    res_list: list[float] = []

    for i in range(n_frames):
        if not observed_mask[i]:
            continue
        _, wrist_i, head_i = forward_kinematics_3dof(
            float(q_traj[i, 0]),
            float(q_traj[i, 1]),
            float(q_traj[i, 2]),
            l1,
            l2,
            l3,
            pivot,
        )
        res_list.append(float(wrist_i[0] - target_grip[i, 0]))
        res_list.append(float(wrist_i[1] - target_grip[i, 1]))
        res_list.append(float(head_i[0] - target_head[i, 0]))
        res_list.append(float(head_i[1] - target_head[i, 1]))

    res_arr = np.array(res_list, dtype=np.float64)
    head_sq = (res_arr[2::4] ** 2) + (res_arr[3::4] ** 2)
    rmse = float(np.sqrt(np.mean(head_sq))) if len(head_sq) > 0 else 0.0
    return res_arr, rmse


def _evaluate_triple_unforced_baseline(
    dynamics: TriplePendulumDynamics,
    target: TriplePendulumFitTarget,
    p0: np.ndarray,
    observed_mask: np.ndarray,
    duration: float,
) -> float:
    """Evaluate unforced baseline RMSE."""
    zero_profile = BernsteinTripleTorqueProfile(
        hub_controls=np.zeros(COEFFS_PER_JOINT),
        arm_controls=np.zeros(COEFFS_PER_JOINT),
        wrist_controls=np.zeros(COEFFS_PER_JOINT),
        duration_s=duration,
    )
    q_unforced, _ = integrate_triple_pendulum_rollout(
        dynamics, target.q0, target.v0, target.times, zero_profile
    )
    grip_arr = np.asarray(target.grip, dtype=np.float64)[:, :2]
    head_arr = np.asarray(target.head, dtype=np.float64)[:, :2]
    _, unforced_rmse = _compute_triple_tracking_errors(
        q_unforced,
        target.l1,
        target.l2,
        target.l3,
        p0,
        grip_arr,
        head_arr,
        observed_mask,
    )
    return unforced_rmse


def _build_triple_torque_bounds(
    opts: TriplePendulumFitOptions,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the per-parameter (lo, hi) bound vectors for the 3-joint Bernstein basis."""
    lo = np.concatenate(
        [
            np.full(COEFFS_PER_JOINT, opts.tau1_bounds[0]),
            np.full(COEFFS_PER_JOINT, opts.tau2_bounds[0]),
            np.full(COEFFS_PER_JOINT, opts.tau3_bounds[0]),
        ]
    )
    hi = np.concatenate(
        [
            np.full(COEFFS_PER_JOINT, opts.tau1_bounds[1]),
            np.full(COEFFS_PER_JOINT, opts.tau2_bounds[1]),
            np.full(COEFFS_PER_JOINT, opts.tau3_bounds[1]),
        ]
    )
    return lo, hi


def fit_bounded_triple_pendulum(
    target: TriplePendulumFitTarget,
    dynamics: TriplePendulumDynamics,
    options: TriplePendulumFitOptions | None = None,
) -> TripleFitTrajectoryResult:
    """Optimize 21 degree-6 Bernstein control points subject to explicit bounds and regularization."""
    opts = options or TriplePendulumFitOptions()
    p0, times, duration, grip_arr, head_arr, observed_mask = prepare_fit_target_arrays(
        target, opts.pivot
    )

    lo, hi = _build_triple_torque_bounds(opts)

    unforced_rmse = _evaluate_triple_unforced_baseline(
        dynamics, target, p0, observed_mask, duration
    )

    n_eval = 0

    def residual_func(params: np.ndarray) -> np.ndarray:
        nonlocal n_eval
        n_eval += 1
        prof = BernsteinTripleTorqueProfile(
            hub_controls=params[:COEFFS_PER_JOINT],
            arm_controls=params[COEFFS_PER_JOINT : 2 * COEFFS_PER_JOINT],
            wrist_controls=params[2 * COEFFS_PER_JOINT :],
            duration_s=duration,
        )
        q_rollout, _ = integrate_triple_pendulum_rollout(
            dynamics, target.q0, target.v0, times, prof
        )
        tracking_res, _ = _compute_triple_tracking_errors(
            q_rollout,
            target.l1,
            target.l2,
            target.l3,
            p0,
            grip_arr,
            head_arr,
            observed_mask,
        )
        curv_res = prof.curvature_penalty(opts.curvature_weight)
        eff_res = prof.effort_penalty(opts.effort_weight)
        return np.concatenate([tracking_res, curv_res, eff_res])

    x0: np.ndarray = np.zeros(3 * COEFFS_PER_JOINT, dtype=np.float64)
    opt_res = least_squares(
        residual_func,
        x0,
        bounds=(lo, hi),
        max_nfev=max(opts.max_nfev, 5),
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
    )

    optimal_profile = BernsteinTripleTorqueProfile(
        hub_controls=opt_res.x[:COEFFS_PER_JOINT],
        arm_controls=opt_res.x[COEFFS_PER_JOINT : 2 * COEFFS_PER_JOINT],
        wrist_controls=opt_res.x[2 * COEFFS_PER_JOINT :],
        duration_s=duration,
    )
    best_q, best_v = integrate_triple_pendulum_rollout(
        dynamics, target.q0, target.v0, times, optimal_profile
    )
    _, final_rmse = _compute_triple_tracking_errors(
        best_q, target.l1, target.l2, target.l3, p0, grip_arr, head_arr, observed_mask
    )

    return TripleFitTrajectoryResult(
        profile=optimal_profile,
        q_traj=best_q,
        v_traj=best_v,
        final_cost=float(opt_res.cost),
        final_rmse_m=final_rmse,
        unforced_rmse_m=unforced_rmse,
        converged=bool(opt_res.success),
        evaluations=n_eval,
        iterations=int(getattr(opt_res, "nfev", n_eval)),
        message=str(opt_res.message),
        t0_evaluated_before_step=True,
    )
