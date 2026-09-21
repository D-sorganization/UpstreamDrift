"""Bounded Bernstein torque optimization for the closed-loop upper-body golfer (TB-06 #10591).

Reuses the existing Baumgarte-stabilized DAE solver in `constraint_solver.py` for every
integration step -- loop closure is enforced by the real constrained equations of motion,
never by projecting positions back onto the club after an unconstrained rollout. Reuses
the same degree-6 Bernstein basis convention introduced for the double/triple pendulum
fitters (TB-04 #10589, TB-05 #10590): 7 control points per actuated joint, strictly
bounded by construction via the partition of unity.

Singular or infeasible configurations produce an honest `GolferFitOutcome` with
`feasible=False`; no synthetic-success fallback is ever returned.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import OptimizeResult, least_squares

from src.engines.physics_engines.pendulum.python.motion_matching.adapters_golfer import (
    GolferFeasibilityReport,
    grip_and_clubhead_positions,
    solve_feasible_initial_state,
)
from src.shared.python.pendulum_simulator.constraint_solver import (
    constraint_forces,
    constraint_violation,
    equations_of_motion,
)
from src.shared.python.pendulum_simulator.physics_golfer import N_DOF, GolferParams
from src.shared.python.pendulum_simulator.golfer_dynamics import total_energy

# Degree-6 Bernstein basis (7 control points per actuated joint), matching the
# TB-04/TB-05 convention.
COEFFS_PER_JOINT: int = 7
POLY_DEGREE: int = 6

# Club DOF (theta_club) has no independent actuator; the remaining 7 generalized
# coordinates each carry an applied joint torque (hub, RS, RE, RH, LS, LE, LH).
N_ACTUATED_JOINTS: int = N_DOF - 1

_BINOMIAL_6: np.ndarray = np.array(
    [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0], dtype=np.float64
)


@dataclass(frozen=True)
class BernsteinGolferTorqueProfile:
    """Continuous bounded degree-6 Bernstein torque trajectory for all 7 joints."""

    controls: np.ndarray  # Shape (N_ACTUATED_JOINTS, COEFFS_PER_JOINT)
    duration_s: float

    def __post_init__(self) -> None:
        if self.controls.shape != (N_ACTUATED_JOINTS, COEFFS_PER_JOINT):
            raise ValueError(
                f"controls must have shape ({N_ACTUATED_JOINTS}, {COEFFS_PER_JOINT}), "
                f"got {self.controls.shape}"
            )
        if not (self.duration_s > 0.0 and math.isfinite(self.duration_s)):
            raise ValueError("duration_s must be finite and strictly positive")

    def evaluate(
        self, t: float
    ) -> tuple[float, float, float, float, float, float, float]:
        """Evaluate all 7 continuous joint torques at time t.

        Strictly bounded: min(controls[j]) <= tau_j(t) <= max(controls[j]).
        """
        s = float(np.clip(t / self.duration_s, 0.0, 1.0))
        s_powers = s ** np.arange(COEFFS_PER_JOINT)
        om_s_powers = (1.0 - s) ** np.arange(COEFFS_PER_JOINT - 1, -1, -1)
        basis = _BINOMIAL_6 * s_powers * om_s_powers
        values = tuple(
            float(np.dot(self.controls[j], basis)) for j in range(N_ACTUATED_JOINTS)
        )
        return (
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
        )

    def curvature_penalty(self, weight: float = 0.05) -> np.ndarray:
        """Penalize second differences (torque jerk) per joint."""
        if weight <= 0.0:
            return np.zeros(0, dtype=np.float64)
        rw = math.sqrt(weight)
        curv = (
            self.controls[:, 2:] - 2.0 * self.controls[:, 1:-1] + self.controls[:, :-2]
        )
        return rw * curv.reshape(-1)

    def effort_penalty(self, weight: float = 0.001) -> np.ndarray:
        """Penalize L2 magnitude of control points."""
        if weight <= 0.0:
            return np.zeros(0, dtype=np.float64)
        rw = math.sqrt(weight)
        return rw * (self.controls.reshape(-1) / 100.0)


def _make_torque_func(
    profile: BernsteinGolferTorqueProfile,
) -> Callable[[float], tuple[float, float, float, float, float, float, float]]:
    def _torque(
        t: float,
    ) -> tuple[float, float, float, float, float, float, float]:
        return profile.evaluate(t)

    return _torque


@dataclass(frozen=True)
class GolferRolloutResult:
    """One constrained rollout: trajectory plus per-frame constraint diagnostics."""

    q_traj: np.ndarray  # Shape (N, 8)
    v_traj: np.ndarray  # Shape (N, 8)
    constraint_residual_traj: np.ndarray  # Shape (N,) -- ||Phi(q)|| per frame
    lambda_traj: np.ndarray  # Shape (N, 4) -- constraint (reaction) forces per frame
    energy_start_J: float
    energy_end_J: float
    work_done_J: float


def integrate_golfer_rollout(
    params: GolferParams,
    q0: np.ndarray,
    v0: np.ndarray,
    times: np.ndarray,
    profile: BernsteinGolferTorqueProfile,
    *,
    substeps: int = 4,
    alpha: float = 5.0,
    beta: float = 5.0,
) -> GolferRolloutResult:
    """Integrate the constrained golfer EOM across actual timestamps.

    Reuses `constraint_solver.equations_of_motion` (Baumgarte-stabilized DAE) at
    every RK4 substep -- the loop-closure constraint is satisfied by the dynamics
    themselves, not enforced afterward. Frame 0 is evaluated at t0 before any
    integration step, matching the TB-04/TB-05 replay-verification convention.
    """
    if q0.shape != (N_DOF,) or v0.shape != (N_DOF,):
        raise ValueError("q0 and v0 must each have shape (8,)")
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 2:
        raise ValueError("times must contain at least 2 samples")

    n_frames = len(times)
    q_traj = np.zeros((n_frames, N_DOF), dtype=np.float64)
    v_traj = np.zeros((n_frames, N_DOF), dtype=np.float64)
    residual_traj = np.zeros(n_frames, dtype=np.float64)
    lambda_traj = np.zeros((n_frames, 4), dtype=np.float64)

    torque_func = _make_torque_func(profile)
    state = np.concatenate([q0, v0])
    q_traj[0] = q0
    v_traj[0] = v0
    residual_traj[0] = constraint_violation(state, params)
    lambda_traj[0] = constraint_forces(state, 0.0, params, torque_func)

    work_done = 0.0
    energy_start = total_energy(state, params)
    cur_time = float(times[0])

    for i in range(n_frames - 1):
        dt_frame = float(times[i + 1] - times[i])
        if dt_frame <= 0.0:
            raise ValueError(f"times must be strictly increasing, frame {i}")
        dt_sub = dt_frame / substeps

        tau_i = np.array(torque_func(cur_time)[:N_ACTUATED_JOINTS])
        for _ in range(substeps):
            # Classical RK4 on the constrained state derivative.
            k1 = equations_of_motion(state, cur_time, params, torque_func, alpha, beta)
            k2 = equations_of_motion(
                state + 0.5 * dt_sub * k1,
                cur_time + 0.5 * dt_sub,
                params,
                torque_func,
                alpha,
                beta,
            )
            k3 = equations_of_motion(
                state + 0.5 * dt_sub * k2,
                cur_time + 0.5 * dt_sub,
                params,
                torque_func,
                alpha,
                beta,
            )
            k4 = equations_of_motion(
                state + dt_sub * k3, cur_time + dt_sub, params, torque_func, alpha, beta
            )
            work_done += (
                float(np.dot(tau_i, state[N_DOF:][:N_ACTUATED_JOINTS])) * dt_sub
            )
            state = state + (dt_sub / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            cur_time += dt_sub

        q_traj[i + 1] = state[:N_DOF]
        v_traj[i + 1] = state[N_DOF:]
        residual_traj[i + 1] = constraint_violation(state, params)
        lambda_traj[i + 1] = constraint_forces(state, cur_time, params, torque_func)

    energy_end = total_energy(state, params)

    return GolferRolloutResult(
        q_traj=q_traj,
        v_traj=v_traj,
        constraint_residual_traj=residual_traj,
        lambda_traj=lambda_traj,
        energy_start_J=float(energy_start),
        energy_end_J=float(energy_end),
        work_done_J=float(work_done),
    )


@dataclass(frozen=True)
class GolferFitTarget:
    """Target 2D kinematics and initial conditions for closed-loop golfer fitting."""

    times: np.ndarray
    clubhead: np.ndarray  # Shape (N, 2)
    grip_right: np.ndarray  # Shape (N, 2)
    q0: np.ndarray
    v0: np.ndarray


@dataclass(frozen=True)
class GolferFitOptions:
    """Configurable options for bounded golfer trajectory optimization."""

    max_nfev: int = 60
    tau_bounds: tuple[float, float] = (-120.0, 120.0)
    curvature_weight: float = 0.05
    effort_weight: float = 0.001
    feasibility_max_iter: int = 50


@dataclass(frozen=True)
class GolferFitOutcome:
    """Honest outcome of bounded closed-loop golfer fitting.

    `feasible=False` is a first-class diagnostic result (singular constraint
    Jacobian, non-convergent projection, or degenerate target) -- callers must
    check it before trusting `profile`/`final_rmse_m`. This is never
    substituted with a synthetic success.
    """

    feasible: bool
    profile: BernsteinGolferTorqueProfile | None
    q_traj: np.ndarray | None
    v_traj: np.ndarray | None
    constraint_residual_max: float
    final_cost: float
    final_rmse_m: float
    unforced_rmse_m: float
    converged: bool
    evaluations: int
    message: str
    reason: str = ""


def _tracking_errors(
    q_traj: np.ndarray,
    params: GolferParams,
    target_clubhead: np.ndarray,
    target_grip_right: np.ndarray,
) -> tuple[np.ndarray, float]:
    n_frames = len(q_traj)
    residuals = np.zeros(n_frames * 4, dtype=np.float64)
    clubhead_sq = np.zeros(n_frames)
    for i in range(n_frames):
        positions = grip_and_clubhead_positions(q_traj[i], params)
        clubhead = np.array(positions["clubhead"])
        grip_right = np.array(positions["grip_right"])
        residuals[4 * i : 4 * i + 2] = clubhead - target_clubhead[i]
        residuals[4 * i + 2 : 4 * i + 4] = grip_right - target_grip_right[i]
        clubhead_sq[i] = float(
            np.dot(clubhead - target_clubhead[i], clubhead - target_clubhead[i])
        )
    rmse = float(np.sqrt(np.mean(clubhead_sq)))
    return residuals, rmse


def _infeasible_outcome(feasibility: GolferFeasibilityReport) -> GolferFitOutcome:
    """Build the honest failure outcome for an infeasible/non-convergent q0/v0."""
    return GolferFitOutcome(
        feasible=False,
        profile=None,
        q_traj=None,
        v_traj=None,
        constraint_residual_max=feasibility.position_residual,
        final_cost=float("nan"),
        final_rmse_m=float("nan"),
        unforced_rmse_m=float("nan"),
        converged=False,
        evaluations=0,
        message="initial state infeasible; optimizer not invoked",
        reason=feasibility.reason,
    )


def _run_bernstein_least_squares(
    params: GolferParams,
    q0: np.ndarray,
    v0: np.ndarray,
    times: np.ndarray,
    duration: float,
    target_clubhead: np.ndarray,
    target_grip_right: np.ndarray,
    opts: GolferFitOptions,
) -> tuple[BernsteinGolferTorqueProfile, GolferRolloutResult, OptimizeResult, int]:
    """Optimize Bernstein control points against clubhead/grip_right tracking targets."""
    n_params = N_ACTUATED_JOINTS * COEFFS_PER_JOINT
    lo = np.full(n_params, opts.tau_bounds[0])
    hi = np.full(n_params, opts.tau_bounds[1])
    n_eval = 0

    def residual_func(x: np.ndarray) -> np.ndarray:
        nonlocal n_eval
        n_eval += 1
        prof = BernsteinGolferTorqueProfile(
            controls=x.reshape(N_ACTUATED_JOINTS, COEFFS_PER_JOINT),
            duration_s=duration,
        )
        rollout = integrate_golfer_rollout(params, q0, v0, times, prof, substeps=1)
        tracking_res, _ = _tracking_errors(
            rollout.q_traj, params, target_clubhead, target_grip_right
        )
        curv_res = prof.curvature_penalty(opts.curvature_weight)
        eff_res = prof.effort_penalty(opts.effort_weight)
        return np.concatenate([tracking_res, curv_res, eff_res])

    x0 = np.zeros(n_params, dtype=np.float64)
    opt_res = least_squares(
        residual_func,
        x0,
        bounds=(lo, hi),
        max_nfev=max(opts.max_nfev, 5),
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
    )

    optimal_profile = BernsteinGolferTorqueProfile(
        controls=opt_res.x.reshape(N_ACTUATED_JOINTS, COEFFS_PER_JOINT),
        duration_s=duration,
    )
    best = integrate_golfer_rollout(params, q0, v0, times, optimal_profile, substeps=1)
    return optimal_profile, best, opt_res, n_eval


def fit_bounded_golfer(
    target: GolferFitTarget,
    params: GolferParams,
    options: GolferFitOptions | None = None,
) -> GolferFitOutcome:
    """Optimize degree-6 Bernstein control points for all 7 actuated joints.

    Loop closure is enforced throughout by the constrained rollout itself.
    Infeasible initial conditions (singular Jacobian, non-convergent
    projection) are diagnosed up front and returned as an honest failure
    outcome -- the optimizer is never invoked on an unclosable loop.
    """
    opts = options or GolferFitOptions()
    times = np.asarray(target.times, dtype=np.float64)
    duration = float(times[-1] - times[0])
    if duration <= 0.0:
        raise ValueError("Target duration must be strictly positive")

    feasibility = solve_feasible_initial_state(
        target.q0, target.v0, params, max_iter=opts.feasibility_max_iter
    )
    if not feasibility.feasible:
        return _infeasible_outcome(feasibility)

    q0, v0 = feasibility.q0, feasibility.v0
    target_clubhead = np.asarray(target.clubhead, dtype=np.float64)[:, :2]
    target_grip_right = np.asarray(target.grip_right, dtype=np.float64)[:, :2]

    zero_profile = BernsteinGolferTorqueProfile(
        controls=np.zeros((N_ACTUATED_JOINTS, COEFFS_PER_JOINT)),
        duration_s=duration,
    )
    unforced = integrate_golfer_rollout(params, q0, v0, times, zero_profile, substeps=1)
    _, unforced_rmse = _tracking_errors(
        unforced.q_traj, params, target_clubhead, target_grip_right
    )

    optimal_profile, best, opt_res, n_eval = _run_bernstein_least_squares(
        params, q0, v0, times, duration, target_clubhead, target_grip_right, opts
    )
    _, final_rmse = _tracking_errors(
        best.q_traj, params, target_clubhead, target_grip_right
    )

    return GolferFitOutcome(
        feasible=True,
        profile=optimal_profile,
        q_traj=best.q_traj,
        v_traj=best.v_traj,
        constraint_residual_max=float(np.max(best.constraint_residual_traj)),
        final_cost=float(opt_res.cost),
        final_rmse_m=final_rmse,
        unforced_rmse_m=unforced_rmse,
        converged=bool(opt_res.success),
        evaluations=n_eval,
        message=str(opt_res.message),
        reason="",
    )
