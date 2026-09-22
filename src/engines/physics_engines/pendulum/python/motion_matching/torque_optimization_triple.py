"""Bounded Bernstein torque optimization for driven triple pendulum (CO-04 #10608).

Extends the TB-04 degree-6 Bernstein machinery to three joints (hub, arm, club)
while preserving the first-frame-before-step integration invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import numpy as np
from scipy.optimize import least_squares

from src.engines.pendulum_models.python.double_pendulum_model.physics.triple_pendulum import (
    TriplePendulumDynamics,
    TriplePendulumParameters,
    TriplePendulumState,
    TripleSegmentProperties,
)
from src.engines.physics_engines.pendulum.python.motion_matching.torque_optimization import (
    COEFFS_PER_JOINT,
    _BINOMIAL_6,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TripleBernsteinTorqueProfile",
    "TriplePendulumFitOptions",
    "TriplePendulumFitTarget",
    "TripleFitTrajectoryResult",
    "create_calibrated_triple_pendulum_dynamics",
    "fit_bounded_triple_pendulum",
    "forward_kinematics_triple_2d",
    "integrate_triple_pendulum_rollout",
]


@dataclass(frozen=True)
class TripleBernsteinTorqueProfile:
    """Continuous bounded degree-6 Bernstein torque trajectory for three joints."""

    hub_controls: np.ndarray
    arm_controls: np.ndarray
    club_controls: np.ndarray
    duration_s: float

    def __post_init__(self) -> None:
        for name, arr in (
            ("hub_controls", self.hub_controls),
            ("arm_controls", self.arm_controls),
            ("club_controls", self.club_controls),
        ):
            if np.asarray(arr).shape != (COEFFS_PER_JOINT,):
                raise ValueError(f"{name} must have shape ({COEFFS_PER_JOINT},)")
        if not (self.duration_s > 0.0 and math.isfinite(self.duration_s)):
            raise ValueError("duration_s must be finite and strictly positive")

    def evaluate(self, t: float) -> tuple[float, float, float]:
        """Evaluate continuous joint torques at time t (strictly bounded)."""
        s = float(np.clip(t / self.duration_s, 0.0, 1.0))
        s_powers = s ** np.arange(COEFFS_PER_JOINT)
        om_s_powers = (1.0 - s) ** np.arange(COEFFS_PER_JOINT - 1, -1, -1)
        basis = _BINOMIAL_6 * s_powers * om_s_powers
        return (
            float(np.dot(self.hub_controls, basis)),
            float(np.dot(self.arm_controls, basis)),
            float(np.dot(self.club_controls, basis)),
        )

    def as_controls(self) -> np.ndarray:
        return np.concatenate(
            [self.hub_controls, self.arm_controls, self.club_controls]
        )

    def curvature_penalty(self, weight: float = 0.05) -> np.ndarray:
        if weight <= 0.0:
            return np.zeros(0, dtype=np.float64)
        rw = math.sqrt(weight)
        parts = []
        for ctrl in (self.hub_controls, self.arm_controls, self.club_controls):
            parts.append(ctrl[2:] - 2.0 * ctrl[1:-1] + ctrl[:-2])
        return rw * np.concatenate(parts)

    def effort_penalty(self, weight: float = 0.001) -> np.ndarray:
        if weight <= 0.0:
            return np.zeros(0, dtype=np.float64)
        return math.sqrt(weight) * (self.as_controls() / 100.0)


@dataclass(frozen=True)
class TriplePendulumFitOptions:
    """Options for bounded triple-pendulum trajectory optimization."""

    max_nfev: int = 100
    tau_hub_bounds: tuple[float, float] = (-200.0, 200.0)
    tau_arm_bounds: tuple[float, float] = (-250.0, 250.0)
    tau_club_bounds: tuple[float, float] = (-150.0, 150.0)
    curvature_weight: float = 0.05
    effort_weight: float = 0.001
    initial_controls: np.ndarray | None = None


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


@dataclass(frozen=True)
class TripleFitTrajectoryResult:
    """Outcome of bounded triple pendulum optimization."""

    profile: TripleBernsteinTorqueProfile
    q_traj: np.ndarray
    v_traj: np.ndarray
    final_cost: float
    final_rmse_m: float
    unforced_rmse_m: float
    converged: bool
    evaluations: int
    iterations: int
    message: str
    hub_path: np.ndarray
    t0_evaluated_before_step: bool = True


def forward_kinematics_triple_2d(
    theta1: float,
    theta2: float,
    theta3: float,
    l1: float,
    l2: float,
    l3: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hub-relative joint1, grip (joint2), and clubhead tip positions."""
    abs2 = theta1 + theta2
    abs3 = theta1 + theta2 + theta3
    j1 = np.array([l1 * math.sin(theta1), -l1 * math.cos(theta1)], dtype=np.float64)
    grip = j1 + np.array([l2 * math.sin(abs2), -l2 * math.cos(abs2)], dtype=np.float64)
    tip = grip + np.array([l3 * math.sin(abs3), -l3 * math.cos(abs3)], dtype=np.float64)
    return j1, grip, tip


def create_calibrated_triple_pendulum_dynamics(
    l1: float, l2: float, l3: float
) -> TriplePendulumDynamics:
    """Instantiate triple dynamics with calibrated segment lengths."""
    if not all(math.isfinite(x) and x > 0.0 for x in (l1, l2, l3)):
        raise ValueError("segment lengths must be finite and strictly positive")
    base = TriplePendulumParameters.default()
    segs = []
    for seg, length in zip(base.segments, (l1, l2, l3), strict=True):
        segs.append(
            TripleSegmentProperties(
                length_m=float(length),
                mass_kg=float(seg.mass_kg),
                center_of_mass_ratio=float(seg.center_of_mass_ratio),
                inertia_about_com=float(seg.inertia_about_com),
            )
        )
    params = TriplePendulumParameters(
        segments=(segs[0], segs[1], segs[2]),
        damping=base.damping,
        gravity_enabled=base.gravity_enabled,
        gravity_m_s2=base.gravity_m_s2,
    )
    return TriplePendulumDynamics(params)


def integrate_triple_pendulum_rollout(
    dynamics: TriplePendulumDynamics,
    q0: np.ndarray,
    v0: np.ndarray,
    times: np.ndarray,
    profile: TripleBernsteinTorqueProfile,
    *,
    substeps: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate triple pendulum; frame 0 is evaluated before any step."""
    n_frames = len(times)
    q_traj = np.zeros((n_frames, 3), dtype=np.float64)
    v_traj = np.zeros((n_frames, 3), dtype=np.float64)
    hub_path = np.zeros((n_frames, 2), dtype=np.float64)

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
    # Hub remains the chain origin; joint-1 tip tracks moving "hub" body mass.
    j1, _, _ = forward_kinematics_triple_2d(
        state.theta1,
        state.theta2,
        state.theta3,
        dynamics.parameters.segments[0].length_m,
        dynamics.parameters.segments[1].length_m,
        dynamics.parameters.segments[2].length_m,
    )
    hub_path[0] = j1

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
        j1, _, _ = forward_kinematics_triple_2d(
            state.theta1,
            state.theta2,
            state.theta3,
            dynamics.parameters.segments[0].length_m,
            dynamics.parameters.segments[1].length_m,
            dynamics.parameters.segments[2].length_m,
        )
        hub_path[i + 1] = j1
    return q_traj, v_traj, hub_path


def _compute_tracking_errors(
    q_traj: np.ndarray,
    l1: float,
    l2: float,
    l3: float,
    target_grip: np.ndarray,
    target_head: np.ndarray,
    observed_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    res_list: list[float] = []
    head_sq: list[float] = []
    for i in range(len(q_traj)):
        if not observed_mask[i]:
            continue
        _, grip_i, head_i = forward_kinematics_triple_2d(
            float(q_traj[i, 0]),
            float(q_traj[i, 1]),
            float(q_traj[i, 2]),
            l1,
            l2,
            l3,
        )
        dg = grip_i - target_grip[i, :2]
        dh = head_i - target_head[i, :2]
        res_list.extend([float(dg[0]), float(dg[1]), float(dh[0]), float(dh[1])])
        head_sq.append(float(dh[0] ** 2 + dh[1] ** 2))
    res_arr = np.asarray(res_list, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(head_sq))) if head_sq else 0.0
    return res_arr, rmse


def fit_bounded_triple_pendulum(
    target: TriplePendulumFitTarget,
    dynamics: TriplePendulumDynamics,
    options: TriplePendulumFitOptions | None = None,
) -> TripleFitTrajectoryResult:
    """Optimize degree-6 Bernstein control points for three joints."""
    opts = options or TriplePendulumFitOptions()
    times = np.asarray(target.times, dtype=np.float64)
    duration = float(times[-1] - times[0])
    if duration <= 0.0:
        raise ValueError("Target duration must be strictly positive")

    grip_arr = np.asarray(target.grip, dtype=np.float64)[:, :2]
    head_arr = np.asarray(target.head, dtype=np.float64)[:, :2]
    observed_mask = np.asarray(
        np.isfinite(grip_arr).all(axis=1) & np.isfinite(head_arr).all(axis=1),
        dtype=bool,
    )
    if not np.any(observed_mask):
        raise ValueError("Target contains no valid observations")

    lo = np.concatenate(
        [
            np.full(COEFFS_PER_JOINT, opts.tau_hub_bounds[0]),
            np.full(COEFFS_PER_JOINT, opts.tau_arm_bounds[0]),
            np.full(COEFFS_PER_JOINT, opts.tau_club_bounds[0]),
        ]
    )
    hi = np.concatenate(
        [
            np.full(COEFFS_PER_JOINT, opts.tau_hub_bounds[1]),
            np.full(COEFFS_PER_JOINT, opts.tau_arm_bounds[1]),
            np.full(COEFFS_PER_JOINT, opts.tau_club_bounds[1]),
        ]
    )

    zero_profile = TripleBernsteinTorqueProfile(
        hub_controls=np.zeros(COEFFS_PER_JOINT),
        arm_controls=np.zeros(COEFFS_PER_JOINT),
        club_controls=np.zeros(COEFFS_PER_JOINT),
        duration_s=duration,
    )
    q_unforced, _, _ = integrate_triple_pendulum_rollout(
        dynamics, target.q0, target.v0, times, zero_profile
    )
    _, unforced_rmse = _compute_tracking_errors(
        q_unforced, target.l1, target.l2, target.l3, grip_arr, head_arr, observed_mask
    )

    n_eval = 0

    def residual_func(params: np.ndarray) -> np.ndarray:
        nonlocal n_eval
        n_eval += 1
        prof = TripleBernsteinTorqueProfile(
            hub_controls=params[:COEFFS_PER_JOINT],
            arm_controls=params[COEFFS_PER_JOINT : 2 * COEFFS_PER_JOINT],
            club_controls=params[2 * COEFFS_PER_JOINT :],
            duration_s=duration,
        )
        q_rollout, _, _ = integrate_triple_pendulum_rollout(
            dynamics, target.q0, target.v0, times, prof
        )
        tracking_res, _ = _compute_tracking_errors(
            q_rollout,
            target.l1,
            target.l2,
            target.l3,
            grip_arr,
            head_arr,
            observed_mask,
        )
        return np.concatenate(
            [
                tracking_res,
                prof.curvature_penalty(opts.curvature_weight),
                prof.effort_penalty(opts.effort_weight),
            ]
        )

    if opts.initial_controls is None:
        x0 = np.zeros(3 * COEFFS_PER_JOINT, dtype=np.float64)
    else:
        x0 = np.asarray(opts.initial_controls, dtype=np.float64).reshape(-1)
        if x0.size != 3 * COEFFS_PER_JOINT:
            raise ValueError(
                f"initial_controls must have length {3 * COEFFS_PER_JOINT}"
            )
        if not np.all(np.isfinite(x0)):
            raise ValueError("initial_controls must be finite")
        x0 = np.clip(x0, lo, hi)

    opt_res = least_squares(
        residual_func,
        x0,
        bounds=(lo, hi),
        max_nfev=max(opts.max_nfev, 5),
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
    )
    optimal = TripleBernsteinTorqueProfile(
        hub_controls=opt_res.x[:COEFFS_PER_JOINT],
        arm_controls=opt_res.x[COEFFS_PER_JOINT : 2 * COEFFS_PER_JOINT],
        club_controls=opt_res.x[2 * COEFFS_PER_JOINT :],
        duration_s=duration,
    )
    best_q, best_v, hub_path = integrate_triple_pendulum_rollout(
        dynamics, target.q0, target.v0, times, optimal
    )
    _, final_rmse = _compute_tracking_errors(
        best_q, target.l1, target.l2, target.l3, grip_arr, head_arr, observed_mask
    )
    return TripleFitTrajectoryResult(
        profile=optimal,
        q_traj=best_q,
        v_traj=best_v,
        final_cost=float(opt_res.cost),
        final_rmse_m=final_rmse,
        unforced_rmse_m=unforced_rmse,
        converged=bool(opt_res.success),
        evaluations=n_eval,
        iterations=int(getattr(opt_res, "nfev", n_eval)),
        message=str(opt_res.message),
        hub_path=hub_path,
        t0_evaluated_before_step=True,
    )
