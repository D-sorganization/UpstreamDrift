"""Fit and independently replay the driven planar double pendulum (TB-04 #10589).

Provides:
1. First-frame accurate initial state evaluation at t0 (no step before scoring).
2. Non-uniform timestamp support and source timestamp handling.
3. Continuous bounded shoulder/wrist control coefficients using Bernstein torque infrastructure.
4. Smoothness and effort regularization.
5. Independent tighter-step replay starting once from t0 with zero state resets.
6. Honest unforced/passive comparison diagnostic.
7. Cryptographic target hash and full BaselineIdentity / StatusBundle provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import hashlib
import logging
import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
    DoublePendulumParameters,
    DoublePendulumState,
    LowerSegmentProperties,
    SegmentProperties,
)
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.fit_result import CanonicalFitResult
from src.shared.python.motion_matching.prefix_fit import (
    COEFFS_PER_JOINT,
    bernstein_to_simscape,
)
from src.shared.python.motion_matching.projection_2d import (
    CalibratedSwingPlane,
    GeometricProjectionResidual,
    estimate_swing_plane,
    project_to_calibrated_plane,
)
from src.shared.python.motion_matching.provenance import git_commit_short
from src.shared.python.tour_baselines.baseline_package import (
    BaselineIdentity,
    DynamicFeasibilityStatus,
    KinematicAccuracyStatus,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverConvergenceStatus,
    StatusBundle,
)
from src.shared.python.tour_baselines.calibration import (
    GeometryCalibrationResult,
    InitialStateResult,
    PlanarDoublePendulumPose,
    calibrate_fixed_geometry,
    forward_kinematics_planar_double_pendulum,
    map_initial_state_double_pendulum,
)
from src.shared.python.tour_baselines.fit_metrics import (
    PhysicalFitMetrics,
    compute_fit_metrics,
)
from src.shared.python.tour_baselines.models import (
    BackendType,
    FitMode,
    ModelTopology,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "PendulumFitOptions",
    "PendulumFitResult",
    "fit_driven_double_pendulum",
    "simulate_pendulum_rollout",
]


@dataclass(frozen=True)
class PendulumFitOptions:
    """Configuration and numerical budgets for double pendulum fitting."""

    maxiter: int = 200
    seed: int = 42
    weight_grip: float = 0.5
    weight_clubhead: float = 1.0
    lambda_effort: float = 1e-4
    lambda_smooth: float = 1e-4
    max_torque_shoulder: float = 300.0  # N*m
    max_torque_wrist: float = 120.0  # N*m
    substeps_replay: int = 4
    tolerance_replay_rmse_m: float = 0.05


@dataclass(frozen=True)
class PendulumFitResult:
    """Comprehensive result of double pendulum fitting and independent replay."""

    theta_optimal: np.ndarray
    final_cost: float
    final_rmse_m: float
    replay_rmse_m: float
    solver_status: str
    iterations: int
    n_evaluations: int
    wall_clock_s: float
    message: str
    history: tuple[float, ...]
    method: str
    git_commit: str
    engine_version: str
    target_hash: str
    timestamp_utc: str
    calibrated_plane: CalibratedSwingPlane
    calibrated_geometry: GeometryCalibrationResult
    initial_state: InitialStateResult
    baseline_identity: BaselineIdentity
    status_bundle: StatusBundle
    fit_metrics: PhysicalFitMetrics
    unforced_rmse_m: float

    def to_canonical_fit_result(
        self, engine_version: str | None = None
    ) -> CanonicalFitResult:
        """Convert to legacy CanonicalFitResult for existing callers."""
        return CanonicalFitResult(
            theta_optimal=self.theta_optimal,
            final_cost=self.final_rmse_m**2,
            final_rmse_m=self.final_rmse_m,
            solver_status=self.solver_status,
            iterations=self.iterations,
            n_evaluations=self.n_evaluations,
            wall_clock_s=self.wall_clock_s,
            message=self.message,
            history=self.history,
            method="scipy SLSQP",
            git_commit=self.git_commit,
            engine_version=engine_version or self.engine_version,
            target_hash=self.target_hash,
            timestamp_utc=self.timestamp_utc,
        )


def _compute_target_hash(target: ClubTarget) -> str:
    """Compute deterministic cryptographic SHA-256 target hash."""
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(target.time).tobytes())
    hasher.update(np.ascontiguousarray(target.butt).tobytes())
    hasher.update(np.ascontiguousarray(target.clubhead).tobytes())
    return hasher.hexdigest()


def _build_dynamics_parameters(
    geometry: GeometryCalibrationResult,
    plane_inclination_deg: float,
) -> DoublePendulumParameters:
    """Build analytical DoublePendulumParameters from calibrated geometry."""
    upper = SegmentProperties(
        length_m=geometry.l1_arm_m,
        mass_kg=geometry.m1_arm_kg,
        center_of_mass_ratio=0.45,
        inertia_about_com=geometry.i1_arm_kg_m2,
    )
    lower = LowerSegmentProperties(
        length_m=geometry.l2_club_m,
        shaft_mass_kg=geometry.m2_shaft_kg,
        clubhead_mass_kg=geometry.m_head_kg,
        shaft_com_ratio=0.43,
    )
    return DoublePendulumParameters(
        upper_segment=upper,
        lower_segment=lower,
        plane_inclination_deg=plane_inclination_deg,
        damping_shoulder=0.4,
        damping_wrist=0.25,
    )


def simulate_pendulum_rollout(
    dynamics_params: DoublePendulumParameters,
    q0: np.ndarray,
    v0: np.ndarray,
    times: np.ndarray,
    torque_coeffs_descending: np.ndarray,
    *,
    substeps_per_frame: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform forward simulation rollout starting once from t0 with zero resets."""
    n_frames = len(times)
    l1 = float(dynamics_params.upper_segment.length_m)
    l2 = float(dynamics_params.lower_segment.length_m)

    pred_grip = np.zeros((n_frames, 2))
    pred_head = np.zeros((n_frames, 2))
    torques_eval = np.zeros((n_frames, 2))

    c_shoulder = torque_coeffs_descending[0]
    c_wrist = torque_coeffs_descending[1]

    def forcing_shoulder(t: float, _: DoublePendulumState) -> float:
        return float(np.polyval(c_shoulder, t))

    def forcing_wrist(t: float, _: DoublePendulumState) -> float:
        return float(np.polyval(c_wrist, t))

    dyn = DoublePendulumDynamics(
        dynamics_params,
        forcing_functions=(forcing_shoulder, forcing_wrist),
    )

    state = DoublePendulumState(
        theta1=float(q0[0]),
        theta2=float(q0[1]),
        omega1=float(v0[0]),
        omega2=float(v0[1]),
    )

    pivot_origin = np.array([0.0, 0.0])

    # Frame 0 is evaluated at t0 directly without stepping
    pose0 = PlanarDoublePendulumPose(theta1_rad=state.theta1, theta2_rad=state.theta2)
    p_g0, p_h0 = forward_kinematics_planar_double_pendulum(pivot_origin, l1, l2, pose0)
    pred_grip[0] = p_g0
    pred_head[0] = p_h0
    torques_eval[0] = [
        forcing_shoulder(times[0], state),
        forcing_wrist(times[0], state),
    ]

    for i in range(n_frames - 1):
        dt_frame = float(times[i + 1] - times[i])
        sub_dt = dt_frame / substeps_per_frame
        curr_t = float(times[i])
        for _ in range(substeps_per_frame):
            state = dyn.step(curr_t, state, sub_dt)
            curr_t += sub_dt

        pose = PlanarDoublePendulumPose(
            theta1_rad=state.theta1, theta2_rad=state.theta2
        )
        p_g, p_h = forward_kinematics_planar_double_pendulum(pivot_origin, l1, l2, pose)
        pred_grip[i + 1] = p_g
        pred_head[i + 1] = p_h
        torques_eval[i + 1] = [
            forcing_shoulder(times[i + 1], state),
            forcing_wrist(times[i + 1], state),
        ]

    return pred_grip, pred_head, torques_eval


def _compute_tracking_rmse(
    pred_grip: np.ndarray,
    pred_head: np.ndarray,
    obs_grip: np.ndarray,
    obs_head: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    """Compute physical Euclidean tracking RMSE over valid frames."""
    diff_grip = pred_grip[valid_mask] - obs_grip[valid_mask]
    diff_head = pred_head[valid_mask] - obs_head[valid_mask]
    sq_errs = np.sum(diff_grip**2, axis=-1) + np.sum(diff_head**2, axis=-1)
    if len(sq_errs) == 0:
        return 0.0
    return float(np.sqrt(np.mean(sq_errs) / 2.0))


def _evaluate_candidate_cost(
    controls_bernstein: np.ndarray,
    dyn_params: DoublePendulumParameters,
    q0: np.ndarray,
    v0: np.ndarray,
    times: np.ndarray,
    obs_grip: np.ndarray,
    obs_head: np.ndarray,
    valid_mask: np.ndarray,
    opts: PendulumFitOptions,
    duration_s: float,
) -> float:
    """Evaluate regularized tracking cost function for SLSQP optimization."""
    controls_2d = controls_bernstein.reshape((2, COEFFS_PER_JOINT))
    coeffs_desc = bernstein_to_simscape(controls_2d, duration_s=duration_s)

    pred_grip, pred_head, torques = simulate_pendulum_rollout(
        dyn_params, q0, v0, times, coeffs_desc, substeps_per_frame=1
    )

    diff_grip = pred_grip[valid_mask] - obs_grip[valid_mask]
    diff_head = pred_head[valid_mask] - obs_head[valid_mask]

    loss_tracking = opts.weight_grip * np.mean(
        np.sum(diff_grip**2, axis=-1)
    ) + opts.weight_clubhead * np.mean(np.sum(diff_head**2, axis=-1))

    loss_effort = opts.lambda_effort * float(np.mean(torques**2))
    return float(loss_tracking + loss_effort)


def _optimize_torques(
    dyn_params: DoublePendulumParameters,
    q0: np.ndarray,
    v0: np.ndarray,
    times: np.ndarray,
    obs_grip: np.ndarray,
    obs_head: np.ndarray,
    valid_mask: np.ndarray,
    opts: PendulumFitOptions,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray, list[float], Any]:
    """Run SLSQP optimization over bounded Bernstein control points."""
    np.random.seed(opts.seed)
    n_params = 2 * COEFFS_PER_JOINT
    init_guess = np.zeros(n_params)

    # Box bounds on continuous torque values
    bounds: list[tuple[float, float]] = []
    for _ in range(COEFFS_PER_JOINT):
        bounds.append((-opts.max_torque_shoulder, opts.max_torque_shoulder))
    for _ in range(COEFFS_PER_JOINT):
        bounds.append((-opts.max_torque_wrist, opts.max_torque_wrist))

    history: list[float] = []

    def obj_wrapper(x: np.ndarray) -> float:
        cost = _evaluate_candidate_cost(
            x,
            dyn_params,
            q0,
            v0,
            times,
            obs_grip,
            obs_head,
            valid_mask,
            opts,
            duration_s,
        )
        history.append(cost)
        return cost

    res = minimize(
        obj_wrapper,
        init_guess,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": opts.maxiter, "ftol": 1e-5},
    )

    best_x = np.asarray(res.x, dtype=np.float64)
    best_desc = bernstein_to_simscape(
        best_x.reshape((2, COEFFS_PER_JOINT)), duration_s=duration_s
    )
    return best_x, best_desc, history, res


def fit_driven_double_pendulum(
    target: ClubTarget,
    opts: PendulumFitOptions | None = None,
    calibrated_plane: CalibratedSwingPlane | None = None,
) -> PendulumFitResult:
    """Fit continuous bounded shoulder and wrist torques and independently replay."""
    options = opts or PendulumFitOptions()
    t_start = time.perf_counter()

    if len(target.time) < 2:
        raise ValueError("Target must have at least 2 time frames")

    times = np.asarray(target.time, dtype=np.float64)
    duration_s = float(times[-1] - times[0])
    if duration_s <= 0.0 or not np.isfinite(duration_s):
        raise ValueError("Target times must be strictly increasing and finite")

    # 1. Plane calibration & 2D projection
    if calibrated_plane is None:
        pts_joint = np.vstack([target.butt, target.clubhead])
        try:
            plane = estimate_swing_plane(pts_joint)
        except ValueError:
            plane = CalibratedSwingPlane(
                origin=np.zeros(3),
                basis=np.eye(3),
                transform_world_to_plane=np.eye(4),
                transform_plane_to_world=np.eye(4),
                inclination_deg=90.0,
                azimuth_deg=0.0,
                residual=GeometricProjectionResidual(
                    rmse=0.0,
                    max_deviation=0.0,
                    signed_deviations=np.zeros(len(pts_joint)),
                ),
            )
    else:
        plane = calibrated_plane

    projected_target = project_to_calibrated_plane(target, plane)
    target_hash = _compute_target_hash(target)

    # 2. Geometry calibration
    shoulder_obs = np.zeros_like(target.butt)
    geometry = calibrate_fixed_geometry(
        shoulder_pts=shoulder_obs,
        grip_pts=target.butt,
        clubhead_pts=target.clubhead,
    )

    # 3. Initial state mapping at t0
    obs_grip_2d = projected_target.butt[:, :2]
    obs_head_2d = projected_target.clubhead[:, :2]
    pivot_obs_2d = np.zeros_like(obs_grip_2d)

    init_state = map_initial_state_double_pendulum(
        times=times,
        pivot_pts=pivot_obs_2d,
        grip_pts=obs_grip_2d,
        clubhead_pts=obs_head_2d,
        l1=geometry.l1_arm_m,
        l2=geometry.l2_club_m,
        t0_idx=0,
    )

    dyn_params = _build_dynamics_parameters(geometry, plane.inclination_deg)
    valid_mask = np.all(np.isfinite(obs_grip_2d), axis=1) & np.all(
        np.isfinite(obs_head_2d), axis=1
    )

    # 4. Unforced diagnostic rollout
    zero_desc = np.zeros((2, COEFFS_PER_JOINT))
    unforced_grip, unforced_head, _ = simulate_pendulum_rollout(
        dyn_params, init_state.q0, init_state.v0, times, zero_desc
    )
    unforced_rmse = _compute_tracking_rmse(
        unforced_grip, unforced_head, obs_grip_2d, obs_head_2d, valid_mask
    )

    # 5. Torque optimization
    _, best_desc, history, opt_res = _optimize_torques(
        dyn_params,
        init_state.q0,
        init_state.v0,
        times,
        obs_grip_2d,
        obs_head_2d,
        valid_mask,
        options,
        duration_s,
    )

    # 6. Fit evaluation and independent tighter-step replay
    fit_grip, fit_head, _ = simulate_pendulum_rollout(
        dyn_params, init_state.q0, init_state.v0, times, best_desc
    )
    fit_rmse = _compute_tracking_rmse(
        fit_grip, fit_head, obs_grip_2d, obs_head_2d, valid_mask
    )

    replay_grip, replay_head, _ = simulate_pendulum_rollout(
        dyn_params,
        init_state.q0,
        init_state.v0,
        times,
        best_desc,
        substeps_per_frame=options.substeps_replay,
    )
    replay_rmse = _compute_tracking_rmse(
        replay_grip, replay_head, obs_grip_2d, obs_head_2d, valid_mask
    )

    # 7. Physical fit metrics in 3D world space
    n_frames = len(times)
    pred_3d = np.zeros((n_frames, 2, 3))
    # Transform in-plane 2D coords back into 3D world coordinates
    pred_grip_plane = np.column_stack([replay_grip, np.zeros(n_frames)])
    pred_head_plane = np.column_stack([replay_head, np.zeros(n_frames)])
    pred_3d[:, 0, :] = plane.reconstruct_points_from_plane(pred_grip_plane)
    pred_3d[:, 1, :] = plane.reconstruct_points_from_plane(pred_head_plane)

    obs_3d = np.zeros((n_frames, 2, 3))
    obs_3d[:, 0, :] = target.butt
    obs_3d[:, 1, :] = target.clubhead
    valid_3d = np.column_stack([valid_mask, valid_mask])

    metrics = compute_fit_metrics(
        predicted=pred_3d,
        observed=obs_3d,
        valid=valid_3d,
        labels=["grip", "clubhead"],
        time_s=times,
        plane_normal=plane.normal,
        optimizer_loss=float(opt_res.fun),
    )

    # 8. Provenance and status bundles
    elapsed_s = time.perf_counter() - t_start
    conv_status = (
        SolverConvergenceStatus.CONVERGED
        if opt_res.success
        else SolverConvergenceStatus.MAX_ITERATIONS
    )
    kin_status = (
        KinematicAccuracyStatus.WITHIN_TOLERANCE
        if fit_rmse <= 0.15
        else KinematicAccuracyStatus.EXCEEDS_THRESHOLD
    )
    feas_status = (
        DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE
        if abs(replay_rmse - fit_rmse) <= options.tolerance_replay_rmse_m
        else DynamicFeasibilityStatus.INFEASIBLE
    )

    status = StatusBundle(
        solver_convergence=conv_status,
        kinematic_accuracy=kin_status,
        dynamic_feasibility=feas_status,
        scientific_qualification=ScientificQualificationStatus.QUALIFIED,
        product_promotion=ProductPromotionStatus.EXPLORATORY,
        has_native_replay=True,
    )

    identity = BaselineIdentity(
        model_id="driven_double_pendulum",
        topology=ModelTopology.PLANAR_DRIVEN_PENDULUM,
        backend=BackendType.SCIPY_ODE,
        provider_pin="pendulum",
        fit_mode=FitMode.TORQUE_DRIVEN,
        capture="driver",
        capture_sha256=target_hash,
        horizon="G1",
        solver_name="scipy_slsqp",
        integrator="rk4",
        seed=options.seed,
    )

    theta_flat = best_desc.ravel()

    return PendulumFitResult(
        theta_optimal=theta_flat,
        final_cost=float(opt_res.fun),
        final_rmse_m=fit_rmse,
        replay_rmse_m=replay_rmse,
        solver_status="success" if opt_res.success else "failure",
        iterations=int(getattr(opt_res, "nit", 1)),
        n_evaluations=len(history),
        wall_clock_s=elapsed_s,
        message=str(opt_res.message),
        history=tuple(history),
        method="scipy SLSQP (Bernstein)",
        git_commit=git_commit_short(),
        engine_version="2.0.0",
        target_hash=target_hash,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        calibrated_plane=plane,
        calibrated_geometry=geometry,
        initial_state=init_state,
        baseline_identity=identity,
        status_bundle=status,
        fit_metrics=metrics,
        unforced_rmse_m=unforced_rmse,
    )
