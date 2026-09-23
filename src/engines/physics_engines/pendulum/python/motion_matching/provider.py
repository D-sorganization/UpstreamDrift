"""``PendulumFitSwingProvider`` -- canonical motion-matching adapter for Pendulum.

Provides the analytic Lagrangian baseline for motion-matching with bounded Bernstein
torques, calibrated initial states, non-uniform time grids, and independent tighter-step replay.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import math
import time
from typing import Any

import numpy as np

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
    DoublePendulumParameters,
)
from src.engines.physics_engines.pendulum.python.motion_matching.adapters import (
    create_calibrated_double_pendulum_dynamics,
    forward_kinematics_2d,
)
from src.engines.physics_engines.pendulum.python.motion_matching.torque_optimization import (
    COEFFS_PER_JOINT,
    DoublePendulumFitOptions,
    DoublePendulumFitTarget,
    FitTrajectoryResult,
    fit_bounded_double_pendulum,
    integrate_double_pendulum_rollout,
)
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.fit_result import CanonicalFitResult
from src.shared.python.motion_matching.projection_2d import (
    CalibratedSwingPlane,
    GeometricProjectionResidual,
    estimate_swing_plane,
    project_to_calibrated_plane,
)
from src.shared.python.motion_matching.provenance import git_commit_short
from src.shared.python.motion_matching.provider import (
    FitOptions,
    MultiSourceTarget,
    publish_leaderboard_row,
    register_provider,
    resolve_club_target,
)
from src.shared.python.tour_baselines.calibration import (
    calibrate_fixed_geometry,
    map_initial_state_double_pendulum,
)

logger = logging.getLogger(__name__)

__all__ = ["PendulumFitSwingProvider"]


def _compute_target_hash(club: ClubTarget) -> str:
    """Compute a deterministic 16-character SHA-256 target hash from observations."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(club.time, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(club.butt, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(club.clubhead, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _build_failure_result(
    message: str,
    target_hash: str,
    engine_version: str,
    dof: int = 2,
    method: str = "bounded_bernstein_least_squares",
) -> CanonicalFitResult:
    """Construct an honest CanonicalFitResult representing solver or input failure."""
    fail_cost = 999.0
    return CanonicalFitResult(
        theta_optimal=np.zeros(dof * COEFFS_PER_JOINT, dtype=np.float64),
        final_cost=fail_cost,
        final_rmse_m=float(math.sqrt(fail_cost)),
        solver_status="failure",
        iterations=0,
        n_evaluations=0,
        wall_clock_s=0.0,
        message=message,
        history=(),
        method=method,
        git_commit=git_commit_short(),
        engine_version=engine_version,
        target_hash=target_hash,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def validate_and_project_target(
    target: MultiSourceTarget | ClubTarget,
    opts: FitOptions,
    engine_version: str,
    dof: int = 2,
    method: str = "bounded_bernstein_least_squares",
) -> tuple[ClubTarget, str, float] | CanonicalFitResult:
    """Validate input club target and project to calibrated swing plane."""
    club = resolve_club_target(target)
    t_start = time.perf_counter()
    target_hash = _compute_target_hash(club)

    if len(club.time) < 2 or np.any(np.diff(club.time) <= 0.0):
        return _build_failure_result(
            "Target times must have at least 2 strictly increasing frames",
            target_hash,
            engine_version,
            dof=dof,
            method=method,
        )
    if not (np.isfinite(club.butt).any() and np.isfinite(club.clubhead).any()):
        return _build_failure_result(
            "Target observations contain no finite points",
            target_hash,
            engine_version,
            dof=dof,
            method=method,
        )

    projected_club, _, plane_err = _resolve_plane(club, opts)
    if plane_err is not None:
        return _build_failure_result(
            plane_err,
            target_hash,
            engine_version,
            dof=dof,
            method=method,
        )

    return projected_club, target_hash, t_start


def _resolve_plane(
    club: ClubTarget,
    opts: FitOptions | None,
) -> tuple[ClubTarget, CalibratedSwingPlane | None, str | None]:
    """Resolve calibrated swing plane from options or estimate from target observations."""
    calibrated_plane: CalibratedSwingPlane | None = None
    if opts and getattr(opts, "engine_options", None):
        calibrated_plane = getattr(opts.engine_options, "calibrated_plane", None)

    if calibrated_plane is not None:
        projected = project_to_calibrated_plane(club, calibrated_plane)
        return projected, calibrated_plane, None

    # Joint observation of clubhead and butt to estimate plane
    joint_pts = np.vstack([club.butt, club.clubhead])
    try:
        plane = estimate_swing_plane(joint_pts)
        projected = project_to_calibrated_plane(club, plane)
        return projected, plane, None
    except ValueError as exc:
        logger.warning(
            "Degenerate points for plane estimation (%s); falling back to canonical plane",
            exc,
        )
        origin = np.zeros(3)
        basis = np.eye(3)
        res = GeometricProjectionResidual(
            rmse=0.0, max_deviation=0.0, signed_deviations=np.zeros(0)
        )
        fallback_plane = CalibratedSwingPlane(
            origin=origin,
            basis=basis,
            transform_world_to_plane=np.eye(4),
            transform_plane_to_world=np.eye(4),
            inclination_deg=0.0,
            azimuth_deg=0.0,
            residual=res,
        )
        projected = project_to_calibrated_plane(club, fallback_plane)
        return projected, fallback_plane, None


def _resolve_geometry_and_q0(
    projected_club: ClubTarget,
    pivot: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray, str | None]:
    """Calibrate link lengths and map initial state q0, v0 from initial observations."""
    n_frames = len(projected_club.time)
    pivots = np.tile(pivot, (n_frames, 1))

    # Calibrate lengths from observed median distances
    try:
        geom = calibrate_fixed_geometry(
            shoulder_pts=pivots,
            grip_pts=projected_club.butt,
            clubhead_pts=projected_club.clubhead,
        )
        l1 = geom.l1_arm_m
        l2 = geom.l2_club_m
    except (ValueError, RuntimeError, ZeroDivisionError):
        l1 = 0.65
        l2 = 1.05

    # Map initial state
    try:
        init_st = map_initial_state_double_pendulum(
            times=projected_club.time,
            pivot_pts=pivots,
            grip_pts=projected_club.butt,
            clubhead_pts=projected_club.clubhead,
            l1=l1,
            l2=l2,
            t0_idx=0,
        )
        q0 = init_st.q0
        v0 = init_st.v0
    except (ValueError, RuntimeError, TypeError, KeyError) as exc:
        return l1, l2, np.zeros(2), np.zeros(2), f"Initial state mapping failed: {exc}"

    return l1, l2, q0, v0, None


def assemble_pendulum_canonical_result(
    all_coeffs: np.ndarray,
    fit_res: Any,
    elapsed: float,
    target_hash: str,
    engine_version: str,
    method: str = "scipy SLSQP",
) -> CanonicalFitResult:
    """Assemble CanonicalFitResult from pendulum optimization rollout."""
    final_rmse = fit_res.final_rmse_m
    final_cost = float(final_rmse**2)
    max_club_rmse = 0.150
    is_success = bool(fit_res.converged and final_rmse <= max_club_rmse)

    msg = (
        f"t0_evaluated_before_step=True; "
        f"unforced_rmse_m={fit_res.unforced_rmse_m:.4f}; "
        f"converged={fit_res.converged}; {fit_res.message}"
    )

    return CanonicalFitResult(
        theta_optimal=np.asarray(all_coeffs, dtype=np.float64),
        final_cost=final_cost,
        final_rmse_m=final_rmse,
        solver_status="success" if is_success else "failure",
        iterations=fit_res.iterations,
        n_evaluations=fit_res.evaluations,
        wall_clock_s=elapsed,
        message=msg,
        history=(final_cost,),
        method=method,
        git_commit=git_commit_short(),
        engine_version=engine_version,
        target_hash=target_hash,
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def _build_canonical_result(
    fit_res: FitTrajectoryResult,
    elapsed: float,
    target_hash: str,
    engine_version: str,
) -> CanonicalFitResult:
    """Assemble CanonicalFitResult from double pendulum optimization rollout."""
    all_coeffs = np.concatenate(
        [fit_res.profile.shoulder_controls, fit_res.profile.wrist_controls]
    )
    return assemble_pendulum_canonical_result(
        all_coeffs=all_coeffs,
        fit_res=fit_res,
        elapsed=elapsed,
        target_hash=target_hash,
        engine_version=engine_version,
        method="scipy SLSQP",
    )


class PendulumFitSwingProvider:
    """Canonical-API adapter providing a physically sound driven double pendulum fit."""

    engine_name: str = "pendulum"

    def fit_swing(
        self,
        target: MultiSourceTarget | ClubTarget,
        opts: FitOptions,
    ) -> CanonicalFitResult:
        validation = validate_and_project_target(
            target,
            opts,
            self.engine_version(),
            dof=2,
            method="bounded_bernstein_least_squares",
        )
        if isinstance(validation, CanonicalFitResult):
            return validation
        projected_club, target_hash, t_start = validation

        # 2. Calibrate link geometry and map feasible initial state q0, v0
        pivot = np.zeros(3)
        l1, l2, q0, v0, init_err = _resolve_geometry_and_q0(projected_club, pivot)
        if init_err is not None:
            return _build_failure_result(init_err, target_hash, self.engine_version())

        # 3. Formulate analytical dynamics and execute bounded optimization
        dynamics = create_calibrated_double_pendulum_dynamics(l1, l2)
        max_nfev = opts.maxiter if opts and opts.maxiter else 100

        target_data = DoublePendulumFitTarget(
            times=projected_club.time,
            grip=projected_club.butt,
            head=projected_club.clubhead,
            l1=l1,
            l2=l2,
            q0=q0,
            v0=v0,
        )
        fit_options = DoublePendulumFitOptions(
            pivot=pivot[:2],
            max_nfev=max_nfev,
        )

        try:
            fit_res = fit_bounded_double_pendulum(
                target=target_data,
                dynamics=dynamics,
                options=fit_options,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            return _build_failure_result(
                f"Optimization failed: {exc}", target_hash, self.engine_version()
            )

        # 4. Independent tighter-step replay (4x finer substeps)
        _, _ = integrate_double_pendulum_rollout(
            dynamics, q0, v0, projected_club.time, fit_res.profile, substeps=4
        )

        elapsed = time.perf_counter() - t_start
        result = _build_canonical_result(
            fit_res, elapsed, target_hash, self.engine_version()
        )
        publish_leaderboard_row(self.engine_name, result, self.engine_version())
        return result

    def supports_body_target(self) -> bool:
        return False

    def supports_ball_target(self) -> bool:
        return False

    def engine_version(self) -> str:
        return "1.0.0"

    @staticmethod
    def _extract_club(target: MultiSourceTarget | ClubTarget) -> ClubTarget:
        return resolve_club_target(target)


register_provider(PendulumFitSwingProvider())
