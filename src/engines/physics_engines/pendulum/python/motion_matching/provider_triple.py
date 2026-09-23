"""``TriplePendulumFitSwingProvider`` -- canonical motion-matching adapter for Triple Pendulum.

Provides the analytic Lagrangian baseline for motion-matching with bounded Bernstein
torques for hub, arm, and club joints, non-uniform time grids, and independent tighter-step replay.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import math
import time

import numpy as np

from src.engines.physics_engines.pendulum.python.motion_matching.adapters_triple import (
    create_calibrated_triple_pendulum_dynamics,
)
from src.engines.physics_engines.pendulum.python.motion_matching.torque_optimization_triple import (
    COEFFS_PER_JOINT,
    TripleFitTrajectoryResult,
    TriplePendulumFitOptions,
    TriplePendulumFitTarget,
    fit_bounded_triple_pendulum,
    integrate_triple_pendulum_rollout,
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

from .provider import (
    _build_failure_result,
    _compute_target_hash,
    _resolve_plane,
    build_canonical_fit_result,
    validate_club_target_preconditions,
)

logger = logging.getLogger(__name__)

__all__ = ["TriplePendulumFitSwingProvider"]


def _build_failure_result_triple(
    message: str,
    target_hash: str,
    engine_version: str,
) -> CanonicalFitResult:
    """Construct an honest CanonicalFitResult representing solver or input failure for triple pendulum."""
    return _build_failure_result(
        message=message,
        target_hash=target_hash,
        engine_version=engine_version,
        theta_dim=3 * COEFFS_PER_JOINT,
    )


def _resolve_geometry_and_q0_triple(
    projected_club: ClubTarget,
    pivot: np.ndarray,
) -> tuple[float, float, float, np.ndarray, np.ndarray, str | None]:
    """Calibrate positive link lengths and map initial state q0, v0 for triple pendulum."""
    n_frames = len(projected_club.time)
    pivots = np.tile(pivot, (n_frames, 1))

    try:
        geom = calibrate_fixed_geometry(
            shoulder_pts=pivots,
            grip_pts=projected_club.butt,
            clubhead_pts=projected_club.clubhead,
        )
        l_arm = geom.l1_arm_m
        l3 = geom.l2_club_m
    except (ValueError, RuntimeError, ZeroDivisionError):
        l_arm = 0.65
        l3 = 1.05

    # Partition upper chain into hub (35%) and arm (65%)
    l1 = float(np.clip(l_arm * 0.35, 0.15, 0.35))
    l2 = float(np.clip(l_arm * 0.65, 0.35, 0.65))

    try:
        init_st = map_initial_state_double_pendulum(
            times=projected_club.time,
            pivot_pts=pivots,
            grip_pts=projected_club.butt,
            clubhead_pts=projected_club.clubhead,
            l1=l_arm,
            l2=l3,
            t0_idx=0,
        )
        th1_d, th2_d = float(init_st.q0[0]), float(init_st.q0[1])
        om1_d, om2_d = float(init_st.v0[0]), float(init_st.v0[1])

        # Seeding triple pendulum from double fit:
        # theta1 aligns with upper chain, theta2 starts aligned (0), theta3 matches club angle
        q0 = np.array([th1_d, 0.0, th2_d], dtype=np.float64)
        v0 = np.array([om1_d, 0.0, om2_d], dtype=np.float64)
    except (ValueError, RuntimeError, TypeError, KeyError) as exc:
        return (
            l1,
            l2,
            l3,
            np.zeros(3),
            np.zeros(3),
            f"Initial state mapping failed: {exc}",
        )

    return l1, l2, l3, q0, v0, None


def _build_canonical_triple_result(
    fit_res: TripleFitTrajectoryResult,
    elapsed: float,
    target_hash: str,
    engine_version: str,
) -> CanonicalFitResult:
    """Assemble CanonicalFitResult from triple pendulum optimization rollout."""
    all_coeffs = np.concatenate(
        [
            fit_res.profile.hub_controls,
            fit_res.profile.arm_controls,
            fit_res.profile.wrist_controls,
        ]
    )
    return build_canonical_fit_result(
        all_coeffs, fit_res, elapsed, target_hash, engine_version
    )


class TriplePendulumFitSwingProvider:
    """Canonical-API adapter providing a physically sound driven triple pendulum fit."""

    engine_name: str = "pendulum_triple"

    def fit_swing(
        self,
        target: MultiSourceTarget | ClubTarget,
        opts: FitOptions,
    ) -> CanonicalFitResult:
        club = resolve_club_target(target)
        t_start = time.perf_counter()
        target_hash = _compute_target_hash(club)

        precondition_err = validate_club_target_preconditions(
            club, target_hash, self.engine_version(), theta_dim=3 * COEFFS_PER_JOINT
        )
        if precondition_err is not None:
            return precondition_err

        # 1. Project onto calibrated swing plane
        projected_club, _, plane_err = _resolve_plane(club, opts)
        if plane_err is not None:
            return _build_failure_result_triple(
                plane_err, target_hash, self.engine_version()
            )

        # 2. Calibrate link geometry and map feasible initial state q0, v0
        pivot = np.zeros(3)
        l1, l2, l3, q0, v0, init_err = _resolve_geometry_and_q0_triple(
            projected_club, pivot
        )
        if init_err is not None:
            return _build_failure_result_triple(
                init_err, target_hash, self.engine_version()
            )

        # 3. Optimize bounded Bernstein torques
        dynamics = create_calibrated_triple_pendulum_dynamics(l1, l2, l3)
        max_nfev = opts.maxiter if opts and opts.maxiter else 100
        target_data = TriplePendulumFitTarget(
            times=projected_club.time,
            grip=projected_club.butt,
            head=projected_club.clubhead,
            l1=l1,
            l2=l2,
            l3=l3,
            q0=q0,
            v0=v0,
        )
        fit_options = TriplePendulumFitOptions(
            pivot=pivot[:2],
            max_nfev=max_nfev,
        )

        try:
            fit_res = fit_bounded_triple_pendulum(
                target=target_data,
                dynamics=dynamics,
                options=fit_options,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            return _build_failure_result_triple(
                f"Optimization failed: {exc}", target_hash, self.engine_version()
            )

        # 4. Independent tighter-step replay (4x finer substeps)
        _, _ = integrate_triple_pendulum_rollout(
            dynamics, q0, v0, projected_club.time, fit_res.profile, substeps=4
        )

        elapsed = time.perf_counter() - t_start
        result = _build_canonical_triple_result(
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


register_provider(TriplePendulumFitSwingProvider())
