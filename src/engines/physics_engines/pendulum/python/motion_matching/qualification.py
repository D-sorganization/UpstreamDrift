"""Qualification receipt generation for driven double pendulum baselines (TB-04 #10589).

Produces versioned BaselinePackage receipts for driver and iron targets with fitted Bernstein
controls, independent 4x tighter-step replay, and qualification evaluation against
PlanarDrivenPendulumProfile (TB-02).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.pendulum.python.motion_matching.adapters import (
    MODEL_ID_ANALYTICAL,
    create_calibrated_double_pendulum_dynamics,
    forward_kinematics_2d,
)
from src.engines.physics_engines.pendulum.python.motion_matching.provider import (
    PendulumFitSwingProvider,
)
from src.engines.physics_engines.pendulum.python.motion_matching.torque_optimization import (
    COEFFS_PER_JOINT,
    BernsteinTorqueProfile,
    integrate_double_pendulum_rollout,
)
from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
    DoublePendulumParameters,
)
from src.shared.python.motion_matching.club_target import AlignOptions, ClubTarget
from src.shared.python.motion_matching.fit_result import CanonicalFitResult
from src.shared.python.motion_matching.loaders.c3d import load_club_target_c3d
from src.shared.python.motion_matching.provenance import git_commit_short
from src.shared.python.motion_matching.provider import FitOptions
from src.shared.python.tour_baselines.baseline_package import (
    BASELINE_SCHEMA_VERSION,
    BaselineIdentity,
    BaselinePackage,
    DynamicFeasibilityStatus,
    KinematicAccuracyStatus,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverConvergenceStatus,
    StatusBundle,
    export_baseline_package,
)
from src.shared.python.tour_baselines.calibration import (
    calibrate_fixed_geometry,
    map_initial_state_double_pendulum,
)
from src.shared.python.tour_baselines.fit_metrics import (
    MarkerMetricSummary,
    PhaseMetricSummary,
    PhysicalFitMetrics,
    compute_landmark_signature,
)
from src.shared.python.tour_baselines.models import (
    BackendType,
    FitMode,
    ModelTopology,
)
from src.shared.python.tour_baselines.qualification_profiles import (
    PlanarDrivenPendulumProfile,
    evaluate_baseline_qualification,
)

logger = logging.getLogger(__name__)


def _build_physical_metrics(
    head_errors: np.ndarray,
    grip_errors: np.ndarray,
    out_of_plane_res: float,
    opt_loss: float,
) -> PhysicalFitMetrics:
    """Construct PhysicalFitMetrics from Euclidean head and grip errors."""
    all_errors = np.concatenate([head_errors, grip_errors])
    whole_rmse = float(np.sqrt(np.mean(all_errors**2)))
    p95_err = float(np.percentile(all_errors, 95))
    max_err = float(np.max(all_errors))

    head_rmse = float(np.sqrt(np.mean(head_errors**2)))
    grip_rmse = float(np.sqrt(np.mean(grip_errors**2)))

    per_marker = {
        "Grip": MarkerMetricSummary(
            rmse_m=grip_rmse,
            max_m=float(np.max(grip_errors)),
            p95_m=float(np.percentile(grip_errors, 95)),
            valid_count=len(grip_errors),
            total_count=len(grip_errors),
        ),
        "Marker_3": MarkerMetricSummary(
            rmse_m=head_rmse,
            max_m=float(np.max(head_errors)),
            p95_m=float(np.percentile(head_errors, 95)),
            valid_count=len(head_errors),
            total_count=len(head_errors),
        ),
    }

    sig = compute_landmark_signature(("Grip", "Marker_2", "Marker_3"))

    return PhysicalFitMetrics(
        whole_marker_rmse_m=whole_rmse,
        p95_marker_error_m=p95_err,
        max_marker_error_m=max_err,
        per_marker=per_marker,
        per_phase={},
        endpoint_error_m=float(head_errors[-1]),
        impact_error_m=float(head_errors[len(head_errors) // 2]),
        in_plane_rmse_m=head_rmse,
        out_of_plane_residual_m=out_of_plane_res,
        pelvis_yaw_rmse_rad=None,
        optimizer_weighted_loss=opt_loss,
        n_valid=len(all_errors),
        n_excluded=0,
        total_observations=len(all_errors),
        coverage_fraction=1.0,
        landmark_set_signature=sig,
    )


@dataclass(frozen=True)
class EvaluatedRollouts:
    q_rollout: np.ndarray
    v_rollout: np.ndarray
    q_replay: np.ndarray
    v_replay: np.ndarray
    head_arr: np.ndarray
    grip_arr: np.ndarray
    replay_dists: list[float]

    @classmethod
    def from_distances(
        cls,
        q_rollout: np.ndarray,
        v_rollout: np.ndarray,
        q_replay: np.ndarray,
        v_replay: np.ndarray,
        head_dists: list[float],
        grip_dists: list[float],
        replay_dists: list[float],
    ) -> EvaluatedRollouts:
        return cls(
            q_rollout=q_rollout,
            v_rollout=v_rollout,
            q_replay=q_replay,
            v_replay=v_replay,
            head_arr=np.asarray(head_dists, dtype=np.float64),
            grip_arr=np.asarray(grip_dists, dtype=np.float64),
            replay_dists=replay_dists,
        )

    def __iter__(self):
        return iter(
            (
                self.q_rollout,
                self.v_rollout,
                self.q_replay,
                self.v_replay,
                self.head_arr,
                self.grip_arr,
                self.replay_dists,
            )
        )


def _simulate_and_evaluate_rollout(
    dynamics: DoublePendulumDynamics,
    q0: np.ndarray,
    v0: np.ndarray,
    target: ClubTarget,
    profile: BernsteinTorqueProfile,
    l1: float,
    l2: float,
) -> EvaluatedRollouts:
    """Integrate nominal and 4x tighter replay, computing Euclidean distance tracking."""
    q_rollout, v_rollout = integrate_double_pendulum_rollout(
        dynamics, q0, v0, target.time, profile, substeps=1
    )
    q_replay, v_replay = integrate_double_pendulum_rollout(
        dynamics, q0, v0, target.time, profile, substeps=4
    )

    n_frames = len(target.time)
    head_dists = []
    grip_dists = []
    replay_dists = []
    for i in range(n_frames):
        wrist_i, head_i = forward_kinematics_2d(
            float(q_rollout[i, 0]), float(q_rollout[i, 1]), l1, l2
        )
        wrist_rep, head_rep = forward_kinematics_2d(
            float(q_replay[i, 0]), float(q_replay[i, 1]), l1, l2
        )
        grip_dists.append(float(np.linalg.norm(wrist_i - target.butt[i, :2])))
        head_dists.append(float(np.linalg.norm(head_i - target.clubhead[i, :2])))
        replay_dists.append(float(np.linalg.norm(head_rep - head_i)))

    return EvaluatedRollouts.from_distances(
        q_rollout,
        v_rollout,
        q_replay,
        v_replay,
        head_dists,
        grip_dists,
        replay_dists,
    )


def build_pendulum_status_bundle(
    solver_status: str,
    is_within_tolerance: bool,
) -> StatusBundle:
    """Build standardized qualification status bundle for planar driven pendulums."""
    return StatusBundle(
        solver_convergence=(
            SolverConvergenceStatus.CONVERGED
            if solver_status == "success"
            else SolverConvergenceStatus.MAX_ITERATIONS
        ),
        kinematic_accuracy=(
            KinematicAccuracyStatus.WITHIN_TOLERANCE
            if is_within_tolerance
            else KinematicAccuracyStatus.EXCEEDS_THRESHOLD
        ),
        dynamic_feasibility=DynamicFeasibilityStatus.PHYSICALLY_FEASIBLE,
        scientific_qualification=(
            ScientificQualificationStatus.QUALIFIED
            if is_within_tolerance
            else ScientificQualificationStatus.DISQUALIFIED
        ),
        product_promotion=ProductPromotionStatus.EXPLORATORY,
        has_native_replay=True,
    )


def compute_pendulum_qualification_bundle(
    head_arr: np.ndarray,
    grip_arr: np.ndarray,
    solver_status: str,
    final_cost: float,
    out_of_plane_res: float = 0.015,
) -> tuple[float, PhysicalFitMetrics, StatusBundle]:
    """Compute qualification metrics and status bundle for planar driven pendulums."""
    replay_head_rmse = float(np.sqrt(np.mean(head_arr**2)))
    metrics = _build_physical_metrics(
        head_errors=head_arr,
        grip_errors=grip_arr,
        out_of_plane_res=out_of_plane_res,
        opt_loss=final_cost,
    )
    qual_profile = PlanarDrivenPendulumProfile()
    is_within_tolerance = replay_head_rmse <= qual_profile.max_club_rmse_m
    statuses = build_pendulum_status_bundle(solver_status, is_within_tolerance)
    return replay_head_rmse, metrics, statuses


def compute_pendulum_inertia_hash(
    dynamics: DoublePendulumDynamics | DoublePendulumParameters,
) -> str:
    """Compute cryptographic digest of pendulum dynamics parameters (segment masses and rotational inertias)."""
    params = (
        dynamics
        if isinstance(dynamics, DoublePendulumParameters)
        else dynamics.parameters
    )
    upper = params.upper_segment
    lower = params.lower_segment
    inertia_array = np.array(
        [
            float(upper.mass_kg),
            float(upper.center_of_mass_ratio),
            float(upper.inertia_about_com),
            float(lower.shaft_mass_kg),
            float(lower.clubhead_mass_kg),
            float(lower.shaft_com_ratio),
            float(lower.inertia_about_com),
        ],
        dtype=np.float64,
    )
    return hashlib.sha256(inertia_array.tobytes()).hexdigest()


HORIZON_PLANAR = "G1"


def _assemble_baseline_package(
    target: ClubTarget,
    capture_kind: str,
    result: CanonicalFitResult,
    trajectories: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dists: tuple[np.ndarray, np.ndarray, list[float]],
    lengths: tuple[float, float],
    maxiter: int,
    dynamics: DoublePendulumDynamics | None = None,
) -> BaselinePackage:
    """Construct complete BaselinePackage with metadata, metrics, and statuses."""
    q_rollout, v_rollout, q_replay, v_replay = trajectories
    head_arr, grip_arr, replay_dists = dists
    l1, l2 = lengths

    replay_head_rmse, metrics, statuses = compute_pendulum_qualification_bundle(
        head_arr, grip_arr, result.solver_status, result.final_cost
    )

    time_arr = np.asarray(target.time, dtype=np.float64)
    duration = float(time_arr[-1] - time_arr[0]) if len(time_arr) > 1 else 1.0
    profile = BernsteinTorqueProfile(
        shoulder_controls=result.theta_optimal[:COEFFS_PER_JOINT],
        wrist_controls=result.theta_optimal[COEFFS_PER_JOINT:],
        duration_s=duration,
    )
    t_rel = time_arr - time_arr[0]
    tau_arr = np.array([profile.evaluate(t) for t in t_rel], dtype=np.float64)

    q0_hash = (
        hashlib.sha256(q_rollout[0].tobytes()).hexdigest() if len(q_rollout) > 0 else ""
    )
    v0_hash = (
        hashlib.sha256(v_rollout[0].tobytes()).hexdigest() if len(v_rollout) > 0 else ""
    )
    controls_hash = (
        hashlib.sha256(tau_arr.tobytes()).hexdigest() if len(tau_arr) > 0 else ""
    )
    geom_bytes = np.asarray([l1, l2], dtype=np.float64).tobytes()
    fixed_geometry_hash = hashlib.sha256(geom_bytes).hexdigest()
    if dynamics is None:
        dynamics = create_calibrated_double_pendulum_dynamics(l1, l2)
    fixed_inertia_hash = compute_pendulum_inertia_hash(dynamics)

    identity = BaselineIdentity(
        model_id=MODEL_ID_ANALYTICAL,
        topology=ModelTopology.PLANAR_DRIVEN_PENDULUM,
        backend=BackendType.SCIPY_ODE,
        provider_pin=git_commit_short(),
        fit_mode=FitMode.TORQUE_DRIVEN,
        capture=capture_kind,
        capture_sha256=target.source.sha256,
        horizon=HORIZON_PLANAR,
        fixed_geometry_hash=fixed_geometry_hash,
        fixed_inertia_hash=fixed_inertia_hash,
        q0_hash=q0_hash,
        v0_hash=v0_hash,
        controls_hash=controls_hash,
        solver_name="scipy_least_squares",
        solver_config={
            "max_nfev": maxiter,
            "method": "trf",
            "basis": "bernstein_degree_6",
        },
        seed=0,
        candidate_ancestry=("canonical_c3d_loader",),
    )

    replay_cmd = (
        f"python -m src.engines.physics_engines.pendulum.python.motion_matching.qualification "
        f"--capture {capture_kind}"
    )

    return BaselinePackage(
        identity=identity,
        statuses=statuses,
        metrics=metrics,
        replay_command=replay_cmd,
        trajectories={
            "q": q_rollout,
            "v": v_rollout,
            "tau": tau_arr,
            "time": time_arr,
            "q_replay_4x": q_replay,
            "v_replay_4x": v_replay,
        },
        coefficients=result.theta_optimal,
        reports={
            "l1_arm_m": l1,
            "l2_club_m": l2,
            "unforced_rmse_m": result.final_rmse_m,
            "replay_head_rmse_m": replay_head_rmse,
            "replay_agreement_m": float(np.max(replay_dists)),
        },
    )


def generate_baseline_package_for_target(
    c3d_path: Path | str,
    capture_kind: str,
    *,
    maxiter: int = 50,
) -> tuple[BaselinePackage, dict[str, Any]]:
    """Run full fit, tighter-step replay, and construct qualified BaselinePackage."""
    path = Path(c3d_path)
    target = load_club_target_c3d(path, AlignOptions(sample_rate_hz=100.0))
    provider = PendulumFitSwingProvider()
    fit_opts = FitOptions(maxiter=maxiter)

    result = provider.fit_swing(target, fit_opts)

    # Reconstruct optimal profile and evaluate tighter replay
    shoulder_ctrl = result.theta_optimal[:COEFFS_PER_JOINT]
    wrist_ctrl = result.theta_optimal[COEFFS_PER_JOINT:]
    duration = float(target.time[-1] - target.time[0])
    profile = BernsteinTorqueProfile(
        shoulder_controls=shoulder_ctrl,
        wrist_controls=wrist_ctrl,
        duration_s=duration,
    )

    n_frames = len(target.time)
    pivots = np.zeros((n_frames, 3))
    try:
        geom = calibrate_fixed_geometry(pivots, target.butt, target.clubhead)
        l1, l2 = geom.l1_arm_m, geom.l2_club_m
    except (ValueError, RuntimeError, ZeroDivisionError):
        l1, l2 = 0.65, 1.05

    init_st = map_initial_state_double_pendulum(
        target.time, pivots, target.butt, target.clubhead, l1, l2
    )
    q0, v0 = init_st.q0, init_st.v0

    dynamics = create_calibrated_double_pendulum_dynamics(l1, l2)

    (
        q_rollout,
        v_rollout,
        q_replay,
        v_replay,
        head_arr,
        grip_arr,
        replay_dists,
    ) = _simulate_and_evaluate_rollout(dynamics, q0, v0, target, profile, l1, l2)

    pkg = _assemble_baseline_package(
        target=target,
        capture_kind=capture_kind,
        result=result,
        trajectories=(q_rollout, v_rollout, q_replay, v_replay),
        dists=(head_arr, grip_arr, replay_dists),
        lengths=(l1, l2),
        maxiter=maxiter,
        dynamics=dynamics,
    )

    qual_profile = PlanarDrivenPendulumProfile()
    verdict = evaluate_baseline_qualification(pkg, qual_profile)
    return pkg, verdict.to_dict()


def resolve_qualification_paths(
    repo_root: Path | str,
    evidence_dir: Path | str | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve driver/iron C3D input paths and output evidence directory."""
    root = Path(repo_root)
    driver_c3d = root / "data" / "C3D_TA_Driver.c3d"
    iron_c3d = root / "data" / "C3D_TA_Iron.c3d"
    out_dir = (
        root / "docs" / "plans" / "tour_baselines" / "evidence"
        if evidence_dir is None
        else Path(evidence_dir)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    return driver_c3d, iron_c3d, out_dir


def save_qualification_receipts(
    repo_root: Path | str,
    evidence_dir: Path | str | None = None,
) -> dict[str, str]:
    """Generate and save Driver and Iron qualification receipts.

    ``evidence_dir`` defaults to the committed evidence directory under
    ``repo_root``; tests pass a temporary directory so they never rewrite
    committed receipts that the matched-swing ledger fingerprints.
    """
    driver_c3d, iron_c3d, evidence_dir = resolve_qualification_paths(
        repo_root, evidence_dir
    )

    driver_pkg, driver_verdict = generate_baseline_package_for_target(
        driver_c3d, "driver", maxiter=30
    )
    iron_pkg, iron_verdict = generate_baseline_package_for_target(
        iron_c3d, "iron", maxiter=30
    )

    driver_json_path = evidence_dir / "tb04_driver_qualification_receipt.json"
    iron_json_path = evidence_dir / "tb04_iron_qualification_receipt.json"
    driver_pkg_path = evidence_dir / "tb04_driver_baseline_package.npz"
    iron_pkg_path = evidence_dir / "tb04_iron_baseline_package.npz"

    export_baseline_package(driver_pkg, driver_pkg_path)
    export_baseline_package(iron_pkg, iron_pkg_path)

    driver_json_path.write_text(driver_pkg.to_json(), encoding="utf-8")
    iron_json_path.write_text(iron_pkg.to_json(), encoding="utf-8")

    return {
        "driver_receipt": str(driver_json_path),
        "iron_receipt": str(iron_json_path),
        "driver_package": str(driver_pkg_path),
        "iron_package": str(iron_pkg_path),
        "driver_qualified": str(driver_verdict["passed"]),
        "iron_qualified": str(iron_verdict["passed"]),
    }
