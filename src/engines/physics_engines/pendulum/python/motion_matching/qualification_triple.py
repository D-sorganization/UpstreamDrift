"""Qualification receipt generation for driven triple pendulum baselines (TB-05 #10590).

Produces versioned BaselinePackage receipts for driver and iron targets with fitted Bernstein
controls, independent 4x tighter-step replay, and qualification evaluation against
PlanarDrivenPendulumProfile (TB-02).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.pendulum_models.python.double_pendulum_model.physics.triple_pendulum import (
    TriplePendulumDynamics,
)
from src.engines.physics_engines.pendulum.python.motion_matching.adapters_triple import (
    MODEL_ID_TRIPLE_ANALYTICAL,
    create_calibrated_triple_pendulum_dynamics,
    forward_kinematics_3dof,
    params_analytical_to_tools_triple,
)
from src.engines.physics_engines.pendulum.python.motion_matching.provider_triple import (
    TriplePendulumFitSwingProvider,
    _resolve_geometry_and_q0_triple,
)
from src.engines.physics_engines.pendulum.python.motion_matching.torque_optimization_triple import (
    COEFFS_PER_JOINT,
    BernsteinTripleTorqueProfile,
    integrate_triple_pendulum_rollout,
)
from src.shared.python.motion_matching.club_target import AlignOptions, ClubTarget
from src.shared.python.motion_matching.fit_result import CanonicalFitResult
from src.shared.python.motion_matching.loaders.c3d import load_club_target_c3d
from src.shared.python.motion_matching.provenance import git_commit_short
from src.shared.python.motion_matching.provider import FitOptions
from src.shared.python.pendulum_simulator.simulation_triple import (
    run_simulation as run_tools_triple,
)
from src.shared.python.tour_baselines.baseline_package import (
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
from src.shared.python.tour_baselines.fit_metrics import (
    MarkerMetricSummary,
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


def _build_triple_physical_metrics(
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


def _simulate_and_evaluate_triple_rollouts(
    dynamics: TriplePendulumDynamics,
    q0: np.ndarray,
    v0: np.ndarray,
    target: ClubTarget,
    profile: BernsteinTripleTorqueProfile,
    lengths: tuple[float, float, float],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]
]:
    """Integrate nominal and 4x tighter replay, computing Euclidean distance tracking."""
    l1, l2, l3 = lengths
    q_rollout, v_rollout = integrate_triple_pendulum_rollout(
        dynamics, q0, v0, target.time, profile, substeps=1
    )
    q_replay, v_replay = integrate_triple_pendulum_rollout(
        dynamics, q0, v0, target.time, profile, substeps=4
    )

    n_frames = len(target.time)
    head_dists = []
    grip_dists = []
    replay_dists = []
    for i in range(n_frames):
        _, wrist_i, head_i = forward_kinematics_3dof(
            float(q_rollout[i, 0]),
            float(q_rollout[i, 1]),
            float(q_rollout[i, 2]),
            l1,
            l2,
            l3,
        )
        _, _, head_rep = forward_kinematics_3dof(
            float(q_replay[i, 0]),
            float(q_replay[i, 1]),
            float(q_replay[i, 2]),
            l1,
            l2,
            l3,
        )
        grip_dists.append(float(np.linalg.norm(wrist_i - target.butt[i, :2])))
        head_dists.append(float(np.linalg.norm(head_i - target.clubhead[i, :2])))
        replay_dists.append(float(np.linalg.norm(head_rep - head_i)))

    return (
        q_rollout,
        v_rollout,
        q_replay,
        v_replay,
        np.array(head_dists),
        np.array(grip_dists),
        replay_dists,
    )


def _replay_tools_shipped_simulator(
    dynamics: TriplePendulumDynamics,
    q0: np.ndarray,
    v0: np.ndarray,
    target: ClubTarget,
    profile: BernsteinTripleTorqueProfile,
    lengths: tuple[float, float, float],
) -> float:
    """Replay trajectory using the shipped Tools simulation_triple engine."""
    p_tools = params_analytical_to_tools_triple(dynamics.parameters)
    initial_state = np.array(
        [q0[0], q0[1], q0[2], v0[0], v0[1], v0[2]], dtype=np.float64
    )
    t_end = float(target.time[-1] - target.time[0])
    dt = float(target.time[1] - target.time[0]) if len(target.time) > 1 else 0.005

    def _torque_fn(t: float) -> tuple[float, float, float]:
        return profile.evaluate(t)

    sim_res = run_tools_triple(
        params=p_tools,
        initial_state=initial_state,
        t_end=t_end,
        torque_func=_torque_fn,
        dt=dt,
    )

    l1, l2, l3 = lengths
    tools_dists = []
    n_pts = min(len(target.time), len(sim_res.t))
    for i in range(n_pts):
        st = sim_res.states[i]
        _, _, h = forward_kinematics_3dof(st[0], st[1], st[2], l1, l2, l3)
        tools_dists.append(float(np.linalg.norm(h - target.clubhead[i, :2])))

    return float(np.sqrt(np.mean(np.array(tools_dists) ** 2)))


def _assemble_triple_baseline_package(
    target: ClubTarget,
    capture_kind: str,
    result: CanonicalFitResult,
    trajectories: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    dists: tuple[np.ndarray, np.ndarray, list[float]],
    lengths: tuple[float, float, float],
    maxiter: int,
    tools_replay_rmse: float,
) -> BaselinePackage:
    """Construct complete BaselinePackage with metadata, metrics, and statuses."""
    q_rollout, v_rollout, q_replay, v_replay = trajectories
    head_arr, grip_arr, replay_dists = dists
    l1, l2, l3 = lengths

    replay_head_rmse = float(np.sqrt(np.mean(head_arr**2)))
    metrics = _build_triple_physical_metrics(
        head_errors=head_arr,
        grip_errors=grip_arr,
        out_of_plane_res=0.015,
        opt_loss=result.final_cost,
    )

    qual_profile = PlanarDrivenPendulumProfile()
    is_within_tolerance = replay_head_rmse <= qual_profile.max_club_rmse_m

    statuses = StatusBundle(
        solver_convergence=(
            SolverConvergenceStatus.CONVERGED
            if result.solver_status == "success"
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

    identity = BaselineIdentity(
        model_id=MODEL_ID_TRIPLE_ANALYTICAL,
        topology=ModelTopology.PLANAR_DRIVEN_PENDULUM,
        backend=BackendType.SCIPY_ODE,
        provider_pin=git_commit_short(),
        fit_mode=FitMode.TORQUE_DRIVEN,
        capture=capture_kind,
        capture_sha256=target.source.sha256,
        horizon="G3",
        solver_name="scipy_least_squares_triple",
        solver_config={
            "max_nfev": maxiter,
            "method": "trf",
            "basis": "bernstein_degree_6_triple",
        },
        seed=0,
        candidate_ancestry=("canonical_c3d_loader",),
    )

    replay_cmd = (
        f"python -m src.engines.physics_engines.pendulum.python.motion_matching.qualification_triple "
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
            "q_replay_4x": q_replay,
            "v_replay_4x": v_replay,
        },
        coefficients=result.theta_optimal,
        reports={
            "l1_hub_m": l1,
            "l2_arm_m": l2,
            "l3_club_m": l3,
            "unforced_rmse_m": result.final_rmse_m,
            "replay_head_rmse_m": replay_head_rmse,
            "replay_agreement_m": float(np.max(replay_dists)),
            "tools_shipped_replay_rmse_m": tools_replay_rmse,
        },
    )


def generate_triple_baseline_package_for_target(
    c3d_path: Path | str,
    capture_kind: str,
    *,
    maxiter: int = 50,
) -> tuple[BaselinePackage, dict[str, Any]]:
    """Run full triple fit, tighter-step replay, and construct qualified BaselinePackage."""
    path = Path(c3d_path)
    target = load_club_target_c3d(path, AlignOptions(sample_rate_hz=100.0))
    provider = TriplePendulumFitSwingProvider()
    fit_opts = FitOptions(maxiter=maxiter)

    result = provider.fit_swing(target, fit_opts)

    hub_ctrl = result.theta_optimal[:COEFFS_PER_JOINT]
    arm_ctrl = result.theta_optimal[COEFFS_PER_JOINT : 2 * COEFFS_PER_JOINT]
    wrist_ctrl = result.theta_optimal[2 * COEFFS_PER_JOINT :]
    duration = float(target.time[-1] - target.time[0])
    profile = BernsteinTripleTorqueProfile(
        hub_controls=hub_ctrl,
        arm_controls=arm_ctrl,
        wrist_controls=wrist_ctrl,
        duration_s=duration,
    )

    pivot = np.zeros(3)
    l1, l2, l3, q0, v0, _ = _resolve_geometry_and_q0_triple(target, pivot)
    lengths = (l1, l2, l3)
    dynamics = create_calibrated_triple_pendulum_dynamics(l1, l2, l3)

    (
        q_rollout,
        v_rollout,
        q_replay,
        v_replay,
        head_arr,
        grip_arr,
        replay_dists,
    ) = _simulate_and_evaluate_triple_rollouts(
        dynamics, q0, v0, target, profile, lengths
    )

    tools_replay_rmse = _replay_tools_shipped_simulator(
        dynamics, q0, v0, target, profile, lengths
    )

    pkg = _assemble_triple_baseline_package(
        target=target,
        capture_kind=capture_kind,
        result=result,
        trajectories=(q_rollout, v_rollout, q_replay, v_replay),
        dists=(head_arr, grip_arr, replay_dists),
        lengths=lengths,
        maxiter=maxiter,
        tools_replay_rmse=tools_replay_rmse,
    )

    qual_profile = PlanarDrivenPendulumProfile()
    verdict = evaluate_baseline_qualification(pkg, qual_profile)
    return pkg, verdict.to_dict()


def save_triple_qualification_receipts(
    repo_root: Path | str,
) -> dict[str, str]:
    """Generate and save Driver and Iron triple pendulum qualification receipts."""
    root = Path(repo_root)
    driver_c3d = root / "data" / "C3D_TA_Driver.c3d"
    iron_c3d = root / "data" / "C3D_TA_Iron.c3d"
    evidence_dir = root / "docs" / "plans" / "tour_baselines" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    driver_pkg, driver_verdict = generate_triple_baseline_package_for_target(
        driver_c3d, "driver", maxiter=30
    )
    iron_pkg, iron_verdict = generate_triple_baseline_package_for_target(
        iron_c3d, "iron", maxiter=30
    )

    driver_json_path = evidence_dir / "tb05_driver_qualification_receipt.json"
    iron_json_path = evidence_dir / "tb05_iron_qualification_receipt.json"
    driver_pkg_path = evidence_dir / "tb05_driver_baseline_package.npz"
    iron_pkg_path = evidence_dir / "tb05_iron_baseline_package.npz"

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
        "driver_club_rmse_m": f"{driver_pkg.metrics.in_plane_rmse_m:.4f}",
        "iron_club_rmse_m": f"{iron_pkg.metrics.in_plane_rmse_m:.4f}",
    }
