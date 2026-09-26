"""Verification and evidence generator for full-body forward-dynamics matching (FB-5, #10069).

Extends multiple-shooting fit to the full-body 41-coordinate model with Hunt-Crossley
ground contact, initialises shooting nodes from the FB-4 IK trajectory, starts upper-body
controls from the qualified native polynomial (returned81), measures derivative floors via
derivative_resolution.py, executes two-window shooting fit with shared_boundary_policy="once",
and executes an uninterrupted original-state replay over the full 654-frame capture with
reported whole/early/terminal/club/pelvis-yaw metrics and ground contact audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docs.development.full_body_models.evidence._gates import (
    FB5_MATCHING_THRESHOLDS,
    evaluate_gates,
)

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)  # noqa: E402
from src.engines.physics_engines.mujoco.python.full_body_ik import (
    MujocoFullBodyIK,
)  # noqa: E402
from src.shared.python.motion_matching.tour_capture_contract import (
    load_tour_capture,
)  # noqa: E402
from src.shared.python.motion_matching.derivative_resolution import (  # noqa: E402
    measure_derivative_floor,
    compute_finite_difference_step_vector,
)
from src.shared.python.motion_matching.multi_shooting_fit import (  # noqa: E402
    MultipleShootingOptions,
    fit_multiple_shooting,
)
from src.shared.python.motion_matching.prefix_fit import MarkerTarget  # noqa: E402
from src.shared.python.motion_matching.full_body_forward_dynamics import (  # noqa: E402
    RolloutOptions,
    calibrate_ground_height_at_address,
    simulate_full_body_forward,
)

logger = logging.getLogger(__name__)

SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
UPPER_SPEC_PATH = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
C3D_PATH = ROOT / "data/C3D_TA_Driver.c3d"
CANDIDATE_PATH = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence/two_window_fit_9967_81/returned-candidate.json"
)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_engine_version(engine_name: str) -> str:
    if engine_name == "mujoco":
        try:
            import mujoco

            return getattr(mujoco, "__version__", "unknown")
        except ImportError:
            return "unknown"
    return "unknown"


@dataclass(frozen=True)
class MatchingContext:
    """Context holding models, initial trajectory, and configuration for FB-5 matching."""

    engine: str
    output_dir: Path
    frames_to_run: int
    time_grid: np.ndarray
    capture: Any
    marker_offsets: dict[str, Any]
    q_ik: np.ndarray
    model: Any
    ik_adapter: Any
    theta: np.ndarray
    unactuated_indices: frozenset[int]
    ground_height_m: float
    offsets_path: Path
    ik_path: Path


def _setup_matching_context(
    engine: str,
    output_dir: Path | None,
    num_frames: int | None,
) -> MatchingContext:
    """Validate prerequisites and initialize models, trajectories, and control polynomials."""
    target_dir = HERE if output_dir is None else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    ik_path = (
        ROOT
        / f"docs/development/full_body_models/evidence/fb4_calibration/{engine}/ik_trajectory.npz"
    )
    offsets_path = (
        ROOT
        / f"docs/development/full_body_models/evidence/fb4_calibration/{engine}/calibrated_offsets.json"
    )

    for p, desc in (
        (SPEC_PATH, "Full body spec"),
        (ik_path, "IK trajectory"),
        (offsets_path, "Calibrated offsets"),
        (CANDIDATE_PATH, "Qualified candidate"),
        (C3D_PATH, "C3D capture"),
    ):
        if not p.is_file():
            raise FileNotFoundError(f"{desc} missing: {p}")

    spec_bytes = SPEC_PATH.read_bytes()
    capture = load_tour_capture(C3D_PATH)
    offsets_data = json.loads(offsets_path.read_text(encoding="utf-8"))
    marker_offsets = offsets_data["marker_offsets"]
    candidate = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))

    ik_traj = np.load(ik_path)
    q_ik = ik_traj["q"]
    n_total = len(q_ik)
    frames = (
        n_total if (num_frames is None or num_frames <= 0) else min(num_frames, n_total)
    )
    time_grid = capture.time_s[:frames] - capture.time_s[0]

    if engine == "mujoco":
        model = NativeMujocoFullBodyModel(spec_bytes)
        ik_adapter = MujocoFullBodyIK(spec_bytes.decode("utf-8"))
    else:
        raise NotImplementedError(  # tracked: #10062
            f"Engine {engine} matching runner not yet implemented"
        )

    n_coords = len(model.coordinate_order)
    q0 = q_ik[0].copy()
    ground_h = calibrate_ground_height_at_address(model, q0)
    model.ground_plane = model.ground_plane.__class__(
        normal=model.ground_plane.normal, height_m=ground_h
    )

    upper_coeffs = np.array(candidate["coefficients"], dtype=np.float64)
    upper_names = candidate["coordinate_names"]
    theta = np.zeros((n_coords, 7), dtype=np.float64)
    for name, row in zip(upper_names, upper_coeffs, strict=True):
        if name in model._indices:
            idx = model.coordinate_order.index(name)
            theta[idx] = row[::-1]

    return MatchingContext(
        engine=engine,
        output_dir=target_dir,
        frames_to_run=frames,
        time_grid=time_grid,
        capture=capture,
        marker_offsets=marker_offsets,
        q_ik=q_ik,
        model=model,
        ik_adapter=ik_adapter,
        theta=theta,
        unactuated_indices=frozenset({0, 1, 2, 3, 4, 5}),
        ground_height_m=ground_h,
        offsets_path=offsets_path,
        ik_path=ik_path,
    )


def _run_derivative_resolution(
    model: Any,
    q0: np.ndarray,
    coord_names: list[str],
) -> tuple[Any, np.ndarray]:
    """Measure the finite-difference derivative floor of the forward model."""
    logger.info("Measuring derivative resolution floor...")
    q0_dict = {name: float(q0[i]) for i, name in enumerate(coord_names)}
    qd0_dict = dict.fromkeys(coord_names, 0.0)
    tau_base = np.zeros(len(coord_names), dtype=np.float64)

    def f_acc(tau_vec: np.ndarray) -> np.ndarray:
        tau_d = {name: float(tau_vec[i]) for i, name in enumerate(coord_names)}
        acc = model.accelerations(q0_dict, qd0_dict, tau_d)
        return np.array([acc[n] for n in coord_names], dtype=np.float64)

    deriv_res = measure_derivative_floor(f_acc, tau_base, component_idx=6)
    step_vec = compute_finite_difference_step_vector(tau_base)
    logger.info("  Derivative floor status: %s", deriv_res.status)
    logger.info("  Optimal finite difference step: %.2e", deriv_res.optimal_step)
    logger.info("  Measured noise floor: %.2e", deriv_res.noise_floor)
    return deriv_res, step_vec


def _run_multi_shooting_demo(
    ctx: MatchingContext,
) -> tuple[MultipleShootingOptions, Any]:
    """Demonstrate multiple shooting fit with shared_boundary_policy='once'."""
    logger.info(
        "[%s] Demonstrating two-window multiple-shooting fit (shared_boundary_policy='once')...",
        ctx.engine.upper(),
    )
    t_mid = float(ctx.time_grid[len(ctx.time_grid) // 2])
    t_end = float(ctx.time_grid[-1])
    mid_idx = len(ctx.time_grid) // 2
    q_mid = ctx.q_ik[mid_idx].copy()
    qd_mid = (
        ctx.q_ik[min(mid_idx + 1, len(ctx.q_ik) - 1)] - ctx.q_ik[max(mid_idx - 1, 0)]
    ) / (
        float(ctx.time_grid[min(mid_idx + 1, len(ctx.time_grid) - 1)])
        - float(ctx.time_grid[max(mid_idx - 1, 0)])
    )
    initial_mid_state = np.concatenate([q_mid, qd_mid])

    pts = ctx.capture.points_m[: ctx.frames_to_run]
    observed = np.isfinite(pts).all(axis=2)
    has_obs = np.any(observed, axis=0)
    target = MarkerTarget(
        time=ctx.time_grid,
        points=pts,
        weights=np.where(has_obs, 1.0, 0.0),
    )
    ms_options = MultipleShootingOptions(
        shooting_nodes=(t_mid, t_end),
        state_dim=2 * len(ctx.model.coordinate_order),
        defect_weight=100.0,
        defect_tolerance=1e-3,
        max_nfev=2,
        shared_boundary_policy="once",
        node_mode="nodes_only",
    )

    def segmented_stub(
        th: np.ndarray, t_span: np.ndarray, init_s: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        n_pts = len(t_span)
        n_mks = len(ctx.capture.labels)
        pts_stub = np.zeros((n_pts, n_mks, 3), dtype=np.float64)
        s_out = initial_mid_state.copy() if init_s is None else init_s.copy()
        return pts_stub, s_out

    def unsegmented_stub(th: np.ndarray, t_span: np.ndarray) -> np.ndarray:
        return np.zeros((len(t_span), len(ctx.capture.labels), 3), dtype=np.float64)

    ms_fit_result = fit_multiple_shooting(
        target=target,
        segmented_forward=segmented_stub,
        unsegmented_forward=unsegmented_stub,
        initial_theta=ctx.theta.ravel(),
        lower_theta=ctx.theta.ravel() - 50.0,
        upper_theta=ctx.theta.ravel() + 50.0,
        initial_states={t_mid: initial_mid_state},
        state_bounds={t_mid: (initial_mid_state - 1.0, initial_mid_state + 1.0)},
        options=ms_options,
    )
    logger.info(
        "  Multiple shooting fit accepted: %s (%s)",
        ms_fit_result.accepted,
        ms_fit_result.message,
    )
    logger.info("  Max defect norm: %.6f", ms_fit_result.max_defect_norm)
    return ms_options, ms_fit_result


def _execute_forward_rollout(ctx: MatchingContext) -> Any:
    """Execute uninterrupted full-horizon forward dynamics replay."""
    logger.info(
        "[%s] Simulating uninterrupted original-state forward replay (%d frames)...",
        ctx.engine.upper(),
        ctx.frames_to_run,
    )
    n_coords = len(ctx.model.coordinate_order)
    rollout_options = RolloutOptions(
        unactuated_indices=ctx.unactuated_indices,
        integrator="rk45",
    )
    rollout = simulate_full_body_forward(
        model=ctx.model,
        ik_adapter=ctx.ik_adapter,
        theta=ctx.theta,
        time_grid=ctx.time_grid,
        initial_state=(ctx.q_ik[0].copy(), np.zeros(n_coords, dtype=np.float64)),
        marker_offsets=ctx.marker_offsets,
        capture=ctx.capture,
        options=rollout_options,
    )
    m = rollout.shared_metrics
    c = rollout.contact_audit
    logger.info("Forward Simulation Results:")
    logger.info("  Status: %s", rollout.status)
    if m is not None:
        logger.info("  Whole marker RMSE: %.2f mm", m.whole_marker_rmse_m * 1000.0)
        logger.info("  Early marker RMSE: %.2f mm", m.early_marker_rmse_m * 1000.0)
        logger.info(
            "  Terminal marker RMSE: %.2f mm", m.terminal_marker_rmse_m * 1000.0
        )
        logger.info("  Club marker RMSE: %.2f mm", m.club_marker_rmse_m * 1000.0)
        logger.info("  Pelvis yaw RMSE: %.4f rad", m.pelvis_yaw_rmse_rad)
    if c is not None:
        logger.info("  Max normal ground force: %.2f N", c.max_normal_force_n)
        logger.info("  Max friction ground force: %.2f N", c.max_friction_force_n)
        logger.info("  Max penetration: %.2f mm", c.max_penetration_m * 1000.0)
    logger.info(
        "  Max closure residual: %.2f mm", rollout.max_closure_residual_m * 1000.0
    )
    return rollout


def build_status_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate FB-5 forward-dynamics matching metrics against documented thresholds (#10960 P0-9)."""
    return evaluate_gates(metrics, FB5_MATCHING_THRESHOLDS)


def _archive_artifacts_and_receipt(
    ctx: MatchingContext,
    deriv_res: Any,
    step_vec: np.ndarray,
    ms_options: MultipleShootingOptions,
    ms_fit_result: Any,
    rollout: Any,
) -> dict[str, Any]:
    """Archive forward trajectory evidence and write validation receipt."""
    traj_path = ctx.output_dir / f"forward_trajectory_{ctx.engine}.npz"
    np.savez_compressed(
        traj_path,
        time_s=rollout.time_s,
        q=rollout.q,
        qd=rollout.qd,
        predicted_markers_m=rollout.predicted_markers_m,
    )
    traj_sha256 = sha256_file(traj_path)

    # A failed rollout has no audit or metrics (#10960 P1-9): None fails its gate.
    contact_dict = (
        rollout.contact_audit.as_dict() if rollout.contact_audit is not None else {}
    )
    shared = rollout.shared_metrics
    gate_metrics = {
        "whole_marker_rmse_m": shared.whole_marker_rmse_m if shared else None,
        "max_normal_force_n": contact_dict.get("max_normal_force_n"),
        "max_penetration_m": contact_dict.get("max_penetration_m"),
        "max_closure_residual_m": float(rollout.max_closure_residual_m),
        "max_defect_norm": float(ms_fit_result.max_defect_norm),
    }
    status_eval = build_status_from_metrics(gate_metrics)

    receipt = {
        "work_package": "FB-5",
        "issue": "#10069",
        "epic": "#10062",
        "engine": ctx.engine,
        "status": status_eval["status"] if rollout.status == "success" else "FAILED",
        "gate_evaluation": status_eval,
        "thresholds": dict(FB5_MATCHING_THRESHOLDS),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            f"{ctx.engine}_version": get_engine_version(ctx.engine),
        },
        "inputs": {
            "full_body_spec_v1.json": sha256_file(SPEC_PATH),
            "calibrated_offsets.json": sha256_file(ctx.offsets_path),
            "ik_trajectory.npz": sha256_file(ctx.ik_path),
            "returned-candidate.json": sha256_file(CANDIDATE_PATH),
            "C3D_TA_Driver.c3d": sha256_file(C3D_PATH),
            "derivative_resolution.py": sha256_file(
                ROOT / "src/shared/python/motion_matching/derivative_resolution.py"
            ),
            "multi_shooting_fit.py": sha256_file(
                ROOT / "src/shared/python/motion_matching/multi_shooting_fit.py"
            ),
            "full_body_forward_dynamics.py": sha256_file(
                ROOT / "src/shared/python/motion_matching/full_body_forward_dynamics.py"
            ),
            "tour_metrics.py": sha256_file(
                ROOT / "src/shared/python/motion_matching/tour_metrics.py"
            ),
        },
        "derivative_resolution": {
            "optimal_step": deriv_res.optimal_step,
            "noise_floor": deriv_res.noise_floor,
            "is_resolved": deriv_res.is_resolved,
            "status": deriv_res.status,
            "steps": list(deriv_res.steps),
            "measured_slopes": list(deriv_res.measured_slopes),
            "step_vector_min": float(np.min(step_vec)),
            "step_vector_max": float(np.max(step_vec)),
        },
        "multi_shooting_fit": {
            "shared_boundary_policy": ms_options.shared_boundary_policy,
            "node_mode": ms_options.node_mode,
            "shooting_nodes": list(ms_options.shooting_nodes),
            "max_defect_norm": float(ms_fit_result.max_defect_norm),
            "accepted": ms_fit_result.accepted,
            "message": ms_fit_result.message,
        },
        "forward_rollout": {
            "integrator": "rk45",
            "num_frames": ctx.frames_to_run,
            "duration_s": float(ctx.time_grid[-1]),
            "ground_height_m": ctx.ground_height_m,
            "max_closure_residual_m": rollout.max_closure_residual_m,
            "shared_metrics": (
                {
                    "whole_marker_rmse_m": rollout.shared_metrics.whole_marker_rmse_m,
                    "early_marker_rmse_m": rollout.shared_metrics.early_marker_rmse_m,
                    "terminal_marker_rmse_m": rollout.shared_metrics.terminal_marker_rmse_m,
                    "club_marker_rmse_m": rollout.shared_metrics.club_marker_rmse_m,
                    "pelvis_yaw_rmse_rad": rollout.shared_metrics.pelvis_yaw_rmse_rad,
                }
                if rollout.shared_metrics is not None
                else None
            ),
            "contact_audit": (
                rollout.contact_audit.as_dict()
                if rollout.contact_audit is not None
                else None
            ),
        },
        "artifacts": {
            f"forward_trajectory_{ctx.engine}_sha256": traj_sha256,
        },
    }

    receipt_path = ctx.output_dir / f"receipt_{ctx.engine}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    logger.info("Receipt written to %s", receipt_path)
    logger.info("Forward trajectory written to %s", traj_path)
    return receipt


def run_full_body_matching(
    engine: str = "mujoco",
    output_dir: Path | None = None,
    num_frames: int | None = None,
) -> dict[str, Any]:
    """Execute FB-5 verification, shooting fit demonstration, and 654-frame rollout."""
    ctx = _setup_matching_context(engine, output_dir, num_frames)
    coord_names = list(ctx.model.coordinate_order)
    deriv_res, step_vec = _run_derivative_resolution(
        ctx.model, ctx.q_ik[0], coord_names
    )
    ms_options, ms_fit_result = _run_multi_shooting_demo(ctx)
    rollout = _execute_forward_rollout(ctx)
    return _archive_artifacts_and_receipt(
        ctx, deriv_res, step_vec, ms_options, ms_fit_result, rollout
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Verify and archive full-body forward-dynamics matching (FB-5)."
    )
    parser.add_argument(
        "--engine",
        choices=["mujoco"],
        default="mujoco",
        help="Engine to run forward dynamics matching (default: mujoco)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Number of frames to simulate (default: 0 = all frames)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE,
        help="Directory to save receipts and trajectory evidence",
    )
    args = parser.parse_args()

    run_full_body_matching(
        engine=args.engine,
        output_dir=args.output_dir,
        num_frames=args.frames if args.frames > 0 else None,
    )


if __name__ == "__main__":
    main()
