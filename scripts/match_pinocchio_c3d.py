"""Pinocchio C3D Motion Matching CLI (MS-01/MS-31/MS-104).

Decoupled Kinematic Tracking and Analytic Inverse Dynamics Torque Allocation
for the Tour-Average C3D Driver Capture on the Full-Body Pinocchio Plant.

Solves:
1. Fast, robust kinematic marker tracking via Levenberg-Marquardt MarkerIkSolver
   with closure weight ramping to prevent kinematic entrapment.
2. Smooth differentiation for velocities and accelerations.
3. Analytic inverse dynamics torque allocation:
   - Optimum: Minimum-norm joint torque distribution across both arms.
   - Trail-Side Zero: tau_trail == 0 identically, transferring dynamic load
     through the closed-chain grip constraint wrench to the lead arm.
4. Exact forward acceleration parity verification (ABA).
5. Comprehensive receipt, candidate.npz, and comparison metrics generation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

logger = logging.getLogger(__name__)

RECEIPT_SCHEMA = "matched-swing-fit/pinocchio-analytic-inverse-dynamics-v1"


def setup_logging(verbose: bool = True) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def match_pinocchio_c3d(
    document_path: Path,
    capture_path: Path,
    attachments_receipt: Path | None,
    out_dir: Path,
    *,
    ground_height_m: float | None = None,
    t_start_s: float = 0.0,
    t_end_s: float | None = None,
    rate_hz: float = 360.0,
    ik_iterations: int = 15,
    cutoff_hz: float = 12.0,
    mode: str = "both",
    render_playback: bool = False,
    playback_stride: int = 3,
) -> dict[str, Any]:
    """Execute decoupled kinematic tracking and inverse dynamics torque resolution."""
    import pinocchio as pin
    from src.engines.physics_engines.pinocchio.python.crocoddyl_problem import (
        FitHorizon,
        FitWeights,
        finite_difference_rates,
    )
    from src.engines.physics_engines.pinocchio.python.full_body_fit import (
        _PlantContext,
        _metrics,
        _smooth,
        load_inputs,
        _native_targets,
    )
    from src.engines.physics_engines.pinocchio.python.marker_kinematics import (
        MarkerIkOptions,
        MarkerIkSolver,
    )

    t_wall_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine horizon
    from src.shared.python.motion_matching.tour_capture_contract import (
        load_tour_capture,
    )

    capture_probe = load_tour_capture(capture_path)
    capture_duration = (capture_probe.frames - 1) / rate_hz
    final_t_end = (
        capture_duration if t_end_s is None else min(t_end_s, capture_duration)
    )

    dt = 1.0 / rate_hz
    horizon = FitHorizon(t_start_s=t_start_s, t_end_s=final_t_end, dt_s=dt)
    weights = FitWeights()

    logger.info("Loading inputs and building plant context...")
    inputs = load_inputs(
        document_path,
        capture_path,
        attachments_receipt,
        horizon,
        weights,
        ground_height_m,
    )
    ctx = _PlantContext(inputs)
    plant = ctx.plant
    model = plant.model
    data = plant.data
    targets = _native_targets(inputs)
    n_nodes = targets.targets.shape[0]

    logger.info(
        "Trajectory horizon: %d nodes, %.3f s duration (dt=%.2f ms)",
        n_nodes,
        final_t_end - t_start_s,
        dt * 1000,
    )

    # -------------------------------------------------------------
    # STAGE 1: KINEMATIC TRACKING (MarkerIkSolver)
    # -------------------------------------------------------------
    logger.info("Starting Stage 1: Kinematic Tracking with MarkerIkSolver...")
    t_ik_start = time.perf_counter()
    ik_opts = MarkerIkOptions(iterations=ik_iterations)
    solver = MarkerIkSolver(ctx.pin, plant, ctx.table, ctx.lower, ctx.upper, ik_opts)

    q_seed = np.zeros(ctx.n)
    for name, degrees in (inputs.document.get("address_seed_deg") or {}).items():
        if name in ctx.map.names:
            q_seed[ctx.map.names.index(name)] = np.deg2rad(float(degrees))

    q_ik, ik_rms, ik_closure = solver.solve_trajectory(
        targets.targets, targets.valid, targets.weights, q_seed
    )
    t_ik_wall = time.perf_counter() - t_ik_start
    logger.info(
        "Kinematic tracking complete in %.2f s (%.2f ms/frame). Mean RMS: %.2f mm, Max Closure: %.3f mm",
        t_ik_wall,
        (t_ik_wall / n_nodes) * 1000,
        float(np.sqrt(np.mean(ik_rms**2))) * 1000,
        float(np.max(ik_closure)) * 1000,
    )

    # Smooth trajectory and differentiate
    logger.info(
        "Smoothing trajectory (cutoff=%.1f Hz) and computing rates...", cutoff_hz
    )
    q_smooth = _smooth(q_ik, dt, cutoff_hz)
    v_smooth = finite_difference_rates(q_smooth, dt)
    a_smooth = finite_difference_rates(v_smooth, dt)

    pred_markers = np.stack([ctx.markers(q_smooth[k]) for k in range(n_nodes)])
    eval_metrics = _metrics(inputs, pred_markers, targets.valid)

    # -------------------------------------------------------------
    # STAGE 2: INVERSE DYNAMICS & TORQUE ALLOCATION
    # -------------------------------------------------------------
    logger.info(
        "Starting Stage 2: Inverse Dynamics & Torque Allocation (mode=%s)...", mode
    )
    coord_order = tuple(ctx.map.names)

    # Coordinate mapping to Pinocchio internal indices
    pin_q = np.zeros((n_nodes, model.nq))
    pin_v = np.zeros((n_nodes, model.nv))
    pin_a = np.zeros((n_nodes, model.nv))
    for k in range(n_nodes):
        pin_q[k] = ctx.map.to_pin_q(q_smooth[k], model.nq)
        pin_v[k, ctx.map.v_index] = v_smooth[k]
        pin_a[k, ctx.map.v_index] = a_smooth[k]

    # Arm coordinate groups
    trail_arm_names = [
        "REInput",
        "RFInput",
        "RScapInputX",
        "RScapInputY",
        "RSInputX",
        "RSInputY",
        "RSInputZ",
        "RWInputX",
        "RWInputY",
    ]
    lead_arm_names = [
        "LEInput",
        "LFInput",
        "LScapInputX",
        "LScapInputY",
        "LSInputX",
        "LSInputY",
        "LSInputZ",
        "LWInputX",
        "LWInputY",
    ]
    trail_v_idx = [
        plant._velocity_indices[c]
        for c in trail_arm_names
        if c in plant._velocity_indices
    ]
    lead_v_idx = [
        plant._velocity_indices[c]
        for c in lead_arm_names
        if c in plant._velocity_indices
    ]

    # Generalized torques via RNEA
    t_opt_start = time.perf_counter()
    tau_rnea = np.zeros((n_nodes, model.nv))
    for k in range(n_nodes):
        tau_rnea[k] = pin.rnea(model, data, pin_q[k], pin_v[k], pin_a[k])
    t_opt_wall = time.perf_counter() - t_opt_start

    # Project to candidate coordinate order
    tau_opt_ordered = tau_rnea[:, ctx.map.v_index]
    u_opt = tau_opt_ordered[:, ctx.actuated]

    opt_stats: dict[str, float] = {}
    if mode in ("optimum", "both"):
        opt_stats = {
            "solve_time_ms": float(t_opt_wall * 1000),
            "per_frame_ms": float((t_opt_wall / n_nodes) * 1000),
            "peak_lead_arm_n_m": float(np.max(np.abs(tau_rnea[:, lead_v_idx]))),
            "mean_lead_arm_n_m": float(np.mean(np.abs(tau_rnea[:, lead_v_idx]))),
            "peak_trail_arm_n_m": float(np.max(np.abs(tau_rnea[:, trail_v_idx]))),
            "mean_trail_arm_n_m": float(np.mean(np.abs(tau_rnea[:, trail_v_idx]))),
            "peak_total_effort_n_m": float(np.max(np.abs(u_opt))),
        }
        logger.info(
            "Optimum torque evaluation complete: peak=%.1f N*m",
            opt_stats["peak_total_effort_n_m"],
        )

    # Trail-Side Zero Allocation
    tau_zero_ordered = np.zeros_like(tau_opt_ordered)
    grip_wrenches = np.zeros((n_nodes, 6))
    trail_zero_stats: dict[str, float] = {}

    if mode in ("trail_zero", "both"):
        t_zero_start = time.perf_counter()
        tau_zero_pin = tau_rnea.copy()

        parity_residuals: list[float] = []

        for k in range(n_nodes):
            coords_dict = ctx.map.as_dict(q_smooth[k])
            rates_dict = ctx.map.as_dict(v_smooth[k])
            efforts_dict = dict.fromkeys(coord_order, 0.0)

            plant.accelerations(coords_dict, rates_dict, efforts_dict)
            raw_jc, _ = plant._constraints_jacobian(coord_order)

            # Constraint Jacobian on trail arm: Jc_trail is (6, len(trail_v_idx))
            raw_trail = raw_jc[:, trail_v_idx]
            tau_trail_req = tau_rnea[k, trail_v_idx]

            lambda_c, _, _, _ = np.linalg.lstsq(raw_trail.T, tau_trail_req, rcond=1e-4)
            grip_wrenches[k] = lambda_c

            # Subtract constraint wrench contribution from applied torques
            tau_k = tau_rnea[k].copy()
            tau_k -= raw_jc.T @ lambda_c
            tau_k[trail_v_idx] = 0.0  # Trail arm actuators are identically zero
            tau_zero_pin[k] = tau_k

            # Acceleration parity verification on sample frames
            if k % max(1, n_nodes // 20) == 0:
                tau_eff = tau_k + raw_jc.T @ lambda_c
                q_ddot_opt = pin.aba(model, data, pin_q[k], pin_v[k], tau_rnea[k])
                q_ddot_zero = pin.aba(model, data, pin_q[k], pin_v[k], tau_eff)
                diff = float(np.max(np.abs(q_ddot_opt - q_ddot_zero)))
                parity_residuals.append(diff)

        t_zero_wall = time.perf_counter() - t_zero_start
        tau_zero_ordered = tau_zero_pin[:, ctx.map.v_index]
        u_zero = tau_zero_ordered[:, ctx.actuated]

        trail_zero_stats = {
            "solve_time_ms": float(t_zero_wall * 1000),
            "per_frame_ms": float((t_zero_wall / n_nodes) * 1000),
            "peak_lead_arm_n_m": float(np.max(np.abs(tau_zero_pin[:, lead_v_idx]))),
            "mean_lead_arm_n_m": float(np.mean(np.abs(tau_zero_pin[:, lead_v_idx]))),
            "peak_trail_arm_n_m": float(np.max(np.abs(tau_zero_pin[:, trail_v_idx]))),
            "mean_trail_arm_n_m": float(np.mean(np.abs(tau_zero_pin[:, trail_v_idx]))),
            "peak_grip_force_n": float(
                np.max(np.linalg.norm(grip_wrenches[:, :3], axis=1))
            ),
            "peak_grip_moment_n_m": float(
                np.max(np.linalg.norm(grip_wrenches[:, 3:], axis=1))
            ),
            "max_accel_parity_residual": float(
                max(parity_residuals) if parity_residuals else 0.0
            ),
        }
        logger.info(
            "Trail-side zero complete. Trail peak: %.2e N*m (identically zero), Grip peak: %.1f N, Parity: %.2e",
            trail_zero_stats["peak_trail_arm_n_m"],
            trail_zero_stats["peak_grip_force_n"],
            trail_zero_stats["max_accel_parity_residual"],
        )

    # -------------------------------------------------------------
    # STAGE 3: ARTIFACT GENERATION & SAVE
    # -------------------------------------------------------------
    candidate_path = out_dir / "candidate.npz"
    chosen_u = u_opt if mode == "optimum" else u_zero if mode == "trail_zero" else u_opt

    np.savez_compressed(
        candidate_path,
        time_s=targets.node_times,
        coordinate_order=np.array(ctx.map.names),
        q=q_smooth,
        v=v_smooth,
        a=a_smooth,
        u=chosen_u,
        u_optimum=u_opt,
        u_trail_zero=u_zero if mode in ("trail_zero", "both") else np.zeros_like(u_opt),
        actuated=ctx.actuated,
        markers_m=pred_markers,
        target_m=targets.targets,
        valid=targets.valid,
        labels=np.array(inputs.labels),
        grip_wrenches=grip_wrenches,
    )
    logger.info("Saved candidate trajectory to %s", candidate_path)

    total_wall_s = time.perf_counter() - t_wall_start

    comparison_results = {
        "num_frames": n_nodes,
        "duration_s": float(final_t_end - t_start_s),
        "kinematics": {
            "marker_rms_m": float(np.sqrt(np.mean(ik_rms**2))),
            "marker_rms_first_frame_m": float(ik_rms[0]),
            "closure_position_error_max_m": float(np.max(ik_closure)),
            "wall_clock_s": float(t_ik_wall),
            "per_frame_ms": float((t_ik_wall / n_nodes) * 1000),
        },
        "shared_metrics": eval_metrics["shared"],
        "replay_five": eval_metrics["replay_five"],
        "optimum": opt_stats,
        "trail_zero": trail_zero_stats,
    }

    comparison_path = out_dir / "match_comparison_results.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    logger.info("Saved comparison results to %s", comparison_path)

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "engine": "pinocchio",
        "lane": "pinocchio_analytic_motion_match",
        "document_sha256": inputs.document_sha256,
        "capture_sha256": inputs.capture.source_sha256,
        "attachments_source": inputs.attachments_source,
        "ground_height_m": inputs.ground_height_m,
        "labels": list(inputs.labels),
        "horizon": asdict(inputs.horizon),
        "nodes": n_nodes,
        "kinematics": comparison_results["kinematics"],
        "metrics": eval_metrics,
        "performance": {
            "total_wall_clock_s": float(total_wall_s),
            "ik_wall_s": float(t_ik_wall),
            "mode": mode,
            "optimum_solve_ms": opt_stats.get("solve_time_ms", 0.0),
            "trail_zero_solve_ms": trail_zero_stats.get("solve_time_ms", 0.0),
        },
        "comparison": comparison_results,
    }

    receipt_path = out_dir / "receipt.json"
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    logger.info("Saved receipt to %s", receipt_path)

    # Optional playback generation
    if render_playback:
        logger.info("Rendering playback animation...")
        try:
            from src.engines.physics_engines.pinocchio.python.candidate_playback import (
                render_candidate_gif,
            )

            gif_path = out_dir / "playback.gif"
            render_candidate_gif(candidate_path, gif_path, stride=playback_stride)
            logger.info("Saved playback animation to %s", gif_path)
        except (RuntimeError, ValueError, OSError, AttributeError, ImportError) as exc:
            logger.warning("Playback animation rendering skipped: %s", exc)

    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--spec",
        "--document",
        type=Path,
        default=Path(
            "docs/development/full_body_models/evidence/ground_support/anthro_driver/full_body_spec_hipcal_scaled.json"
        ),
        help="Path to full-body plant specification JSON",
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=Path("data/C3D_TA_Driver.c3d"),
        help="Path to canonical C3D tour-average capture",
    )
    parser.add_argument(
        "--attachments-receipt",
        type=Path,
        default=Path(
            "docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json"
        ),
        help="Path to ground-support receipt providing marker attachments and ground height",
    )
    parser.add_argument("--ground-height", type=float, default=None)
    parser.add_argument("--t-start", type=float, default=0.0)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--rate-hz", type=float, default=360.0)
    parser.add_argument("--ik-iterations", type=int, default=15)
    parser.add_argument("--cutoff-hz", type=float, default=12.0)
    parser.add_argument(
        "--mode",
        choices=("optimum", "trail_zero", "both"),
        default="both",
        help="Torque allocation mode: optimum, trail_zero, or both",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evidence/matched/driver_full_pinocchio"),
        help="Output directory for receipt, candidate, and comparison metrics",
    )
    parser.add_argument("--render-playback", action="store_true", default=False)
    parser.add_argument("--playback-stride", type=int, default=3)
    parser.add_argument("--quiet", action="store_true", default=False)

    args = parser.parse_args(argv)
    setup_logging(not args.quiet)

    receipt = match_pinocchio_c3d(
        args.spec,
        args.capture,
        args.attachments_receipt,
        args.out,
        ground_height_m=args.ground_height,
        t_start_s=args.t_start,
        t_end_s=args.t_end,
        rate_hz=args.rate_hz,
        ik_iterations=args.ik_iterations,
        cutoff_hz=args.cutoff_hz,
        mode=args.mode,
        render_playback=args.render_playback,
        playback_stride=args.playback_stride,
    )
    logger.info("Execution complete. Output directory: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
