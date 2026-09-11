"""Direct constrained sextic transition optimizer on fixed normalized time (#9921).

Implements:
- Fixed full-swing normalized time interval s = t / 1.814.
- Exact elevation from best 0.60s cubic checkpoint (prefix-600ms-cubic-refine-02).
- Parameterization options:
  - higher_order: optimizes [a4, a5, a6] per actuator (tau = tau_cubic + a4*s^4 + a5*s^5 + a6*s^6).
  - svd_subspace: optimizes along small-singular-value right singular vectors (minimal effect on 0-0.6s).
  - full_constrained: optimizes all 7 parameters with strong early-retention penalty.
- Strict multi-metric gates: early retention <= 12mm, transition RMSE <= 25mm, terminal RMSE <= 35mm,
  clubhead error <= 60mm, pelvis yaw error < 5.0%.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
repo_root = Path(__file__).resolve().parents[5]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.shared.python.motion_matching.constrained_sextic import (
    COEFFS_PER_ACTUATOR,
    T_FULL_DEFAULT,
    build_prefix_svd_basis,
    cubic_to_normalized_sextic,
    evaluate_normalized_sextic,
    normalized_to_simscape_powers,
    simscape_powers_to_normalized,
)
from src.shared.python.motion_matching.prefix_fit import (
    build_anatomical_marker_weights,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--cubic-checkpoint", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=0.70)
    parser.add_argument(
        "--mode",
        choices=["higher_order", "svd_subspace", "full_constrained"],
        default="higher_order",
        help="Search subspace mode",
    )
    parser.add_argument("--max-nfev", type=int, default=15)
    parser.add_argument("--finite-difference-step", type=float, default=0.0001)
    parser.add_argument("--smoothness-weight", type=float, default=0.08)
    parser.add_argument("--terminal-weight", type=float, default=5.0)
    parser.add_argument("--pelvis-yaw-weight", type=float, default=30.0)
    parser.add_argument("--pelvis-yaw-max-error-pct", type=float, default=5.0)
    parser.add_argument("--early-retention-weight", type=float, default=2.0)
    parser.add_argument("--early-prefix-s", type=float, default=0.60)
    return parser.parse_args()


def load_cubic_candidate(checkpoint_dir: Path) -> tuple[np.ndarray, list[str]]:
    fit_path = checkpoint_dir / "first_prefix_fit.json"
    data = json.loads(fit_path.read_text())
    # Retrieve joint names and efforts/parameters
    efforts = np.array(data["evaluations"][-1]["efforts"])
    # If cubic Bernstein was used, the native Simscape replay has the powers
    mat_path = checkpoint_dir / "final_native_replay.mat"
    import scipy.io

    mat = scipy.io.loadmat(str(mat_path))
    theta = np.array(mat["fit_theta"]).ravel()
    # theta is shape (n_joints * 7,) in Simscape descending power order
    n_joints = len(theta) // 7
    theta_matrix = theta.reshape(n_joints, 7)
    labels = data["labels"]
    return theta_matrix, labels


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    root = args.run_dir
    root.mkdir(parents=True, exist_ok=True)

    capture_path = root / "driver_marker_payload.json"
    if not capture_path.exists():
        # Fallback to copy from checkpoint
        src_payload = args.cubic_checkpoint / "driver_marker_payload.json"
        capture_path.write_bytes(src_payload.read_bytes())

    capture = json.loads(capture_path.read_text())
    seed = json.loads(args.initial_state.read_text())

    # Load baseline cubic candidate
    cubic_native, labels = load_cubic_candidate(args.cubic_checkpoint)
    n_joints = cubic_native.shape[0]

    # Convert cubic native to normalized degree-6 ascending powers on s = t / 1.814
    p_baseline = cubic_to_normalized_sextic(cubic_native, T_full=T_FULL_DEFAULT)

    # Prepare capture data
    indices = [capture["labels"].index(lbl) for lbl in labels]
    time_s = np.asarray(capture["time_s"])
    mask = time_s <= args.duration + 1e-12
    requested_times = time_s[mask]
    observed_points = np.asarray(capture["points_world_m"])[mask][:, indices]
    valid_points = np.asarray(capture["valid"])[mask][:, indices]
    observed_points[~valid_points] = np.nan

    marker_weights = build_anatomical_marker_weights(labels)
    wl_idx = labels.index("WaistLeft")
    wr_idx = labels.index("WaistRight")

    # Early retention mask (0 to 0.60s)
    early_mask = requested_times <= args.early_prefix_s + 1e-12

    # SVD basis for Vandermonde on early prefix
    W_svd, Sigma_svd, V_svd = build_prefix_svd_basis(
        T_prefix=args.early_prefix_s, T_full=T_FULL_DEFAULT, n_samples=np.count_nonzero(early_mask)
    )

    logger.info("Baseline 0.60s cubic elevated to degree 6 on s = t / 1.814")
    logger.info("Subspace mode: %s | Joints: %d | Duration: %.2f s", args.mode, n_joints, args.duration)

    # Define parameterization mapping
    if args.mode == "higher_order":
        # Search variables: [a4, a5, a6] per joint => shape (n_joints, 3)
        # p = p_baseline.copy(); p[:, 4:] += x.reshape(n_joints, 3)
        n_params = n_joints * 3
        x0 = np.zeros(n_params, dtype=np.float64)
        # Bounds on higher-order terms: allow reasonable torque range
        bounds = (-500.0 * np.ones(n_params), 500.0 * np.ones(n_params))

        def x_to_powers(x: np.ndarray) -> np.ndarray:
            p = p_baseline.copy()
            p[:, 4:] += x.reshape(n_joints, 3)
            return p

    elif args.mode == "svd_subspace":
        # Search variables: coefficients along w_5, w_6, w_7 (the 3 smallest singular values)
        # Delta p = x_j * W[:, j]
        n_sub = 3
        n_params = n_joints * n_sub
        x0 = np.zeros(n_params, dtype=np.float64)
        bounds = (-2000.0 * np.ones(n_params), 2000.0 * np.ones(n_params))
        W_active = W_svd[:, -n_sub:]  # shape (7, 3)

        def x_to_powers(x: np.ndarray) -> np.ndarray:
            p = p_baseline.copy()
            z = x.reshape(n_joints, n_sub)
            delta = z @ W_active.T  # shape (n_joints, 7)
            return p + delta

    else:  # full_constrained
        n_params = n_joints * 7
        x0 = p_baseline.ravel().copy()
        bounds = (-1000.0 * np.ones(n_params), 1000.0 * np.ones(n_params))

        def x_to_powers(x: np.ndarray) -> np.ndarray:
            return x.reshape(n_joints, 7)

    # Initialize MATLAB engine for Simscape forward replay
    import matlab
    import matlab.engine

    engine = matlab.engine.start_matlab("-nodesktop -nosplash")
    report: dict[str, Any] = {
        "status": "running",
        "mode": args.mode,
        "duration_s": args.duration,
        "evaluations": [],
    }

    try:
        engine_dir = args.repo / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab"
        engine.addpath(str(engine_dir / "src/model"), nargout=0)
        engine.addpath(engine.genpath(str(engine_dir / "src/functions")), nargout=0)
        engine.addpath(str(engine_dir / "motion_matching/shared"), nargout=0)
        engine.workspace["fit_seed_path"] = str(args.initial_state)
        engine.workspace["fit_frame_names"] = [seed["body_names"][seed["labels"].index(lbl)] for lbl in labels]
        engine.workspace["fit_duration"] = float(requested_times[-1])

        engine.eval(
            """
load_system('GolfSwing3D_Kinetic'); fit_priority_cleanup=configure_capture_velocity_targets();
fit_seed=jsondecode(fileread(fit_seed_path)); fit_ws=get_param('GolfSwing3D_Kinetic','ModelWorkspace');
fit_geometry_names={'UpperArmLength','LowerArmLength'};
for j=1:2; assignin(fit_ws,fit_geometry_names{j},fit_seed.geometry_in(j)); end
[fit_ks,fit_schema]=build_golf_kinematics();
addTargetVariables(fit_ks,fit_schema.q_ids); addOutputVariables(fit_ks,fit_schema.frame_ids); addOutputVariables(fit_ks,fit_schema.rotation_ids);
fit_opts=capture_fit_sim_options(fit_duration);
fit_opts.sample_rate=360; fit_opts.fast_restart=true; fit_opts.retain_raw_output=true; fit_opts.verbosity='Silent'; fit_opts.joint_names=string(fit_seed.coordinate_names)';
for j=1:2; fit_opts.input_overrides.(fit_geometry_names{j})=fit_seed.geometry_in(j); end
for j=1:numel(fit_seed.q)
 name=fit_opts.joint_names(j); value=fit_seed.q(j); velocity=fit_seed.qd(j);
 if ~startsWith(name,'Translation'); value=rad2deg(value); velocity=rad2deg(velocity); end
 fit_opts.input_overrides.(replace(name,'Input','StartPosition'))=value;
 fit_opts.input_overrides.(replace(name,'Input','StartVelocity'))=velocity;
end
[found,fit_bodies]=ismember(string(fit_frame_names(:)),string({fit_schema.frames.name})); assert(all(found));
fit_offsets=fit_seed.offsets_m;
""",
            nargout=0,
        )

        def forward_sim(powers_normalized: np.ndarray) -> np.ndarray:
            # Convert normalized ascending powers to Simscape native descending A..G
            native_ag = normalized_to_simscape_powers(powers_normalized, T_full=T_FULL_DEFAULT)
            theta_flat = native_ag.ravel()
            engine.workspace["fit_theta"] = matlab.double(theta_flat[:, None].tolist())
            engine.workspace["fit_time"] = matlab.double(requested_times[:, None].tolist())
            engine.eval(
                "[fit_prediction,fit_last_replay]=simulate_golf_markers(fit_theta,fit_opts,fit_ks,fit_schema,fit_bodies,fit_offsets,fit_time);",
                nargout=0,
            )
            pred = np.asarray(engine.workspace["fit_prediction"])
            return pred

        eval_count = 0

        def residual(x: np.ndarray) -> np.ndarray:
            nonlocal eval_count
            eval_count += 1
            t_start = time.monotonic()
            powers = x_to_powers(x)
            pred = forward_sim(powers)

            # 1. Base marker tracking residuals
            diff = pred - observed_points
            dist = np.linalg.norm(diff, axis=-1)  # (N_times, N_markers)

            # Quadratic time weighting
            s_t = np.clip(requested_times / args.duration, 0.0, 1.0)
            t_mult = 1.0 + 4.0 * (s_t**2)
            base_w = np.sqrt(t_mult[:, None] * marker_weights[None, :])
            # Additional weight on early retention
            base_w[early_mask] *= args.early_retention_weight

            res_markers = (diff * base_w[:, :, None]).ravel()
            res_list = [res_markers]

            # 2. Dedicated terminal frame penalty
            term_diff = diff[-1]  # (N_markers, 3)
            term_w = np.sqrt(marker_weights) * args.terminal_weight
            res_list.append((term_diff * term_w[:, None]).ravel())

            # 3. Pelvis yaw penalty (< 5% error requirement)
            v_p = pred[:, wr_idx, :2] - pred[:, wl_idx, :2]
            v_t = observed_points[:, wr_idx, :2] - observed_points[:, wl_idx, :2]
            u_p = v_p / (np.linalg.norm(v_p, axis=-1, keepdims=True) + 1e-9)
            u_t = v_t / (np.linalg.norm(v_t, axis=-1, keepdims=True) + 1e-9)
            sin_yaw = u_p[:, 1] * u_t[:, 0] - u_p[:, 0] * u_t[:, 1]
            yaw_res = sin_yaw * (args.pelvis_yaw_weight * t_mult)
            res_list.append(yaw_res.ravel())
            res_list.append(sin_yaw[-1:] * (args.pelvis_yaw_weight * args.terminal_weight))

            # 4. Regularization on parameter perturbation
            if args.smoothness_weight > 0:
                res_list.append((np.sqrt(args.smoothness_weight) * x).ravel())

            overall_rms = float(np.sqrt(np.nanmean(dist**2)))
            term_rms = float(np.sqrt(np.nanmean(dist[-1] ** 2)))
            term_yaw_err_pct = float(abs(np.degrees(np.arcsin(np.clip(sin_yaw[-1], -1.0, 1.0)))) / 60.0 * 100.0)

            report["evaluations"].append(
                {
                    "eval": eval_count,
                    "elapsed_s": time.monotonic() - t_start,
                    "overall_rms_mm": overall_rms * 1000,
                    "term_rms_mm": term_rms * 1000,
                    "term_yaw_err_pct": term_yaw_err_pct,
                }
            )

            if eval_count % 5 == 0:
                logger.info(
                    "Eval %d: overall RMS %.2f mm | terminal RMS %.2f mm | yaw err %.2f%%",
                    eval_count,
                    overall_rms * 1000,
                    term_rms * 1000,
                    term_yaw_err_pct,
                )
                (root / "constrained_sextic_fit.json").write_text(json.dumps(report, indent=2))

            return np.concatenate(res_list)

        # Baseline evaluation
        logger.info("Evaluating baseline elevated cubic...")
        base_pred = forward_sim(p_baseline)
        base_diff = np.linalg.norm(base_pred - observed_points, axis=-1)
        logger.info(
            "Baseline RMS: overall = %.2f mm | early (0-0.6s) = %.2f mm | terminal = %.2f mm",
            np.sqrt(np.nanmean(base_diff**2)) * 1000,
            np.sqrt(np.nanmean(base_diff[early_mask] ** 2)) * 1000,
            np.sqrt(np.nanmean(base_diff[-1] ** 2)) * 1000,
        )

        # Optimize
        logger.info("Starting constrained least_squares optimization (max_nfev=%d)...", args.max_nfev)
        opt_res = least_squares(
            residual,
            x0,
            bounds=bounds,
            max_nfev=args.max_nfev,
            diff_step=args.finite_difference_step,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            x_scale="jac",
        )

        opt_powers = x_to_powers(opt_res.x)
        final_pred = forward_sim(opt_powers)
        final_diff = np.linalg.norm(final_pred - observed_points, axis=-1)

        early_rms = float(np.sqrt(np.nanmean(final_diff[early_mask] ** 2)))
        overall_rms = float(np.sqrt(np.nanmean(final_diff**2)))
        terminal_rms = float(np.sqrt(np.nanmean(final_diff[-1] ** 2)))
        terminal_max = float(np.nanmax(final_diff[-1]))

        # Pelvis yaw check
        v_p_end = final_pred[-1, wr_idx, :2] - final_pred[-1, wl_idx, :2]
        v_t_end = observed_points[-1, wr_idx, :2] - observed_points[-1, wl_idx, :2]
        yaw_target = float(np.degrees(np.arctan2(v_t_end[1], v_t_end[0])))
        yaw_pred = float(np.degrees(np.arctan2(v_p_end[1], v_p_end[0])))
        yaw_diff = float((yaw_pred - yaw_target + 180) % 360 - 180)
        yaw_err_pct = float(abs(yaw_diff) / max(abs(yaw_target), 1.0) * 100.0)

        # Clubhead errors
        club_indices = [i for i, lbl in enumerate(labels) if "club" in lbl.lower() or "marker_3:" in lbl.lower()]
        club_term_rms = (
            float(np.sqrt(np.nanmean(final_diff[-1, club_indices] ** 2))) if club_indices else 0.0
        )

        gates_passed = bool(
            early_rms <= 0.012
            and overall_rms <= 0.025
            and terminal_rms <= 0.035
            and club_term_rms <= 0.060
            and yaw_err_pct < 5.0
        )

        report.update(
            {
                "status": "completed",
                "converged": bool(opt_res.success),
                "message": str(opt_res.message),
                "gates_passed": gates_passed,
                "metrics": {
                    "early_rms_mm": early_rms * 1000,
                    "overall_rms_mm": overall_rms * 1000,
                    "terminal_rms_mm": terminal_rms * 1000,
                    "terminal_max_mm": terminal_max * 1000,
                    "club_terminal_rms_mm": club_term_rms * 1000,
                    "pelvis_yaw_pred_deg": yaw_pred,
                    "pelvis_yaw_target_deg": yaw_target,
                    "pelvis_yaw_diff_deg": yaw_diff,
                    "pelvis_yaw_error_pct": yaw_err_pct,
                },
                "optimized_powers": opt_powers.tolist(),
            }
        )

        (root / "constrained_sextic_fit.json").write_text(json.dumps(report, indent=2))
        logger.info("Optimization finished! Gates passed: %s", gates_passed)
        logger.info("Metrics: overall=%.2f mm, terminal=%.2f mm, yaw_err=%.2f%%", overall_rms*1000, terminal_rms*1000, yaw_err_pct)
        return 0 if gates_passed else 1

    finally:
        engine.eval(
            "if exist('fit_priority_cleanup','var'); clear fit_priority_cleanup; end; if bdIsLoaded('GolfSwing3D_Kinetic'); set_param('GolfSwing3D_Kinetic','FastRestart','off'); close_system('GolfSwing3D_Kinetic',0); bdclose('all'); end",
            nargout=0,
        )
        engine.quit()


if __name__ == "__main__":
    sys.exit(main())
