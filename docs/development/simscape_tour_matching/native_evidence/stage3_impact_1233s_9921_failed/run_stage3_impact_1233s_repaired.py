"""Stage 3: Unified Impact Delivery Polish (t = 1.233s, Frame 444 @ 360 Hz).

Milestone: Advancing from Stage 2 best candidate (480.44 mm clubhead, 11.27 mm early retention)
to certified Gate 3 compliance (clubhead <= 50 mm, terminal <= 35 mm, whole window <= 25 mm).

Key Architectural Invariants:
1. Invariant Global Basis Duration: T_basis = 1.8138888888888889 s (654 frames @ 360 Hz).
2. 100% Continuous Forward Dynamics: Zero target-state resets (Defect Norm = 0.000000 m).
3. Frozen Backswing & Transition: k=0, 1, 2, 3 100% FROZEN across ALL 27 joints.
4. Analytical Transition Decoupling:
     Delta c_{j,4} = - alpha_5 * Delta c_{j,5} - alpha_6 * Delta c_{j,6}
   where alpha_5 = 0.4 * ratio, alpha_6 = (1/15) * ratio^2, ratio = u_0 / (1 - u_0), u_0 = 0.80 / T_basis.
5. Unified Activation:
   Simultaneously optimizes:
   - Pelvis/Torso: HipInputX/Y/Z, SpineInputX/Y, TorsoInput (6 core joints -> 12 free parameters)
   - Dual Arms: Left Arm (9 joints) + Right Arm (9 joints) (18 arm joints -> 36 free parameters)
   Total 24 active joints -> 48 free parameters (k=5, 6).
6. Balanced Loss Weighting:
   - Dominant terminal clubhead tracking: 250.0 weight
   - Shaft triad (Marker_3) tracking: 120.0 weight
   - Shaft inclination vector alignment: 30.0 weight (softened from 60.0 to prevent locking out clubhead)
   - Pelvis yaw delivery alignment: 15.0 weight
   - Strict early retention barrier (t <= 0.60s): 1000.0 penalty for > 11.5 mm
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.optimize import least_squares

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("stage3_unified_impact_1233s")

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))

from src.shared.python.motion_matching.prefix_fit import bernstein_to_simscape

scratch_dir = repo / "scratch"
scratch_dir.mkdir(parents=True, exist_ok=True)

CONTROL_COUNT = 7
T_BASIS = 1.8138888888888889
T_SIM = 1.2333333333333334  # Frame 444 @ 360 Hz
T_TRANS = 0.8000000000000000

U_TRANS = T_TRANS / T_BASIS
RATIO_TRANS = U_TRANS / (1.0 - U_TRANS)
ALPHA_5 = 0.4 * RATIO_TRANS
ALPHA_6 = (1.0 / 15.0) * (RATIO_TRANS ** 2)

CORE_ACTIVE_JOINTS = [3, 4, 5, 6, 7, 8]  # HipX, HipY, HipZ, SpineX, SpineY, Torso
ARM_ACTIVE_JOINTS = [
    9, 10, 11, 12, 13, 14, 15, 16, 17,  # Left arm
    18, 19, 20, 21, 22, 23, 24, 25, 26,  # Right arm
]
UNIFIED_ACTIVE_JOINTS = CORE_ACTIVE_JOINTS + ARM_ACTIVE_JOINTS  # 24 joints


def unpack_unified_params(p_free: np.ndarray, base_theta: np.ndarray) -> np.ndarray:
    full_th = base_theta.copy().reshape(-1, CONTROL_COUNT)
    for idx_in_list, j in enumerate(UNIFIED_ACTIVE_JOINTS):
        dc5 = p_free[2 * idx_in_list]
        dc6 = p_free[2 * idx_in_list + 1]
        dc4 = -ALPHA_5 * dc5 - ALPHA_6 * dc6
        full_th[j, 4] += dc4
        full_th[j, 5] += dc5
        full_th[j, 6] += dc6
    return full_th.ravel()


def load_payload(repo_path: Path, scratch_path: Path) -> dict[str, Any]:
    payload_candidates = [
        scratch_path / "driver_marker_payload.json",
        repo_path / "data" / "driver_marker_payload.json",
        repo_path.parent / "simscape-tour-checkpoints" / "prefix-900ms-sextic-01" / "driver_marker_payload.json",
        Path("C:/Users/diete/SimscapeTour9921/driver_marker_payload.json"),
    ]
    payload_path = next((p for p in payload_candidates if p.exists()), None)
    if payload_path is None:
        raise FileNotFoundError(f"Could not find driver_marker_payload.json in {payload_candidates}")
    logger.info("Loaded marker payload from: %s", payload_path)
    return json.loads(payload_path.read_text(encoding="utf-8"))


def load_seed(repo_path: Path) -> tuple[dict[str, Any], Path]:
    seed_candidates = [
        repo_path / "docs/development/simscape_tour_matching/native_evidence/initial_velocity_seed_qualified_r2025b.json",
        Path("C:/Users/diete/SimscapeTour9921/docs/development/simscape_tour_matching/native_evidence/initial_velocity_seed_qualified_r2025b.json"),
    ]
    seed_path = next((p for p in seed_candidates if p.exists()), None)
    if seed_path is None:
        raise FileNotFoundError(f"Could not find initial velocity seed in {seed_candidates}")
    logger.info("Loaded seed from: %s", seed_path)
    return json.loads(seed_path.read_text(encoding="utf-8")), seed_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: Unified Impact Delivery Polish (t=1.233s)")
    parser.add_argument("--max-nfev", type=int, default=300, help="Max evaluations (default: 300)")
    parser.add_argument("--diff-step", type=float, default=1e-3, help="Finite difference step (default: 1e-3)")
    args = parser.parse_args()

    pid = os.getpid()
    pid_file = scratch_dir / "stage3_unified_impact.pid"
    pid_file.write_text(str(pid), encoding="utf-8")
    logger.info("Stage 3 process started with PID %d", pid)

    # 1. Marker Payload & Targets
    payload = load_payload(repo, scratch_dir)
    seed, seed_path = load_seed(repo)

    coord_names = seed["coordinate_names"]
    joints = len(coord_names)
    assert joints == 27, f"Expected 27 joints, got {joints}"

    assignments = dict(zip(seed["labels"], seed["body_names"], strict=True))
    labels = list(assignments)
    indices = [payload["labels"].index(name) for name in labels]

    raw_points = np.asarray(payload["points_world_m"])[:, indices]
    raw_valid = np.asarray(payload["valid"])[:, indices]
    raw_points[~raw_valid] = np.nan
    time_s_all = np.asarray(payload["time_s"])

    # Occlusion Repair for Marker_2:2 at frame 444
    m2_names = ["Marker_2:2:1", "Marker_2:2:2", "Marker_2:2:3"]
    m2_indices_in_labels = [labels.index(m) for m in m2_names if m in labels]
    if len(m2_indices_in_labels) == 3:
        for idx in m2_indices_in_labels:
            p443 = raw_points[443, idx]
            p445 = raw_points[445, idx]
            if np.isfinite(p443).all() and np.isfinite(p445).all():
                raw_points[444, idx] = 0.5 * (p443 + p445)
                logger.info("Repaired frame 444 occlusion for %s: %s", labels[idx], raw_points[444, idx])

    # Time horizon slicing [0, 1.233333s]
    mask_sim = time_s_all <= T_SIM + 1e-6
    target_time = time_s_all[mask_sim]
    target_points = raw_points[mask_sim]
    n_frames = len(target_time)
    logger.info("Target impact window [0, %.4fs]: %d frames", T_SIM, n_frames)

    idx_1233 = n_frames - 1
    early_mask = target_time <= 0.60 + 1e-12
    obs_all = np.isfinite(target_points).all(axis=2)

    club_indices = [
        i for i, lbl in enumerate(labels)
        if any(k in lbl.lower() for k in ("marker_2", "marker_3", "club"))
    ]
    m2_club_indices = [i for i, lbl in enumerate(labels) if "marker_2" in lbl.lower()]
    m3_shaft_indices = [i for i, lbl in enumerate(labels) if "marker_3" in lbl.lower()]
    arm_indices = [
        i for i, lbl in enumerate(labels)
        if any(k in lbl.lower() for k in ("elbow", "wrist", "uarm"))
    ]
    wr_i = labels.index("WaistRight")
    wl_i = labels.index("WaistLeft")
    lw_i = labels.index("LWristTop")
    rw_i = labels.index("RWristTop")

    # Target shaft delivery vector at impact
    tgt_lw = target_points[idx_1233, lw_i]
    tgt_rw = target_points[idx_1233, rw_i]
    tgt_mid_wrist = 0.5 * (tgt_lw + tgt_rw)
    tgt_clubhead = np.nanmean(target_points[idx_1233, m2_club_indices], axis=0)
    tgt_shaft_vec = tgt_clubhead - tgt_mid_wrist
    tgt_shaft_u = tgt_shaft_vec / np.linalg.norm(tgt_shaft_vec)
    tgt_shaft_inclination = float(np.degrees(np.arccos(abs(tgt_shaft_u[2]))))

    logger.info(
        "Target Shaft Vector at Impact: [%.4f, %.4f, %.4f] (Inclination: %.2f deg)",
        tgt_shaft_u[0], tgt_shaft_u[1], tgt_shaft_u[2], tgt_shaft_inclination,
    )

    # 2. Warm-start from best staged checkpoint (Stage 2 result)
    ckpt_path = scratch_dir / "staged_impact_best_checkpoint.json"
    if ckpt_path.exists():
        logger.info("Warm-starting from staged best checkpoint: %s", ckpt_path)
        with open(ckpt_path, "r", encoding="utf-8") as f:
            best_ckpt = json.load(f)
        base_theta = np.array(best_ckpt["theta"], dtype=np.float64)
    else:
        logger.info("Warm-starting from candidate package: candidate_staged_impact_1233s_package.json")
        pkg_path = scratch_dir / "candidate_staged_impact_1233s_package.json"
        with open(pkg_path, "r", encoding="utf-8") as f:
            pkg_data = json.load(f)
        base_theta = np.array(pkg_data["efforts"], dtype=np.float64)

    # 3. Start MATLAB Engine
    import matlab
    import matlab.engine
    logger.info("Starting MATLAB R2025b engine for Stage 3 Unified Polish...")
    eng = matlab.engine.start_matlab("-nodesktop -nosplash")
    eng_dir = repo / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab"
    eng.addpath(str(eng_dir / "src/model"), nargout=0)
    eng.addpath(eng.genpath(str(eng_dir / "src/functions")), nargout=0)
    eng.addpath(str(eng_dir / "motion_matching/shared"), nargout=0)

    eng.workspace["fit_seed_path"] = str(seed_path)
    eng.workspace["fit_frame_names"] = list(assignments.values())

    eng.eval(
        f"""
    load_system('GolfSwing3D_Kinetic'); fit_priority_cleanup=configure_capture_velocity_targets();
    fit_seed=jsondecode(fileread(fit_seed_path)); fit_ws=get_param('GolfSwing3D_Kinetic','ModelWorkspace');
    fit_geometry_names={{'UpperArmLength','LowerArmLength'}};
    for j=1:2; assignin(fit_ws,fit_geometry_names{{j}},fit_seed.geometry_in(j)); end
    [fit_ks,fit_schema]=build_golf_kinematics();
    addTargetVariables(fit_ks,fit_schema.q_ids); addOutputVariables(fit_ks,fit_schema.frame_ids); addOutputVariables(fit_ks,fit_schema.rotation_ids);
    fit_opts=capture_fit_sim_options({T_SIM});
    fit_opts.sample_rate=360; fit_opts.fast_restart=false; fit_opts.retain_raw_output=true; fit_opts.verbosity='Silent'; fit_opts.joint_names=string(fit_seed.coordinate_names)';
    for j=1:2; fit_opts.input_overrides.(fit_geometry_names{{j}})=fit_seed.geometry_in(j); end
    [found,fit_bodies]=ismember(string(fit_frame_names(:)),string({{fit_schema.frames.name}}));
    fit_offsets=fit_seed.offsets_m;
    fit_opts.simulation_time = {T_SIM + 0.001};
    """,
        nargout=0,
    )

    eng.workspace["fit_time"] = matlab.double(target_time[:, None].tolist())

    eng.eval(
        """
    for j=1:numel(fit_seed.q)
     name=fit_opts.joint_names(j); value=fit_seed.q(j); velocity=fit_seed.qd(j);
     if ~startsWith(name,'Translation'); value=rad2deg(value); velocity=rad2deg(velocity); end
     fit_opts.input_overrides.(replace(name,'Input','StartPosition'))=value;
     fit_opts.input_overrides.(replace(name,'Input','StartVelocity'))=velocity;
    end
    """,
        nargout=0,
    )

    def forward_rollout(full_theta: np.ndarray) -> np.ndarray:
        controls_basis = full_theta.reshape(joints, CONTROL_COUNT)
        native_global = bernstein_to_simscape(controls_basis, duration_s=T_BASIS)
        eng.workspace["fit_theta"] = matlab.double(native_global.ravel()[:, None].tolist())
        eng.eval(
            "[fit_pred, ~]=simulate_golf_markers(fit_theta,fit_opts,fit_ks,fit_schema,fit_bodies,fit_offsets,fit_time);",
            nargout=0,
        )
        return np.asarray(eng.workspace["fit_pred"], dtype=np.float64)

    # State tracking
    current_best_theta = base_theta.copy()
    current_best_cost = float("inf")
    current_best_clubhead_rmse = float("inf")
    eval_count = 0

    def update_best(full_th: np.ndarray, cost: float, club_rmse: float, early_rmse: float, yaw_err: float) -> None:
        nonlocal current_best_theta, current_best_cost, current_best_clubhead_rmse
        is_new_best = False
        if cost < current_best_cost:
            current_best_cost = cost
            is_new_best = True
        if club_rmse < current_best_clubhead_rmse and early_rmse <= 0.0120:
            current_best_clubhead_rmse = club_rmse
            is_new_best = True
        if is_new_best:
            current_best_theta = full_th.copy()
            coeffs = bernstein_to_simscape(
                full_th.reshape(joints, CONTROL_COUNT), duration_s=T_BASIS
            )
            ckpt: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "pid": pid,
                "evaluation": eval_count,
                "cost": cost,
                "best_cost": current_best_cost,
                "best_clubhead_rmse_mm": current_best_clubhead_rmse * 1000.0,
                "early_rmse_mm": early_rmse * 1000.0,
                "yaw_err_1233_pct": yaw_err,
                "theta": full_th.tolist(),
                "coefficients": coeffs.tolist(),
            }
            tmp_p = scratch_dir / "stage3_impact_best_checkpoint.tmp"
            tmp_p.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
            tmp_p.replace(scratch_dir / "stage3_impact_best_checkpoint.json")

    # Evaluation 0 Baseline Verification
    logger.info("=" * 70)
    logger.info("EVALUATION 0: VERIFYING STAGE 3 INITIAL BASELINE")
    logger.info("=" * 70)
    init_pts = forward_rollout(current_best_theta)
    init_diff = init_pts - target_points
    e_dists = np.linalg.norm(init_diff[early_mask][obs_all[early_mask]], axis=1)
    eval0_early_rmse = float(np.sqrt(np.mean(e_dists ** 2)))
    c_dists = np.linalg.norm(init_diff[idx_1233, club_indices], axis=1)
    eval0_club_rmse = float(np.sqrt(np.mean(c_dists ** 2)))
    logger.info("Eval 0 Early Retention: %.2f mm (Gate <= 12.0 mm)", eval0_early_rmse * 1000.0)
    logger.info("Eval 0 Terminal Clubhead: %.2f mm", eval0_club_rmse * 1000.0)

    # Parameter setup: 24 active joints * 2 params = 48 free parameters
    p_init = np.zeros(2 * len(UNIFIED_ACTIVE_JOINTS), dtype=np.float64)
    bounds_lower = np.full_like(p_init, -80.0)
    bounds_upper = np.full_like(p_init, 80.0)

    def residual_fn(p_free: np.ndarray) -> np.ndarray:
        nonlocal eval_count
        eval_count += 1
        full_th = unpack_unified_params(p_free, base_theta)
        pred_pts = forward_rollout(full_th)
        diff = pred_pts - target_points

        # 1. Dominant Terminal Clubhead Tracking at 1.233s
        term_diff = diff[idx_1233]
        res_club_list = []
        for c in m2_club_indices:
            rc = term_diff[c] * 250.0  # Dominant clubhead tracking
            if np.isfinite(rc).all():
                res_club_list.append(rc)
        for c3 in m3_shaft_indices:
            rc3 = term_diff[c3] * 120.0  # Shaft triad tracking
            if np.isfinite(rc3).all():
                res_club_list.append(rc3)
        for a in arm_indices:
            ra = term_diff[a] * 40.0
            if np.isfinite(ra).all():
                res_club_list.append(ra)

        # 2. Shaft delivery orientation
        p_lw = pred_pts[idx_1233, lw_i]
        p_rw = pred_pts[idx_1233, rw_i]
        p_mid_wrist = 0.5 * (p_lw + p_rw)
        p_club = np.nanmean(pred_pts[idx_1233, m2_club_indices], axis=0)
        p_shaft_vec = p_club - p_mid_wrist
        p_shaft_len = np.linalg.norm(p_shaft_vec)
        if p_shaft_len > 1e-4:
            p_shaft_u = p_shaft_vec / p_shaft_len
            res_shaft = (p_shaft_u - tgt_shaft_u) * 30.0
        else:
            res_shaft = np.zeros(3)

        # 3. Pelvis Yaw at 1.233s
        v_p = pred_pts[idx_1233, wr_i, :2] - pred_pts[idx_1233, wl_i, :2]
        v_t = target_points[idx_1233, wr_i, :2] - target_points[idx_1233, wl_i, :2]
        yaw_t = float(np.degrees(np.arctan2(v_t[1], v_t[0])))
        yaw_p = float(np.degrees(np.arctan2(v_p[1], v_p[0])))
        yaw_diff = float((yaw_p - yaw_t + 180) % 360 - 180)
        yaw_err_pct = float(abs(yaw_diff) / max(abs(yaw_t), 1.0) * 100.0)
        res_yaw = np.array([yaw_diff * 4.0])

        # 4. Soft barrier on early retention (t <= 0.60s)
        e_dists = np.linalg.norm(diff[early_mask][obs_all[early_mask]], axis=1)
        e_rmse = float(np.sqrt(np.mean(e_dists ** 2)))
        pen_early = np.array([max(0.0, e_rmse - 0.0115) * 1000.0])

        res_vec = np.concatenate([np.concatenate(res_club_list), res_shaft, res_yaw, pen_early])
        cost = float(0.5 * np.sum(res_vec ** 2))
        c_dists = np.linalg.norm(diff[idx_1233, club_indices], axis=1)
        c_rmse = float(np.sqrt(np.mean(c_dists ** 2)))
        update_best(full_th, cost, c_rmse, e_rmse, yaw_err_pct)

        if eval_count % 5 == 0:
            logger.info(
                "S3 Eval %3d | Cost: %.4e | Club: %6.2fmm | Shaft Diff: %5.2f deg | Early: %5.2fmm | Yaw: %+5.1f deg",
                eval_count, cost, c_rmse * 1000.0,
                abs(float(np.degrees(np.arccos(abs(p_shaft_u[2])))) - tgt_shaft_inclination),
                e_rmse * 1000.0, yaw_diff
            )
        return res_vec

    logger.info("=" * 70)
    logger.info("LAUNCHING STAGE 3: UNIFIED IMPACT DELIVERY POLISH (48 free parameters)")
    logger.info("=" * 70)
    opt_res = least_squares(
        residual_fn,
        p_init,
        bounds=(bounds_lower, bounds_upper),
        diff_step=args.diff_step,
        max_nfev=args.max_nfev,
        ftol=1e-6,
        xtol=1e-7,
        gtol=1e-6,
    )
    logger.info("Stage 3 optimization finished: status=%d msg=%s", opt_res.status, opt_res.message)

    # Rollout final best candidate
    best_pts = forward_rollout(current_best_theta)
    best_diff = best_pts - target_points

    # Gate Evaluation
    # Gate 1: Whole Window RMSE [0, 1.233s] <= 25.0 mm
    all_dists = np.linalg.norm(best_diff[obs_all], axis=1)
    whole_rmse = float(np.sqrt(np.mean(all_dists ** 2)))

    # Gate 2: Terminal marker RMSE @ 1.233s <= 35.0 mm
    term_dists = np.linalg.norm(best_diff[idx_1233, obs_all[idx_1233]], axis=1)
    term_rmse = float(np.sqrt(np.mean(term_dists ** 2)))

    # Gate 3: Clubhead terminal RMSE @ 1.233s <= 50.0 mm
    c_final_dists = np.linalg.norm(best_diff[idx_1233, club_indices], axis=1)
    club_final_rmse = float(np.sqrt(np.mean(c_final_dists ** 2)))

    # Gate 4: Early retention <= 12.0 mm
    e_final_dists = np.linalg.norm(best_diff[early_mask][obs_all[early_mask]], axis=1)
    early_final_rmse = float(np.sqrt(np.mean(e_final_dists ** 2)))

    # Gate 5: Pelvis yaw @ 1.233s < 5.0%
    v_p_f = best_pts[idx_1233, wr_i, :2] - best_pts[idx_1233, wl_i, :2]
    v_t_f = target_points[idx_1233, wr_i, :2] - target_points[idx_1233, wl_i, :2]
    yaw_t = float(np.degrees(np.arctan2(v_t_f[1], v_t_f[0])))  # repaired: was undefined here
    yaw_p_f = float(np.degrees(np.arctan2(v_p_f[1], v_p_f[0])))
    yaw_diff_f = float((yaw_p_f - yaw_t + 180) % 360 - 180)
    yaw_err_f = float(abs(yaw_diff_f) / max(abs(yaw_t), 1.0) * 100.0)

    # Gate 6: Shaft inclination diff < 5.0 deg
    p_lw_f = best_pts[idx_1233, lw_i]
    p_rw_f = best_pts[idx_1233, rw_i]
    p_club_f = np.nanmean(best_pts[idx_1233, m2_club_indices], axis=0)
    p_shaft_vec_f = p_club_f - 0.5 * (p_lw_f + p_rw_f)
    p_shaft_u_f = p_shaft_vec_f / np.linalg.norm(p_shaft_vec_f)
    shaft_inc_pred = float(np.degrees(np.arccos(abs(p_shaft_u_f[2]))))
    shaft_inc_diff = float(abs(shaft_inc_pred - tgt_shaft_inclination))

    gates = {
        "whole_window_pass": whole_rmse <= 0.0250,
        "terminal_rmse_pass": term_rmse <= 0.0350,
        "clubhead_terminal_pass": club_final_rmse <= 0.0500,
        "early_retention_pass": early_final_rmse <= 0.0120,
        "pelvis_yaw_1233_pass": yaw_err_f < 5.0,
        "shaft_inclination_pass": shaft_inc_diff < 5.0,
    }
    gates_passed = f"{sum(gates.values())}/{len(gates)}"

    logger.info("=" * 70)
    logger.info("STAGE 3 FINAL AUDIT")
    logger.info("=" * 70)
    logger.info("Whole Window RMSE:     %6.2f mm (%s)", whole_rmse * 1000.0, "PASS" if gates["whole_window_pass"] else "FAIL")
    logger.info("Terminal Marker RMSE:  %6.2f mm (%s)", term_rmse * 1000.0, "PASS" if gates["terminal_rmse_pass"] else "FAIL")
    logger.info("Clubhead Terminal:     %6.2f mm (%s)", club_final_rmse * 1000.0, "PASS" if gates["clubhead_terminal_pass"] else "FAIL")
    logger.info("Early Retention RMSE:  %6.2f mm (%s)", early_final_rmse * 1000.0, "PASS" if gates["early_retention_pass"] else "FAIL")
    logger.info("Pelvis Yaw @ 1.233s:   %6.2f%% (%+5.1f deg) (%s)", yaw_err_f, yaw_diff_f, "PASS" if gates["pelvis_yaw_1233_pass"] else "FAIL")
    logger.info("Shaft Inclination:     %6.2f deg (Diff: %5.2f deg) (%s)", shaft_inc_pred, shaft_inc_diff, "PASS" if gates["shaft_inclination_pass"] else "FAIL")
    logger.info("Gates Passed:          %s", gates_passed)

    # Save package
    best_coeffs = bernstein_to_simscape(current_best_theta.reshape(joints, CONTROL_COUNT), duration_s=T_BASIS)
    pkg = {
        "schema": "simscape-candidate-package/1",
        "candidate_id": f"prefix-1233ms-stage3-{int(time.time())}",
        "duration_s": T_SIM,
        "basis_duration_s": T_BASIS,
        "time_origin_s": 0.0,
        "polynomial_degree": CONTROL_COUNT - 1,
        "efforts": current_best_theta.tolist(),
        "coefficients": best_coeffs.tolist(),
        "rmse_m": whole_rmse,
        "early_rmse_m": early_final_rmse,
        "terminal_rmse_m": term_rmse,
        "clubhead_terminal_rmse_m": club_final_rmse,
        "pelvis_yaw_1233_diff_deg": yaw_diff_f,
        "pelvis_yaw_1233_error_pct": yaw_err_f,
        "shaft_inclination_target_deg": tgt_shaft_inclination,
        "shaft_inclination_pred_deg": shaft_inc_pred,
        "shaft_inclination_diff_deg": shaft_inc_diff,
        "max_defect_norm": 0.0,
        "gates": gates,
        "gates_passed": gates_passed,
    }

    out_pkg_path = scratch_dir / "candidate_stage3_impact_1233s_package.json"
    with open(out_pkg_path, "w", encoding="utf-8") as f:
        json.dump(pkg, f, indent=2)
    logger.info("Package written to: %s", out_pkg_path)


if __name__ == "__main__":
    main()
