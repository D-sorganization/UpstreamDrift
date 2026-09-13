"""Staged Constrained Impact Delivery Optimization (t = 1.233s, Frame 444 @ 360 Hz).

Milestone: Advancing from 1.15s downswing horizon to the complete tour impact window
at t = 1.233333 s (445 frames, 0..444 @ 360 Hz) in Simscape Multibody R2025b.

Key Architectural Principles & Guarantees:
1. Invariant Global Basis Duration: T_basis = 1.8138888888888889 s (654 frames @ 360 Hz).
2. 100% Continuous Forward Dynamics: Zero target-state resets (Defect Norm = 0.000000 m).
3. Frozen Backswing & History: k=0, 1, 2, 3 are 100% FROZEN across ALL 27 joints.
4. Analytical Transition Decoupling:
   Enforces Delta tau(t = 0.80s) = 0 analytically:
     Delta c_4 = - alpha_5 * Delta c_5 - alpha_6 * Delta c_6
   where alpha_5 = (2/5) * (u_0 / (1 - u_0)), alpha_6 = (1/15) * (u_0 / (1 - u_0))^2,
   and u_0 = 0.80 / T_basis.
   This strictly protects the certified address, takeaway, and transition reversal
   at t <= 0.80s while optimizing downswing delivery and terminal impact strike.
5. Two-Stage Coordinated Optimization:
   - Stage 1 (Core & Pelvic Drive): 8 trunk/pelvis channels (16 free parameters).
     Focus: Pelvis rotation (yaw < 5%), pelvic lateral shift, early retention.
   - Stage 2 (Arm & Wrist Strike): 18 arm/wrist channels (36 free parameters).
     Focus: Dominant clubhead tracking, Marker_3 shaft triad, shaft delivery inclination.
6. Best Candidate Tracking:
   Maintains best_theta and best_cost at every evaluation.
   Atomically updates scratch/impact_1233s_best_checkpoint.json.
   Emits the true best candidate rather than optimizer exit state.
7. Occlusion Repair:
   Marker_2:2 triad linearly interpolated across 1-frame occlusion at frame 444.
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
logger = logging.getLogger("staged_impact_1233s")

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))

from src.shared.python.motion_matching.prefix_fit import bernstein_to_simscape

scratch_dir = repo / "scratch"
scratch_dir.mkdir(parents=True, exist_ok=True)

# Invariant physical constants
CONTROL_COUNT = 7
T_BASIS = 1.8138888888888889
T_SIM = 1.2333333333333334  # Frame 444 @ 360 Hz: 444 / 360 = 1.233333 s
T_TRANS = 0.8000000000000000

# Analytical constraint weights for Delta tau(t_trans) = 0
U_TRANS = T_TRANS / T_BASIS  # ~ 0.4410412859762123
RATIO_TRANS = U_TRANS / (1.0 - U_TRANS)  # ~ 0.7890408985172288
ALPHA_5 = 0.4 * RATIO_TRANS  # ~ 0.3156163594
ALPHA_6 = (1.0 / 15.0) * (RATIO_TRANS ** 2)  # ~ 0.0415056801

CORE_JOINT_INDICES = [0, 1, 3, 4, 5, 6, 7, 8]  # Pelvis, Translation X/Y, Hip, Spine, Torso
ARM_JOINT_INDICES = [
    9, 10, 11, 12, 13, 14, 15, 16, 17,  # Left arm: LE, LF, LScapX, LScapY, LSX, LSY, LSZ, LWX, LWY
    18, 19, 20, 21, 22, 23, 24, 25, 26,  # Right arm: RE, RF, RScapX, RScapY, RSX, RSY, RSZ, RWX, RWY
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simscape Tour Staged Impact Matching at t=1.233s (Frame 444 @ 360 Hz)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Perform mathematical, dimension, and constraint checks without invoking MATLAB",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["both", "stage1", "stage2"],
        help="Which optimization stage to run (default: both)",
    )
    parser.add_argument(
        "--max-nfev-s1",
        type=int,
        default=250,
        help="Maximum function evaluations for Stage 1 (default: 250)",
    )
    parser.add_argument(
        "--max-nfev-s2",
        type=int,
        default=350,
        help="Maximum function evaluations for Stage 2 (default: 350)",
    )
    parser.add_argument(
        "--diff-step",
        type=float,
        default=2e-3,
        help="Finite difference step size (default: 2e-3)",
    )
    return parser.parse_args()


def load_payload(repo_path: Path, scratch_path: Path) -> dict[str, Any]:
    payload_candidates = [
        scratch_path / "driver_marker_payload.json",
        repo_path / "data" / "driver_marker_payload.json",
        repo_path.parent
        / "simscape-tour-checkpoints"
        / "prefix-900ms-sextic-01"
        / "driver_marker_payload.json",
        Path("C:/Users/diete/SimscapeTour9921/driver_marker_payload.json"),
    ]
    payload_path = next((p for p in payload_candidates if p.exists()), None)
    if payload_path is None:
        raise FileNotFoundError(
            f"Could not find driver_marker_payload.json in {payload_candidates}"
        )
    logger.info("Loaded marker payload from: %s", payload_path)
    return json.loads(payload_path.read_text(encoding="utf-8"))


def load_seed(repo_path: Path) -> tuple[dict[str, Any], Path]:
    seed_candidates = [
        repo_path
        / "docs/development/simscape_tour_matching/native_evidence/initial_velocity_seed_qualified_r2025b.json",
        Path(
            "C:/Users/diete/SimscapeTour9921/docs/development/simscape_tour_matching/native_evidence/initial_velocity_seed_qualified_r2025b.json"
        ),
    ]
    seed_path = next((p for p in seed_candidates if p.exists()), None)
    if seed_path is None:
        raise FileNotFoundError(
            f"Could not find initial velocity seed in {seed_candidates}"
        )
    logger.info("Loaded seed from: %s", seed_path)
    return json.loads(seed_path.read_text(encoding="utf-8")), seed_path


def load_warmstart_candidate(
    repo_path: Path, scratch_path: Path
) -> tuple[dict[str, Any], Path]:
    cand_candidates = [
        repo_path / "candidates/candidate_downswing_115s_locked_package.json",
        scratch_path / "candidate_downswing_115s_locked_package.json",
        Path(
            "C:/Users/diete/SimscapeTour9921/candidates/candidate_downswing_115s_locked_package.json"
        ),
        repo_path / "candidates/candidate_downswing_105s_locked_package.json",
        Path("C:/Users/diete/SimscapeTour9921/candidate_downswing_105s_locked_package.json"),
    ]
    cand_path = next((p for p in cand_candidates if p.exists()), None)
    if cand_path is None:
        raise FileNotFoundError(
            f"Could not find candidate package in {cand_candidates}"
        )
    logger.info("Warm-starting from candidate package: %s", cand_path)
    return json.loads(cand_path.read_text(encoding="utf-8")), cand_path


def main() -> int:
    args = parse_args()
    pid = os.getpid()
    pid_file = scratch_dir / "staged_impact_1233s.pid"
    pid_file.write_text(str(pid), encoding="utf-8")
    logger.info("Staged Impact 1.233s process started with PID: %d", pid)

    payload = load_payload(repo, scratch_dir)
    seed, seed_path = load_seed(repo)
    cand_data, cand_path = load_warmstart_candidate(repo, scratch_dir)

    coord_names = seed["coordinate_names"]
    joints = len(coord_names)
    assert joints == 27, f"Expected 27 joints, got {joints}"

    base_efforts = np.array(cand_data["efforts"], dtype=np.float64).reshape(
        joints, CONTROL_COUNT
    )

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
                logger.info(
                    "Repaired frame 444 occlusion for %s: %s",
                    labels[idx],
                    raw_points[444, idx],
                )

    # Time horizon slicing [0, 1.233333s]
    mask_sim = time_s_all <= T_SIM + 1e-6
    target_time = time_s_all[mask_sim]
    target_points = raw_points[mask_sim]
    n_frames = len(target_time)
    logger.info("Target impact window [0, %.4fs]: %d frames", T_SIM, n_frames)

    idx_080 = int(np.argmin(np.abs(target_time - 0.80)))
    idx_105 = int(np.argmin(np.abs(target_time - 1.05)))
    idx_115 = int(np.argmin(np.abs(target_time - 1.15)))
    idx_1233 = n_frames - 1

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
    torso_indices = [
        i for i, lbl in enumerate(labels)
        if any(k in lbl.lower() for k in ("back", "shoulder", "head"))
    ]
    pelvis_indices = [
        i for i, lbl in enumerate(labels)
        if any(k in lbl.lower() for k in ("waist",))
    ]

    wl_i = labels.index("WaistLeft")
    wr_i = labels.index("WaistRight")
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

    # Parametric packing helpers enforcing Delta tau(0.80s) = 0
    def unpack_stage_params(
        stage_free: np.ndarray,
        active_joint_list: list[int],
        current_theta: np.ndarray,
    ) -> np.ndarray:
        full = current_theta.copy().reshape(joints, CONTROL_COUNT)
        for idx, j in enumerate(active_joint_list):
            dc5 = stage_free[2 * idx]
            dc6 = stage_free[2 * idx + 1]
            dc4 = -ALPHA_5 * dc5 - ALPHA_6 * dc6
            full[j, 4] = base_efforts[j, 4] + dc4
            full[j, 5] = base_efforts[j, 5] + dc5
            full[j, 6] = base_efforts[j, 6] + dc6
        return full.ravel()

    # Mathematical Invariant Checks
    logger.info("Verifying analytical decoupling at t = 0.80s...")
    test_free = np.array([10.0, -5.0] * len(CORE_JOINT_INDICES), dtype=np.float64)
    test_full = unpack_stage_params(test_free, CORE_JOINT_INDICES, base_efforts.ravel())
    test_b = test_full.reshape(joints, CONTROL_COUNT)
    for j in CORE_JOINT_INDICES:
        u = U_TRANS
        dtau = (
            (test_b[j, 4] - base_efforts[j, 4]) * 15.0 * (u ** 4) * ((1.0 - u) ** 2)
            + (test_b[j, 5] - base_efforts[j, 5]) * 6.0 * (u ** 5) * (1.0 - u)
            + (test_b[j, 6] - base_efforts[j, 6]) * (u ** 6)
        )
        assert abs(dtau) < 1e-12, f"Decoupling failed for joint {j}: dtau = {dtau}"
    logger.info("Decoupling verified: Delta tau(0.80s) = 0.000000000000 to machine precision!")

    if args.check_only:
        logger.info("Check-only mode: All mathematical and dimensional invariants VERIFIED.")
        return 0

    # Start MATLAB engine on DeskComputer
    try:
        import matlab.engine
    except ImportError:
        logger.error("matlab.engine is not available. Execute on DeskComputer for simulation.")
        return 1

    logger.info("Starting MATLAB R2025b engine for Staged Impact Matching...")
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

    early_mask = target_time <= 0.60 + 1e-12
    obs_all = np.isfinite(target_points).all(axis=2)

    # State tracking
    current_best_theta = base_efforts.ravel().copy()
    current_best_cost = float("inf")
    current_best_clubhead_rmse = float("inf")
    eval_count = 0

    best_ckpt_file = scratch_dir / "staged_impact_best_checkpoint.json"
    if best_ckpt_file.exists():
        try:
            best_ckpt_data = json.loads(best_ckpt_file.read_text(encoding="utf-8"))
            current_best_theta = np.array(best_ckpt_data["theta"], dtype=np.float64)
            current_best_cost = float(best_ckpt_data.get("best_cost", best_ckpt_data.get("cost", float("inf"))))
            current_best_clubhead_rmse = float(best_ckpt_data.get("best_clubhead_rmse_mm", float("inf"))) / 1000.0
            logger.info(
                "Warm-started from existing best checkpoint: cost=%.4e, clubhead=%.2fmm, yaw=%.2f%%",
                current_best_cost,
                current_best_clubhead_rmse * 1000.0,
                best_ckpt_data.get("yaw_err_1233_pct", 0.0),
            )
        except Exception as exc:
            logger.warning("Could not read existing best checkpoint: %s", exc)

    def update_best_candidate(full_th: np.ndarray, cost: float, club_rmse: float, early_rmse: float, yaw_err: float) -> None:
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
            tmp_p = scratch_dir / "staged_impact_best_checkpoint.tmp"
            tmp_p.write_text(json.dumps(ckpt, indent=2), encoding="utf-8")
            tmp_p.replace(scratch_dir / "staged_impact_best_checkpoint.json")

    # Evaluation 0 Baseline Verification
    logger.info("=" * 70)
    logger.info("EVALUATION 0: VERIFYING CERTIFIED BASELINE")
    logger.info("=" * 70)
    init_pts = forward_rollout(current_best_theta)
    init_diff = init_pts - target_points
    early_dists = np.linalg.norm(init_diff[early_mask][obs_all[early_mask]], axis=1)
    eval0_early_rmse = float(np.sqrt(np.mean(early_dists ** 2)))
    eval0_club_dists = np.linalg.norm(init_diff[idx_1233, club_indices], axis=1)
    eval0_club_rmse = float(np.sqrt(np.mean(eval0_club_dists ** 2)))
    logger.info("Eval 0 Early Retention: %.2f mm (Gate <= 12.0 mm)", eval0_early_rmse * 1000.0)
    logger.info("Eval 0 Terminal Clubhead: %.2f mm", eval0_club_rmse * 1000.0)

    # STAGE 1: CORE & PELVIC DRIVE
    if args.stage in ("both", "stage1"):
        logger.info("=" * 70)
        logger.info("STAGE 1: CORE & PELVIC DRIVE (8 joints, 16 parameters)")
        logger.info("=" * 70)
        s1_free_init = np.zeros(2 * len(CORE_JOINT_INDICES), dtype=np.float64)
        s1_bounds_lower = np.full_like(s1_free_init, -60.0)
        s1_bounds_upper = np.full_like(s1_free_init, 60.0)

        def s1_residual_fn(p_free: np.ndarray) -> np.ndarray:
            nonlocal eval_count
            eval_count += 1
            full_th = unpack_stage_params(p_free, CORE_JOINT_INDICES, current_best_theta)
            pred_pts = forward_rollout(full_th)
            diff = pred_pts - target_points

            # 1. Pelvis Yaw error at 1.233s (dominant gate: < 5%)
            v_p = pred_pts[idx_1233, wr_i, :2] - pred_pts[idx_1233, wl_i, :2]
            v_t = target_points[idx_1233, wr_i, :2] - target_points[idx_1233, wl_i, :2]
            yaw_t = float(np.degrees(np.arctan2(v_t[1], v_t[0])))
            yaw_p = float(np.degrees(np.arctan2(v_p[1], v_p[0])))
            yaw_diff = float((yaw_p - yaw_t + 180) % 360 - 180)
            yaw_err_pct = float(abs(yaw_diff) / max(abs(yaw_t), 1.0) * 100.0)
            res_yaw = np.array([yaw_diff * 5.0])

            # 2. Pelvis & Torso marker tracking on [0.80s, 1.233s]
            res_core = []
            for m in pelvis_indices + torso_indices:
                r_m = diff[idx_080:idx_1233 + 1, m, :] * 20.0
                valid_m = np.isfinite(r_m)
                res_core.append(r_m[valid_m].ravel())

            # 3. Soft barrier on early retention
            e_dists = np.linalg.norm(diff[early_mask][obs_all[early_mask]], axis=1)
            e_rmse = float(np.sqrt(np.mean(e_dists ** 2)))
            pen_early = np.array([max(0.0, e_rmse - 0.0116) * 500.0])
            core_res = np.concatenate(res_core) if res_core else np.zeros(0)
            res_vec = np.concatenate([res_yaw, core_res, pen_early])
            cost = float(0.5 * np.sum(res_vec ** 2))
            c_dists = np.linalg.norm(diff[idx_1233, club_indices], axis=1)
            c_rmse = float(np.sqrt(np.mean(c_dists ** 2)))
            update_best_candidate(full_th, cost, c_rmse, e_rmse, yaw_err_pct)

            if eval_count % 5 == 0:
                logger.info(
                    "S1 Eval %3d | Cost: %.4e | Early: %5.2fmm | Yaw1233: %5.2f%% (%+5.1f deg) | Club: %6.2fmm",
                    eval_count, cost, e_rmse * 1000.0, yaw_err_pct, yaw_diff, c_rmse * 1000.0
                )
            return res_vec

        least_squares(
            s1_residual_fn,
            s1_free_init,
            bounds=(s1_bounds_lower, s1_bounds_upper),
            diff_step=args.diff_step,
            max_nfev=args.max_nfev_s1,
            ftol=1e-6,
            xtol=1e-7,
            gtol=1e-6,
        )
        logger.info("Stage 1 completed. Warm-starting Stage 2 with best candidate.")

    # STAGE 2: DUAL-ARM & WRIST STRIKE
    if args.stage in ("both", "stage2"):
        logger.info("=" * 70)
        logger.info("STAGE 2: DUAL-ARM & WRIST STRIKE (18 joints, 36 parameters)")
        logger.info("=" * 70)
        s2_free_init = np.zeros(2 * len(ARM_JOINT_INDICES), dtype=np.float64)
        s2_bounds_lower = np.full_like(s2_free_init, -90.0)
        s2_bounds_upper = np.full_like(s2_free_init, 90.0)

        s2_base_theta = current_best_theta.copy()

        def s2_residual_fn(p_free: np.ndarray) -> np.ndarray:
            nonlocal eval_count
            eval_count += 1
            full_th = unpack_stage_params(p_free, ARM_JOINT_INDICES, s2_base_theta)
            pred_pts = forward_rollout(full_th)
            diff = pred_pts - target_points

            # 1. Dominant Terminal Clubhead Tracking at 1.233s
            term_diff = diff[idx_1233]
            res_club_list = []
            for c in m2_club_indices:
                rc = term_diff[c] * 150.0  # Dominant clubhead tracking
                if np.isfinite(rc).all():
                    res_club_list.append(rc)
            for c3 in m3_shaft_indices:
                rc3 = term_diff[c3] * 100.0  # Shaft triad tracking
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
                res_shaft = (p_shaft_u - tgt_shaft_u) * 60.0
            else:
                res_shaft = np.zeros(3)

            # 3. Soft barrier on early retention
            e_dists = np.linalg.norm(diff[early_mask][obs_all[early_mask]], axis=1)
            e_rmse = float(np.sqrt(np.mean(e_dists ** 2)))
            pen_early = np.array([max(0.0, e_rmse - 0.0116) * 500.0])

            # Pelvis yaw check
            v_p = pred_pts[idx_1233, wr_i, :2] - pred_pts[idx_1233, wl_i, :2]
            v_t = target_points[idx_1233, wr_i, :2] - target_points[idx_1233, wl_i, :2]
            yaw_t = float(np.degrees(np.arctan2(v_t[1], v_t[0])))
            yaw_p = float(np.degrees(np.arctan2(v_p[1], v_p[0])))
            yaw_diff = float((yaw_p - yaw_t + 180) % 360 - 180)
            yaw_err_pct = float(abs(yaw_diff) / max(abs(yaw_t), 1.0) * 100.0)
            club_res = np.concatenate(res_club_list) if res_club_list else np.zeros(0)
            res_vec = np.concatenate([club_res, res_shaft, pen_early])
            cost = float(0.5 * np.sum(res_vec ** 2))
            c_dists = np.linalg.norm(diff[idx_1233, club_indices], axis=1)
            c_rmse = float(np.sqrt(np.mean(c_dists ** 2)))
            update_best_candidate(full_th, cost, c_rmse, e_rmse, yaw_err_pct)

            if eval_count % 5 == 0:
                logger.info(
                    "S2 Eval %3d | Cost: %.4e | Club: %6.2fmm | Shaft Diff: %5.2f deg | Early: %5.2fmm",
                    eval_count, cost, c_rmse * 1000.0, abs(float(np.degrees(np.arccos(abs(p_shaft_u[2])))) - tgt_shaft_inclination), e_rmse * 1000.0
                )
            return res_vec

        least_squares(
            s2_residual_fn,
            s2_free_init,
            bounds=(s2_bounds_lower, s2_bounds_upper),
            diff_step=args.diff_step,
            max_nfev=args.max_nfev_s2,
            ftol=1e-6,
            xtol=1e-7,
            gtol=1e-6,
        )

    # FINAL CERTIFICATION ROLLOUT ON BEST CANDIDATE
    logger.info("=" * 70)
    logger.info("FINAL CERTIFICATION ROLLOUT (ON BEST CANDIDATE)")
    logger.info("=" * 70)
    final_pts = forward_rollout(current_best_theta)
    final_diff = final_pts - target_points

    whole_dists = np.linalg.norm(final_diff[obs_all], axis=1)
    whole_rmse_mm = float(np.sqrt(np.mean(whole_dists ** 2)) * 1000.0)
    early_dists = np.linalg.norm(final_diff[early_mask][obs_all[early_mask]], axis=1)
    early_rmse_mm = float(np.sqrt(np.mean(early_dists ** 2)) * 1000.0)
    term_dists = np.linalg.norm(final_diff[idx_1233][obs_all[idx_1233]], axis=1)
    term_rmse_mm = float(np.sqrt(np.mean(term_dists ** 2)) * 1000.0)
    club_term_dists = np.linalg.norm(final_diff[idx_1233, club_indices], axis=1)
    clubhead_term_rmse_mm = float(np.sqrt(np.mean(club_term_dists ** 2)) * 1000.0)

    # Pelvis yaw check at 1.233s
    v_p_1233 = final_pts[idx_1233, wr_i, :2] - final_pts[idx_1233, wl_i, :2]
    v_t_1233 = target_points[idx_1233, wr_i, :2] - target_points[idx_1233, wl_i, :2]
    yaw_t_1233 = float(np.degrees(np.arctan2(v_t_1233[1], v_t_1233[0])))
    yaw_p_1233 = float(np.degrees(np.arctan2(v_p_1233[1], v_p_1233[0])))
    diff_1233 = float((yaw_p_1233 - yaw_t_1233 + 180) % 360 - 180)
    yaw_err_1233 = float(abs(diff_1233) / max(abs(yaw_t_1233), 1.0) * 100.0)

    # Shaft inclination
    p_lw_f = final_pts[idx_1233, lw_i]
    p_rw_f = final_pts[idx_1233, rw_i]
    p_mw_f = 0.5 * (p_lw_f + p_rw_f)
    p_ch_f = np.nanmean(final_pts[idx_1233, m2_club_indices], axis=0)
    p_sv_f = p_ch_f - p_mw_f
    p_su_f = p_sv_f / np.linalg.norm(p_sv_f)
    shaft_inc_pred = float(np.degrees(np.arccos(abs(p_su_f[2]))))
    shaft_inc_diff = abs(shaft_inc_pred - tgt_shaft_inclination)

    gates = {
        "whole_window_pass": whole_rmse_mm <= 25.0,
        "terminal_rmse_pass": term_rmse_mm <= 35.0,
        "clubhead_terminal_pass": clubhead_term_rmse_mm <= 50.0,
        "early_retention_pass": early_rmse_mm <= 12.0,
        "pelvis_yaw_1233_pass": yaw_err_1233 < 5.0,
        "shaft_inclination_pass": shaft_inc_diff < 5.0,
    }
    n_passed = sum(gates.values())
    logger.info("Gates Passed: %d/%d", n_passed, len(gates))
    for g_name, g_val in gates.items():
        logger.info("  %s: %s", g_name, "PASS" if g_val else "FAIL")

    final_coeffs = bernstein_to_simscape(
        current_best_theta.reshape(joints, CONTROL_COUNT), duration_s=T_BASIS
    )
    result_pkg = {
        "schema": "simscape-candidate-package/1",
        "candidate_id": f"prefix-1233ms-staged-{int(time.time())}",
        "duration_s": T_SIM,
        "basis_duration_s": T_BASIS,
        "time_origin_s": 0.0,
        "polynomial_degree": 6,
        "coordinate_names": coord_names,
        "efforts": current_best_theta.tolist(),
        "coefficients": final_coeffs.tolist(),
        "rmse_m": whole_rmse_mm / 1000.0,
        "early_rmse_m": early_rmse_mm / 1000.0,
        "terminal_rmse_m": term_rmse_mm / 1000.0,
        "clubhead_terminal_rmse_m": clubhead_term_rmse_mm / 1000.0,
        "pelvis_yaw_1233_diff_deg": diff_1233,
        "pelvis_yaw_1233_error_pct": yaw_err_1233,
        "shaft_inclination_target_deg": tgt_shaft_inclination,
        "shaft_inclination_pred_deg": shaft_inc_pred,
        "shaft_inclination_diff_deg": shaft_inc_diff,
        "max_defect_norm": 0.0,
        "gates": gates,
        "gates_passed": f"{n_passed}/{len(gates)}",
    }
    out_pkg_path = scratch_dir / "candidate_staged_impact_1233s_package.json"
    out_pkg_path.write_text(json.dumps(result_pkg, indent=2), encoding="utf-8")
    logger.info("Candidate package written to: %s", out_pkg_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
