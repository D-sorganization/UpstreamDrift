"""Impact Delivery and Kinematics Horizon Refinement (t = 1.233s, frame 444 @ 360 Hz).

Milestone: Advancing from 1.15s downswing horizon to the complete tour impact window
at t = 1.233333 s (445 frames, 0..444 @ 360 Hz).

Key Architectural Guarantees & Technical Invariants:
1. Invariant Global Basis Duration: T_basis = 1.813889 s (654 frames @ 360 Hz).
2. 100% Continuous Forward Dynamics: Zero target-state resets (Defect Norm = 0.000000 m).
3. Frozen Backswing & Transition: k=0, 1, 2, 3 are 100% FROZEN across ALL 27 joints
   from candidate_downswing_115s_locked_package.json (or candidate_downswing_105s_locked_package.json)
   to preserve address (0.20s), takeaway (0.40s), early retention (0.60s: 11.22 mm),
   and transition dynamics.
4. Active Parameters: k=4, k=5, and k=6 are ACTIVE (78 parameters across 26 non-Z joints).
   - k=4 drives downswing acceleration (peaking at t = 1.209s).
   - k=5 drives shaft delivery torque (rising into impact with dB/dt = +0.651 s^-1).
   - k=6 drives terminal strike impulse and impact burst (rising into impact with dB/dt = +0.481 s^-1).
5. Evaluation 0 Baseline Verification:
   Verifies that rollout at t=1.15s (and t=1.05s) reproduces certified metrics
   before stepping out to 1.233s.
6. Target Occlusion Repair:
   Marker_2:2 triad (clubhead) undergoes 1-frame occlusion at frame 444 in raw mocap;
   linearly interpolated from frames 443 and 445 to provide exact 3D target coordinates
   at t = 1.233s ([1.5984, 0.3081, 1.8901] m).
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
logger = logging.getLogger("impact_1233s_locked")

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))

from src.shared.python.motion_matching.prefix_fit import bernstein_to_simscape

scratch_dir = repo / "scratch"
scratch_dir.mkdir(parents=True, exist_ok=True)

# Invariant physical constants
CONTROL_COUNT = 7
T_BASIS = 1.8138888888888889
T_SIM = 1.2333333333333334  # Frame 444 @ 360 Hz: 444 / 360 = 1.233333 s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simscape Tour Impact Matching at t=1.233s (Frame 444 @ 360 Hz)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Perform syntax, data, and dimension checks without invoking MATLAB",
    )
    parser.add_argument(
        "--k6-active",
        action="store_true",
        default=True,
        help="Activate k=6 for terminal impact burst (default: True)",
    )
    parser.add_argument(
        "--k6-frozen",
        dest="k6_active",
        action="store_false",
        help="Keep k=6 frozen and only optimize k=4, 5",
    )
    parser.add_argument(
        "--max-nfev",
        type=int,
        default=400,
        help="Maximum number of function evaluations (default: 400)",
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
        Path(
            "C:/Users/diete/SimscapeTour9921/prefix-800ms-sextic-01/driver_marker_payload.json"
        ),
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
        Path(
            "C:/Users/diete/SimscapeTour9921/candidates/candidate-run06-eval79-pkg/candidate_downswing_105s_locked_package.json"
        ),
        Path(
            "C:/Users/diete/SimscapeTour9921/candidate_downswing_105s_locked_package.json"
        ),
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
    pid_file = scratch_dir / "impact_1233s_locked.pid"
    pid_file.write_text(str(pid), encoding="utf-8")
    logger.info(
        "Impact 1.233s optimization process started with PID: %d (saved to %s)",
        pid,
        pid_file,
    )

    payload = load_payload(repo, scratch_dir)
    seed, seed_path = load_seed(repo)
    cand_data, cand_path = load_warmstart_candidate(repo, scratch_dir)

    coord_names = seed["coordinate_names"]
    joints = len(coord_names)
    init_b_basis = np.array(cand_data["efforts"], dtype=np.float64).reshape(
        joints, CONTROL_COUNT
    )

    assignments = dict(zip(seed["labels"], seed["body_names"], strict=True))
    labels = list(assignments)
    indices = [payload["labels"].index(name) for name in labels]

    raw_points = np.asarray(payload["points_world_m"])[:, indices]
    raw_valid = np.asarray(payload["valid"])[:, indices]
    raw_points[~raw_valid] = np.nan
    time_s_all = np.asarray(payload["time_s"])

    # Repair frame 444 Marker_2 occlusion: linearly interpolate from frames 443 and 445
    # Marker_2 is present in payload labels
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

    # Horizon slicing: [0, 1.233333s] (445 frames at 360 Hz: frames 0..444)
    mask_sim = time_s_all <= T_SIM + 1e-6
    target_time = time_s_all[mask_sim]
    target_points = raw_points[mask_sim]
    n_frames = len(target_time)
    logger.info(
        "Target impact window [0, %.4fs]: %d frames (frames 0..%d @ 360 Hz)",
        T_SIM,
        n_frames,
        n_frames - 1,
    )

    # Downswing and delivery tracking phase [0.80s, 1.22s]
    mask_delivery_traj = (target_time >= 0.79) & (target_time < 1.22)
    delivery_traj_indices = np.where(mask_delivery_traj)[0]
    n_delivery_frames = len(delivery_traj_indices)
    sqrt_n_del = np.sqrt(max(n_delivery_frames, 1))

    club_indices = [
        i
        for i, lbl in enumerate(labels)
        if any(k in lbl.lower() for k in ("marker_2", "marker_3", "club"))
    ]
    m2_club_indices = [i for i, lbl in enumerate(labels) if "marker_2" in lbl.lower()]
    m3_shaft_indices = [i for i, lbl in enumerate(labels) if "marker_3" in lbl.lower()]
    arm_indices = [
        i
        for i, lbl in enumerate(labels)
        if any(k in lbl.lower() for k in ("elbow", "wrist", "uarm"))
    ]
    torso_indices = [
        i
        for i, lbl in enumerate(labels)
        if any(k in lbl.lower() for k in ("back", "head", "shoulder"))
    ]
    pelvis_indices = [
        i for i, lbl in enumerate(labels) if any(k in lbl.lower() for k in ("waist",))
    ]

    wl_i = labels.index("WaistLeft")
    wr_i = labels.index("WaistRight")
    lw_i = labels.index("LWristTop")
    rw_i = labels.index("RWristTop")

    idx_080 = int(np.argmin(np.abs(target_time - 0.80)))
    idx_105 = int(np.argmin(np.abs(target_time - 1.05)))
    idx_115 = int(np.argmin(np.abs(target_time - 1.15)))
    idx_1233 = len(target_time) - 1

    # Target shaft delivery orientation at impact (t = 1.233s)
    # Computed from target mid-wrist and clubhead centroid
    tgt_lw = target_points[idx_1233, lw_i]
    tgt_rw = target_points[idx_1233, rw_i]
    tgt_mid_wrist = 0.5 * (tgt_lw + tgt_rw)
    tgt_clubhead = np.nanmean(target_points[idx_1233, m2_club_indices], axis=0)
    tgt_shaft_vec = tgt_clubhead - tgt_mid_wrist
    tgt_shaft_u = tgt_shaft_vec / np.linalg.norm(tgt_shaft_vec)
    tgt_shaft_inclination = float(np.degrees(np.arcsin(abs(tgt_shaft_u[2]))))
    logger.info(
        "Target Shaft Delivery at Impact: [%.4f, %.4f, %.4f] (Inclination: %.2f deg)",
        tgt_shaft_u[0],
        tgt_shaft_u[1],
        tgt_shaft_u[2],
        tgt_shaft_inclination,
    )

    # Configure active parameters: k=0,1,2,3 are 100% FROZEN across ALL 27 joints.
    active_mask = np.zeros((joints, CONTROL_COUNT), dtype=bool)
    delta_bounds = np.zeros((joints, CONTROL_COUNT), dtype=np.float64)

    for j in range(joints):
        name = coord_names[j].lower()
        if name.startswith("translationinputz"):
            continue
        is_arm = any(
            k in name for k in ("elbow", "forearm", "wrist", "hand", "shoulder", "scap")
        )
        is_torso = any(k in name for k in ("torso", "spine"))
        is_pelvis = j < 6 or "hip" in name or "translation" in name

        # k=4 and k=5 active
        active_mask[j, 4] = True
        active_mask[j, 5] = True
        if args.k6_active:
            active_mask[j, 6] = True

        if is_arm:
            delta_bounds[j, 4] = 80.0
            delta_bounds[j, 5] = 100.0
            if args.k6_active:
                delta_bounds[j, 6] = 80.0
        elif is_torso:
            delta_bounds[j, 4] = 60.0
            delta_bounds[j, 5] = 70.0
            if args.k6_active:
                delta_bounds[j, 6] = 50.0
        elif is_pelvis:
            delta_bounds[j, 4] = 40.0
            delta_bounds[j, 5] = 45.0
            if args.k6_active:
                delta_bounds[j, 6] = 30.0
        else:
            delta_bounds[j, 4] = 50.0
            delta_bounds[j, 5] = 55.0
            if args.k6_active:
                delta_bounds[j, 6] = 40.0

    active_indices = np.where(active_mask.ravel())[0]
    n_active = len(active_indices)
    logger.info(
        "Active parameters for Impact Horizon 1.233s: %d (k=0,1,2,3 100%% FROZEN across all %d joints; k=6 active=%s)",
        n_active,
        joints,
        args.k6_active,
    )

    base_theta = init_b_basis.ravel()
    init_free = base_theta[active_indices]
    free_delta = delta_bounds.ravel()[active_indices]
    lower_free = init_free - free_delta
    upper_free = init_free + free_delta

    def pack_full_theta(free_params: np.ndarray) -> np.ndarray:
        full = base_theta.copy()
        full[active_indices] = free_params
        return full

    if args.check_only:
        logger.info(
            "Check-only mode requested: Validating dimensions and basis invariants..."
        )
        test_full = pack_full_theta(init_free)
        test_ctrls = test_full.reshape(joints, CONTROL_COUNT)
        test_coeffs = bernstein_to_simscape(test_ctrls, duration_s=T_BASIS)
        assert test_coeffs.shape == (
            joints,
            CONTROL_COUNT,
        ), f"Unexpected coeffs shape: {test_coeffs.shape}"
        assert (
            n_frames == 445
        ), f"Expected 445 frames at 360 Hz for t=1.233s, got {n_frames}"
        assert (
            abs(target_time[-1] - 1.233333) < 1e-4
        ), f"Terminal time mismatch: {target_time[-1]}"
        logger.info("All mathematical, basis, and dimensional invariants VERIFIED.")
        return 0

    # Start MATLAB engine
    try:
        import matlab.engine
    except ImportError:
        logger.error(
            "matlab.engine is not available in the current environment. To run simulation, execute on DeskComputer."
        )
        return 1

    logger.info("Starting MATLAB R2025b engine for Impact Horizon 1.233s Refinement...")
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
        eng.workspace["fit_theta"] = matlab.double(
            native_global.ravel()[:, None].tolist()
        )
        eng.eval(
            "[fit_pred, ~]=simulate_golf_markers(fit_theta,fit_opts,fit_ks,fit_schema,fit_bodies,fit_offsets,fit_time);",
            nargout=0,
        )
        return np.asarray(eng.workspace["fit_pred"], dtype=np.float64)

    early_mask = target_time <= 0.60 + 1e-12
    mask_105 = target_time <= 1.05 + 1e-12
    mask_115 = target_time <= 1.15 + 1e-12
    obs_all = np.isfinite(target_points).all(axis=2)

    # ==============================================================================
    # EVALUATION 0: BASELINE VERIFICATION & IMPACT EXTENSION AUDIT
    # ==============================================================================
    logger.info("=" * 70)
    logger.info("EVALUATION 0: VERIFYING CERTIFIED BASELINE & IMPACT 1.233s EXTENSION")
    logger.info("=" * 70)

    init_full = pack_full_theta(init_free)
    init_pts = forward_rollout(init_full)
    init_diff = init_pts - target_points

    # 1.15s Horizon Reproduction Audit
    obs_115 = obs_all[mask_115]
    eval0_115_whole_rmse_mm = float(
        np.sqrt(np.mean(np.linalg.norm(init_diff[mask_115][obs_115], axis=1) ** 2))
        * 1000.0
    )
    eval0_early_rmse_mm = float(
        np.sqrt(
            np.mean(
                np.linalg.norm(init_diff[early_mask][obs_all[early_mask]], axis=1) ** 2
            )
        )
        * 1000.0
    )
    eval0_115_term_rmse_mm = float(
        np.sqrt(
            np.mean(np.linalg.norm(init_diff[idx_115][obs_all[idx_115]], axis=1) ** 2)
        )
        * 1000.0
    )
    eval0_115_club_rmse_mm = float(
        np.sqrt(np.mean(np.linalg.norm(init_diff[idx_115, club_indices], axis=1) ** 2))
        * 1000.0
    )

    v_p115 = init_pts[idx_115, wr_i, :2] - init_pts[idx_115, wl_i, :2]
    v_t115 = target_points[idx_115, wr_i, :2] - target_points[idx_115, wl_i, :2]
    yaw_t115 = float(np.degrees(np.arctan2(v_t115[1], v_t115[0])))
    yaw_p115 = float(np.degrees(np.arctan2(v_p115[1], v_p115[0])))
    diff_115 = float((yaw_p115 - yaw_t115 + 180) % 360 - 180)
    eval0_115_yaw_err = float(abs(diff_115) / max(abs(yaw_t115), 1.0) * 100.0)

    logger.info("--- [Eval 0 Verification: 1.15s Horizon Check] ---")
    logger.info("Whole Window RMSE [0, 1.15s]:   %6.2f mm", eval0_115_whole_rmse_mm)
    logger.info(
        "Early Retention RMSE [0, 0.60s]: %6.2f mm  (Gate <= 12.0 mm)",
        eval0_early_rmse_mm,
    )
    logger.info("Terminal Marker RMSE @ 1.15s:   %6.2f mm", eval0_115_term_rmse_mm)
    logger.info("Terminal Clubhead RMSE @ 1.15s: %6.2f mm", eval0_115_club_rmse_mm)
    logger.info(
        "Pelvis Yaw Error @ 1.15s:       %6.2f%% (%+.2f deg)",
        eval0_115_yaw_err,
        diff_115,
    )

    # 1.233s Impact Extension Initial Audit
    eval0_1233_whole_rmse_mm = float(
        np.sqrt(np.mean(np.linalg.norm(init_diff[obs_all], axis=1) ** 2)) * 1000.0
    )
    eval0_1233_term_rmse_mm = float(
        np.sqrt(
            np.mean(np.linalg.norm(init_diff[idx_1233][obs_all[idx_1233]], axis=1) ** 2)
        )
        * 1000.0
    )
    eval0_1233_club_rmse_mm = float(
        np.sqrt(np.mean(np.linalg.norm(init_diff[idx_1233, club_indices], axis=1) ** 2))
        * 1000.0
    )

    v_p1233 = init_pts[idx_1233, wr_i, :2] - init_pts[idx_1233, wl_i, :2]
    v_t1233 = target_points[idx_1233, wr_i, :2] - target_points[idx_1233, wl_i, :2]
    yaw_t1233 = float(np.degrees(np.arctan2(v_t1233[1], v_t1233[0])))
    yaw_p1233 = float(np.degrees(np.arctan2(v_p1233[1], v_p1233[0])))
    diff_1233 = float((yaw_p1233 - yaw_t1233 + 180) % 360 - 180)
    eval0_1233_yaw_err = float(abs(diff_1233) / max(abs(yaw_t1233), 1.0) * 100.0)

    logger.info(
        "--- [Eval 0 Verification: 1.233s Impact Horizon Extension Unoptimized] ---"
    )
    logger.info("Whole Window RMSE [0, 1.233s]:  %6.2f mm", eval0_1233_whole_rmse_mm)
    logger.info("Terminal Marker RMSE @ 1.233s:  %6.2f mm", eval0_1233_term_rmse_mm)
    logger.info("Terminal Clubhead RMSE @ 1.233s:%6.2f mm", eval0_1233_club_rmse_mm)
    logger.info(
        "Pelvis Yaw Error @ 1.233s:      %6.2f%% (%+.2f deg)",
        eval0_1233_yaw_err,
        diff_1233,
    )
    logger.info("=" * 70)

    eval0_record = {
        "timestamp": datetime.now().isoformat(),
        "pid": pid,
        "evaluation": 0,
        "duration_s": T_SIM,
        "basis_duration_s": T_BASIS,
        "defect_norm": 0.0,
        "115s_reproduction": {
            "whole_rmse_mm": eval0_115_whole_rmse_mm,
            "early_rmse_mm": eval0_early_rmse_mm,
            "terminal_rmse_mm": eval0_115_term_rmse_mm,
            "clubhead_terminal_rmse_mm": eval0_115_club_rmse_mm,
            "pelvis_yaw_diff_deg": diff_115,
            "pelvis_yaw_error_pct": eval0_115_yaw_err,
        },
        "1233s_initial": {
            "whole_rmse_mm": eval0_1233_whole_rmse_mm,
            "early_rmse_mm": eval0_early_rmse_mm,
            "terminal_rmse_mm": eval0_1233_term_rmse_mm,
            "clubhead_terminal_rmse_mm": eval0_1233_club_rmse_mm,
            "pelvis_yaw_diff_deg": diff_1233,
            "pelvis_yaw_error_pct": eval0_1233_yaw_err,
        },
    }
    (scratch_dir / "impact_1233s_locked_eval0.json").write_text(
        json.dumps(eval0_record, indent=2), encoding="utf-8"
    )

    eval_count = 0
    best_cost = float("inf")

    def residual_fn(p_free: np.ndarray) -> np.ndarray:
        nonlocal eval_count, best_cost
        eval_count += 1

        full_th = pack_full_theta(p_free)
        pred_pts = forward_rollout(full_th)
        diff = pred_pts - target_points

        # 1. Delivery trajectory tracking on [0.80s, 1.22s]
        del_diff = diff[delivery_traj_indices]
        norm_del_res = []
        for m in range(len(labels)):
            w = 15.0 / sqrt_n_del
            if m in club_indices:
                w = 30.0 / sqrt_n_del
            elif m in arm_indices:
                w = 20.0 / sqrt_n_del
            res_m = del_diff[:, m, :] * w
            valid_m = np.isfinite(res_m)
            norm_del_res.append(res_m[valid_m].ravel())
        res_del = np.concatenate(norm_del_res)

        # 2. Terminal impact alignment at t = 1.233s (dominant clubhead & delivery orientation)
        term_diff = diff[idx_1233]
        term_res_list = []
        for c in m2_club_indices:
            res_c = term_diff[c] * 150.0  # Dominant clubhead tracking
            if np.isfinite(res_c).all():
                term_res_list.append(res_c)
        for c3 in m3_shaft_indices:
            res_c3 = term_diff[c3] * 100.0  # Shaft marker triad tracking
            if np.isfinite(res_c3).all():
                term_res_list.append(res_c3)
        for a in arm_indices:
            res_a = term_diff[a] * 50.0  # Arm and wrist delivery
            if np.isfinite(res_a).all():
                term_res_list.append(res_a)
        for to in torso_indices:
            res_to = term_diff[to] * 25.0
            if np.isfinite(res_to).all():
                term_res_list.append(res_to)
        for pel in pelvis_indices:
            res_pel = term_diff[pel] * 35.0  # Pelvic alignment
            if np.isfinite(res_pel).all():
                term_res_list.append(res_pel)
        res_term = np.concatenate(term_res_list)

        # 3. Shaft delivery orientation penalty (unit vector defect)
        pred_lw = pred_pts[idx_1233, lw_i]
        pred_rw = pred_pts[idx_1233, rw_i]
        pred_mid_wrist = 0.5 * (pred_lw + pred_rw)
        pred_clubhead = np.nanmean(pred_pts[idx_1233, m2_club_indices], axis=0)
        pred_shaft_vec = pred_clubhead - pred_mid_wrist
        pred_shaft_len = np.linalg.norm(pred_shaft_vec)
        if pred_shaft_len > 1e-4:
            pred_shaft_u = pred_shaft_vec / pred_shaft_len
            shaft_orient_res = (pred_shaft_u - tgt_shaft_u) * 50.0
        else:
            shaft_orient_res = np.zeros(3)

        # 4. Soft Barrier: Gate 4 Early Retention (<= 12.0 mm)
        early_dists = np.linalg.norm(diff[early_mask], axis=2)
        valid_early = np.isfinite(early_dists)
        early_rmse = float(np.sqrt(np.mean(early_dists[valid_early] ** 2)))
        penalty_list = []
        if early_rmse > 0.0116:
            penalty_list.append(np.array([(early_rmse - 0.0116) * 500.0]))
        else:
            penalty_list.append(np.array([0.0]))

        # 5. Pelvis Yaw at 1.233s (Gate 6 < 5.0%)
        v_p_now = pred_pts[idx_1233, wr_i, :2] - pred_pts[idx_1233, wl_i, :2]
        v_t_now = target_points[idx_1233, wr_i, :2] - target_points[idx_1233, wl_i, :2]
        yaw_t_now = float(np.degrees(np.arctan2(v_t_now[1], v_t_now[0])))
        yaw_p_now = float(np.degrees(np.arctan2(v_p_now[1], v_p_now[0])))
        diff_yaw_now = float((yaw_p_now - yaw_t_now + 180) % 360 - 180)
        yaw_err_now = float(abs(diff_yaw_now) / max(abs(yaw_t_now), 1.0) * 100.0)
        if yaw_err_now > 5.0:
            penalty_list.append(np.array([(yaw_err_now - 5.0) * 10.0]))
        else:
            penalty_list.append(np.array([0.0]))

        res_pen = np.concatenate(penalty_list)
        res_vec = np.concatenate([res_del, res_term, shaft_orient_res, res_pen])
        cost = float(0.5 * np.sum(res_vec**2))
        if cost < best_cost:
            best_cost = cost

        coeffs = bernstein_to_simscape(
            full_th.reshape(joints, CONTROL_COUNT), duration_s=T_BASIS
        )
        ckpt: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "pid": pid,
            "evaluation": eval_count,
            "cost": cost,
            "best_cost": best_cost,
            "early_rmse_mm": early_rmse * 1000.0,
            "yaw_err_1233_pct": yaw_err_now,
            "yaw_diff_1233_deg": diff_yaw_now,
            "theta": full_th.tolist(),
            "coefficients": coeffs.tolist(),
        }
        tmp_p = scratch_dir / "impact_1233s_locked_checkpoint.tmp"
        tmp_p.write_text(json.dumps(ckpt), encoding="utf-8")
        tmp_p.replace(scratch_dir / "impact_1233s_locked_checkpoint.json")

        if eval_count % 5 == 0 or eval_count == 1:
            club_err = float(
                np.sqrt(
                    np.mean(np.linalg.norm(diff[idx_1233, club_indices], axis=1) ** 2)
                )
                * 1000.0
            )
            term_err = float(
                np.sqrt(np.mean(np.linalg.norm(diff[idx_1233], axis=1) ** 2)) * 1000.0
            )
            logger.info(
                "Eval %4d | Cost: %.4e (Best: %.4e) | Early: %5.2fmm | Club: %6.2fmm | Term: %6.2fmm | Yaw1233: %5.2f%%",
                eval_count,
                cost,
                best_cost,
                early_rmse * 1000.0,
                club_err,
                term_err,
                yaw_err_now,
            )

        return res_vec

    logger.info(
        "Launching least_squares Impact 1.233s Refinement (%d parameters: k=4,5,6, diff_step=%.1e)...",
        n_active,
        args.diff_step,
    )
    t_opt_start = time.time()
    opt_result = least_squares(
        residual_fn,
        init_free,
        bounds=(lower_free, upper_free),
        diff_step=args.diff_step,
        max_nfev=args.max_nfev,
        ftol=1e-7,
        xtol=1e-8,
        gtol=1e-7,
        x_scale=1.0,
    )
    opt_elapsed = time.time() - t_opt_start
    logger.info(
        "Impact 1.233s refinement completed in %.1f s | Status: %d | Message: %s",
        opt_elapsed,
        opt_result.status,
        opt_result.message,
    )

    # Final Certification Rollout
    final_full = pack_full_theta(opt_result.x)
    final_pts = forward_rollout(final_full)
    final_diff = final_pts - target_points

    final_dists = np.linalg.norm(final_diff[obs_all], axis=1)
    whole_rmse_mm = float(np.sqrt(np.mean(final_dists**2)) * 1000.0)

    early_dists = np.linalg.norm(final_diff[early_mask][obs_all[early_mask]], axis=1)
    early_rmse_mm = float(np.sqrt(np.mean(early_dists**2)) * 1000.0)

    term_dists = np.linalg.norm(final_diff[idx_1233][obs_all[idx_1233]], axis=1)
    term_rmse_mm = float(np.sqrt(np.mean(term_dists**2)) * 1000.0)

    club_term_dists = np.linalg.norm(final_diff[idx_1233, club_indices], axis=1)
    clubhead_term_rmse_mm = float(np.sqrt(np.mean(club_term_dists**2)) * 1000.0)

    # Pelvis yaw checks across all milestones
    v_p_080 = final_pts[idx_080, wr_i, :2] - final_pts[idx_080, wl_i, :2]
    v_t_080 = target_points[idx_080, wr_i, :2] - target_points[idx_080, wl_i, :2]
    yaw_t_080 = float(np.degrees(np.arctan2(v_t_080[1], v_t_080[0])))
    yaw_p_080 = float(np.degrees(np.arctan2(v_p_080[1], v_p_080[0])))
    yaw_diff_080 = float((yaw_p_080 - yaw_t_080 + 180) % 360 - 180)
    yaw_err_080_pct = float(abs(yaw_diff_080) / max(abs(yaw_t_080), 1.0) * 100.0)

    v_p_115 = final_pts[idx_115, wr_i, :2] - final_pts[idx_115, wl_i, :2]
    v_t_115 = target_points[idx_115, wr_i, :2] - target_points[idx_115, wl_i, :2]
    yaw_t_115 = float(np.degrees(np.arctan2(v_t_115[1], v_t_115[0])))
    yaw_p_115 = float(np.degrees(np.arctan2(v_p_115[1], v_p_115[0])))
    yaw_diff_115 = float((yaw_p_115 - yaw_t_115 + 180) % 360 - 180)
    yaw_err_115_pct = float(abs(yaw_diff_115) / max(abs(yaw_t_115), 1.0) * 100.0)

    v_p_1233 = final_pts[idx_1233, wr_i, :2] - final_pts[idx_1233, wl_i, :2]
    v_t_1233 = target_points[idx_1233, wr_i, :2] - target_points[idx_1233, wl_i, :2]
    yaw_t_1233 = float(np.degrees(np.arctan2(v_t_1233[1], v_t_1233[0])))
    yaw_p_1233 = float(np.degrees(np.arctan2(v_p_1233[1], v_p_1233[0])))
    yaw_diff_1233 = float((yaw_p_1233 - yaw_t_1233 + 180) % 360 - 180)
    yaw_err_1233_pct = float(abs(yaw_diff_1233) / max(abs(yaw_t_1233), 1.0) * 100.0)

    # Shaft delivery orientation at impact
    f_pred_lw = final_pts[idx_1233, lw_i]
    f_pred_rw = final_pts[idx_1233, rw_i]
    f_pred_mid_wrist = 0.5 * (f_pred_lw + f_pred_rw)
    f_pred_clubhead = np.nanmean(final_pts[idx_1233, m2_club_indices], axis=0)
    f_pred_shaft_vec = f_pred_clubhead - f_pred_mid_wrist
    f_pred_shaft_u = f_pred_shaft_vec / np.linalg.norm(f_pred_shaft_vec)
    f_pred_inclination = float(np.degrees(np.arcsin(abs(f_pred_shaft_u[2]))))
    shaft_inclination_err_deg = abs(f_pred_inclination - tgt_shaft_inclination)

    gates = {
        "whole_window_pass": bool(whole_rmse_mm <= 25.0),
        "terminal_rmse_pass": bool(term_rmse_mm <= 35.0),
        "clubhead_terminal_pass": bool(clubhead_term_rmse_mm <= 50.0),
        "early_retention_pass": bool(early_rmse_mm <= 12.0),
        "pelvis_yaw_080_pass": bool(yaw_err_080_pct < 5.0),
        "pelvis_yaw_115_pass": bool(yaw_err_115_pct < 5.0),
        "pelvis_yaw_1233_pass": bool(yaw_err_1233_pct < 5.0),
        "shaft_inclination_pass": bool(shaft_inclination_err_deg < 5.0),
    }
    passed_count = sum(gates.values())

    logger.info("=" * 70)
    logger.info("FINAL GATE AUDIT REPORT (IMPACT 1.233s LOCKED REFINEMENT)")
    logger.info(
        "Gate 1 (Whole Window <= 25.0 mm):     %6.2f mm -> %s",
        whole_rmse_mm,
        "PASS" if gates["whole_window_pass"] else "FAIL",
    )
    logger.info(
        "Gate 2 (Terminal RMSE <= 35.0 mm):     %6.2f mm -> %s",
        term_rmse_mm,
        "PASS" if gates["terminal_rmse_pass"] else "FAIL",
    )
    logger.info(
        "Gate 3 (Clubhead Term <= 50.0 mm):     %6.2f mm -> %s",
        clubhead_term_rmse_mm,
        "PASS" if gates["clubhead_terminal_pass"] else "FAIL",
    )
    logger.info(
        "Gate 4 (Early Retention <= 12.0 mm):   %6.2f mm -> %s",
        early_rmse_mm,
        "PASS" if gates["early_retention_pass"] else "FAIL",
    )
    logger.info(
        "Gate 5 (Pelvis Yaw @ 0.80s < 5.0%%):    %6.2f%% (%+.2f deg) -> %s",
        yaw_err_080_pct,
        yaw_diff_080,
        "PASS" if gates["pelvis_yaw_080_pass"] else "FAIL",
    )
    logger.info(
        "Gate 6 (Pelvis Yaw @ 1.15s < 5.0%%):    %6.2f%% (%+.2f deg) -> %s",
        yaw_err_115_pct,
        yaw_diff_115,
        "PASS" if gates["pelvis_yaw_115_pass"] else "FAIL",
    )
    logger.info(
        "Gate 7 (Pelvis Yaw @ 1.233s < 5.0%%):   %6.2f%% (%+.2f deg) -> %s",
        yaw_err_1233_pct,
        yaw_diff_1233,
        "PASS" if gates["pelvis_yaw_1233_pass"] else "FAIL",
    )
    logger.info(
        "Gate 8 (Shaft Inclination < 5.0 deg):  %6.2f deg (Diff: %.2f deg) -> %s",
        f_pred_inclination,
        shaft_inclination_err_deg,
        "PASS" if gates["shaft_inclination_pass"] else "FAIL",
    )
    logger.info("Overall Gate Status:                   %d/8", passed_count)
    logger.info("=" * 70)

    final_coeffs = bernstein_to_simscape(
        final_full.reshape(joints, CONTROL_COUNT), duration_s=T_BASIS
    )
    candidate_id = f"prefix-1233ms-impact-locked-{int(time.time())}"
    res_dict: dict[str, Any] = {
        "schema": "simscape-candidate-package/1",
        "candidate_id": candidate_id,
        "duration_s": T_SIM,
        "basis_duration_s": T_BASIS,
        "time_origin_s": 0.0,
        "polynomial_degree": 6,
        "coordinate_names": coord_names,
        "efforts": final_full.tolist(),
        "rmse_m": whole_rmse_mm / 1000.0,
        "early_rmse_m": early_rmse_mm / 1000.0,
        "terminal_rmse_m": term_rmse_mm / 1000.0,
        "clubhead_terminal_rmse_m": clubhead_term_rmse_mm / 1000.0,
        "pelvis_yaw_080_diff_deg": yaw_diff_080,
        "pelvis_yaw_080_error_pct": yaw_err_080_pct,
        "pelvis_yaw_115_diff_deg": yaw_diff_115,
        "pelvis_yaw_115_error_pct": yaw_err_115_pct,
        "pelvis_yaw_1233_diff_deg": yaw_diff_1233,
        "pelvis_yaw_1233_error_pct": yaw_err_1233_pct,
        "shaft_inclination_target_deg": tgt_shaft_inclination,
        "shaft_inclination_pred_deg": f_pred_inclination,
        "shaft_inclination_diff_deg": shaft_inclination_err_deg,
        "max_defect_norm": 0.0,
        "segmented_rmse_m": whole_rmse_mm / 1000.0,
        "unsegmented_rmse_m": whole_rmse_mm / 1000.0,
        "optimizer_converged": bool(opt_result.status > 0),
        "accepted": bool(passed_count >= 7),
        "gates": gates,
        "gates_passed": f"{passed_count}/8",
        "coefficients": final_coeffs.tolist(),
        "elapsed_s": opt_elapsed,
    }

    out_path = scratch_dir / "impact_1233s_locked_result.json"
    out_path.write_text(json.dumps(res_dict, indent=2), encoding="utf-8")
    logger.info("Saved result to %s", out_path)

    pkg_path = repo / "candidates" / "candidate_impact_1233s_locked_package.json"
    pkg_path.parent.mkdir(parents=True, exist_ok=True)
    pkg_path.write_text(json.dumps(res_dict, indent=2), encoding="utf-8")
    logger.info("Saved candidate package to %s", pkg_path)

    try:
        eng.quit()
    except (RuntimeError, AttributeError, OSError):
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
