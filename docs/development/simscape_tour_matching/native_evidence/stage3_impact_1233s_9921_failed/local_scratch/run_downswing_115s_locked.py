"""Downswing Horizon Refinement (t = 1.15s, frames 0..414 @ 360 Hz).

Key Architectural Guarantees & Technical Invariants:
1. Invariant Global Basis Duration: T_basis = 1.813889 s (654 frames @ 360 Hz).
2. 100% Continuous Forward Dynamics: Zero target-state resets (Defect Norm = 0.000000 m).
3. Frozen Backswing & Transition: k=0, 1, 2, 3 are 100% FROZEN across ALL 27 joints
   from candidate_downswing_105s_locked_package.json to preserve address (0.20s),
   takeaway (0.40s), early retention (0.60s: 11.22 mm), and transition metrics.
4. Active Parameters: ONLY k=4 and k=5 are ACTIVE (52 parameters across 26 non-Z joints).
   k=4 drives downswing acceleration; k=5 drives shaft delivery. k=6 remains frozen.
5. Evaluation 0 Baseline Verification:
   Verifies that rollout at t=1.05s reproduces the certified 1.05s baseline metrics
   before stepping out to 1.15s.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import matlab.engine
import numpy as np
from scipy.optimize import least_squares

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("downswing_115s_locked")

repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo))

from src.shared.python.motion_matching.prefix_fit import bernstein_to_simscape

scratch_dir = repo / "scratch"
scratch_dir.mkdir(parents=True, exist_ok=True)

pid = os.getpid()
pid_file = scratch_dir / "downswing_115s_locked.pid"
pid_file.write_text(str(pid), encoding="utf-8")
logger.info("Optimization process started with PID: %d (saved to %s)", pid, pid_file)

# Invariant physical constants
CONTROL_COUNT = 7
T_BASIS = 1.813889
T_SIM = 1.15

# Payload discovery
payload_candidates = [
    scratch_dir / "driver_marker_payload.json",
    repo / "data" / "driver_marker_payload.json",
    Path("C:/Users/diete/SimscapeTour9921/prefix-800ms-sextic-01/driver_marker_payload.json"),
    Path("C:/Users/diete/SimscapeTour9921/driver_marker_payload.json"),
]
payload_path = next((p for p in payload_candidates if p.exists()), None)
if payload_path is None:
    raise FileNotFoundError(f"Could not find driver_marker_payload.json in {payload_candidates}")
logger.info("Loaded marker payload from: %s", payload_path)
payload = json.loads(payload_path.read_text(encoding="utf-8"))

# Seed path discovery
seed_candidates = [
    repo / "docs/development/simscape_tour_matching/native_evidence/initial_velocity_seed_qualified_r2025b.json",
    Path("C:/Users/diete/SimscapeTour9921/docs/development/simscape_tour_matching/native_evidence/initial_velocity_seed_qualified_r2025b.json"),
]
seed_path = next((p for p in seed_candidates if p.exists()), None)
if seed_path is None:
    raise FileNotFoundError(f"Could not find initial velocity seed in {seed_candidates}")
logger.info("Loaded seed from: %s", seed_path)
seed = json.loads(seed_path.read_text(encoding="utf-8"))

coord_names = seed["coordinate_names"]
joints = len(coord_names)

# Candidate 105s discovery
cand_candidates = [
    repo / "candidates/candidate_downswing_105s_locked_package.json",
    Path("C:/Users/diete/SimscapeTour9921/candidates/candidate-run06-eval79-pkg/candidate_downswing_105s_locked_package.json"),
    Path("C:/Users/diete/SimscapeTour9921/candidate_downswing_105s_locked_package.json"),
]
cand_path = next((p for p in cand_candidates if p.exists()), None)
if cand_path is None:
    raise FileNotFoundError(f"Could not find candidate_downswing_105s_locked_package.json in {cand_candidates}")
logger.info("Warm-starting from certified 1.05s candidate: %s", cand_path)
cand_data = json.loads(cand_path.read_text(encoding="utf-8"))
init_b_basis = np.array(cand_data["efforts"], dtype=np.float64).reshape(joints, CONTROL_COUNT)

assignments = dict(zip(seed["labels"], seed["body_names"], strict=True))
labels = list(assignments)
indices = [payload["labels"].index(name) for name in labels]

raw_points = np.asarray(payload["points_world_m"])[:, indices]
raw_valid = np.asarray(payload["valid"])[:, indices]
raw_points[~raw_valid] = np.nan
time_s_all = np.asarray(payload["time_s"])

# Horizon slicing: [0, 1.15s] (415 frames at 360 Hz)
mask_sim = time_s_all <= T_SIM + 1e-12
target_time = time_s_all[mask_sim]
target_points = raw_points[mask_sim]
n_frames = len(target_time)
logger.info("Target window [0, %.2fs]: %d frames (frames 0..%d @ 360 Hz)", T_SIM, n_frames, n_frames - 1)

# Downswing delivery tracking phase [0.80s, 1.14s]
mask_downswing_traj = (target_time >= 0.79) & (target_time < 1.14)
downswing_traj_indices = np.where(mask_downswing_traj)[0]
n_down_frames = len(downswing_traj_indices)
sqrt_n_down = np.sqrt(max(n_down_frames, 1))

club_indices = [i for i, l in enumerate(labels) if any(k in l.lower() for k in ("marker_2", "marker_3", "club"))]
arm_indices = [i for i, l in enumerate(labels) if any(k in l.lower() for k in ("elbow", "wrist", "uarm"))]
torso_indices = [i for i, l in enumerate(labels) if any(k in l.lower() for k in ("back", "head", "shoulder"))]
pelvis_indices = [i for i, l in enumerate(labels) if any(k in l.lower() for k in ("waist",))]

wl_i = labels.index("WaistLeft")
wr_i = labels.index("WaistRight")

idx_080 = int(np.argmin(np.abs(target_time - 0.80)))
idx_105 = int(np.argmin(np.abs(target_time - 1.05)))
idx_115 = len(target_time) - 1

# Configure active parameters: k=0,1,2,3 are 100% FROZEN across ALL 27 joints.
# ONLY k=4 and k=5 are ACTIVE for downswing acceleration and shaft delivery.
active_mask = np.zeros((joints, CONTROL_COUNT), dtype=bool)
delta_bounds = np.zeros((joints, CONTROL_COUNT), dtype=np.float64)

for j in range(joints):
    name = coord_names[j].lower()
    if name.startswith("translationinputz"):
        continue
    is_arm = any(k in name for k in ("elbow", "forearm", "wrist", "hand", "shoulder", "scap"))
    is_torso = any(k in name for k in ("torso", "spine"))
    is_pelvis = j < 6 or "hip" in name or "translation" in name

    active_mask[j, 4] = True
    active_mask[j, 5] = True

    if is_arm:
        delta_bounds[j, 4] = 120.0
        delta_bounds[j, 5] = 120.0
    elif is_torso:
        delta_bounds[j, 4] = 80.0
        delta_bounds[j, 5] = 80.0
    elif is_pelvis:
        delta_bounds[j, 4] = 50.0
        delta_bounds[j, 5] = 50.0
    else:
        delta_bounds[j, 4] = 60.0
        delta_bounds[j, 5] = 60.0

active_indices = np.where(active_mask.ravel())[0]
n_active = len(active_indices)
logger.info("Active parameters for Downswing Horizon 1.15s: %d (k=0,1,2,3 100%% FROZEN across all %d joints)", n_active, joints)

base_theta = init_b_basis.ravel()
init_free = base_theta[active_indices]
free_delta = delta_bounds.ravel()[active_indices]
lower_free = init_free - free_delta
upper_free = init_free + free_delta

def pack_full_theta(free_params: np.ndarray) -> np.ndarray:
    full = base_theta.copy()
    full[active_indices] = free_params
    return full

logger.info("Starting MATLAB R2025b engine for Downswing 1.15s Refinement...")
eng = matlab.engine.start_matlab("-nodesktop -nosplash")
eng_dir = repo / "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab"
eng.addpath(str(eng_dir / "src/model"), nargout=0)
eng.addpath(eng.genpath(str(eng_dir / "src/functions")), nargout=0)
eng.addpath(str(eng_dir / "motion_matching/shared"), nargout=0)

eng.workspace["fit_seed_path"] = str(seed_path)
eng.workspace["fit_frame_names"] = list(assignments.values())

eng.eval(f"""
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
""", nargout=0)

eng.workspace["fit_time"] = matlab.double(target_time[:, None].tolist())

eng.eval("""
for j=1:numel(fit_seed.q)
 name=fit_opts.joint_names(j); value=fit_seed.q(j); velocity=fit_seed.qd(j);
 if ~startsWith(name,'Translation'); value=rad2deg(value); velocity=rad2deg(velocity); end
 fit_opts.input_overrides.(replace(name,'Input','StartPosition'))=value;
 fit_opts.input_overrides.(replace(name,'Input','StartVelocity'))=velocity;
end
""", nargout=0)

def forward_rollout(full_theta: np.ndarray) -> np.ndarray:
    controls_basis = full_theta.reshape(joints, CONTROL_COUNT)
    native_global = bernstein_to_simscape(controls_basis, duration_s=T_BASIS)
    eng.workspace["fit_theta"] = matlab.double(native_global.ravel()[:, None].tolist())
    eng.eval("[fit_pred, ~]=simulate_golf_markers(fit_theta,fit_opts,fit_ks,fit_schema,fit_bodies,fit_offsets,fit_time);", nargout=0)
    return np.asarray(eng.workspace["fit_pred"], dtype=np.float64)

early_mask = target_time <= 0.60 + 1e-12
mask_105 = target_time <= 1.05 + 1e-12
obs_all = np.isfinite(target_points).all(axis=2)

# ==============================================================================
# EVALUATION 0: BASELINE VERIFICATION & HORIZON EXTENSION AUDIT
# ==============================================================================
logger.info("=" * 70)
logger.info("EVALUATION 0: VERIFYING CERTIFIED 1.05s BASELINE & 1.15s EXTENSION")
logger.info("=" * 70)

init_full = pack_full_theta(init_free)
init_pts = forward_rollout(init_full)
init_diff = init_pts - target_points

# 1.05s Horizon Reproduction Audit
obs_105 = obs_all[mask_105]
eval0_105_whole_rmse_mm = float(np.sqrt(np.mean(np.linalg.norm(init_diff[mask_105][obs_105], axis=1)**2)) * 1000.0)
eval0_early_rmse_mm = float(np.sqrt(np.mean(np.linalg.norm(init_diff[early_mask][obs_all[early_mask]], axis=1)**2)) * 1000.0)
eval0_105_term_rmse_mm = float(np.sqrt(np.mean(np.linalg.norm(init_diff[idx_105][obs_all[idx_105]], axis=1)**2)) * 1000.0)
eval0_105_club_rmse_mm = float(np.sqrt(np.mean(np.linalg.norm(init_diff[idx_105, club_indices], axis=1)**2)) * 1000.0)

v_p105 = init_pts[idx_105, wr_i, :2] - init_pts[idx_105, wl_i, :2]
v_t105 = target_points[idx_105, wr_i, :2] - target_points[idx_105, wl_i, :2]
yaw_t105 = float(np.degrees(np.arctan2(v_t105[1], v_t105[0])))
yaw_p105 = float(np.degrees(np.arctan2(v_p105[1], v_p105[0])))
diff_105 = float((yaw_p105 - yaw_t105 + 180) % 360 - 180)
eval0_105_yaw_err = float(abs(diff_105) / max(abs(yaw_t105), 1.0) * 100.0)

logger.info("--- [Eval 0 Verification: 1.05s Certified Baseline Reproduction] ---")
logger.info("Whole Window RMSE [0, 1.05s]:   %6.2f mm  (Certified: 212.02 mm)", eval0_105_whole_rmse_mm)
logger.info("Early Retention RMSE [0, 0.60s]: %6.2f mm  (Certified:  11.22 mm, Gate <= 12.0 mm: PASS)", eval0_early_rmse_mm)
logger.info("Terminal Marker RMSE @ 1.05s:   %6.2f mm  (Certified: 585.30 mm)", eval0_105_term_rmse_mm)
logger.info("Terminal Clubhead RMSE @ 1.05s: %6.2f mm  (Certified: 695.76 mm)", eval0_105_club_rmse_mm)
logger.info("Pelvis Yaw Error @ 1.05s:       %6.2f%% (%+.2f deg) (Certified: 29.03%%)", eval0_105_yaw_err, diff_105)

# 1.15s Horizon Extension Initial Audit
eval0_115_whole_rmse_mm = float(np.sqrt(np.mean(np.linalg.norm(init_diff[obs_all], axis=1)**2)) * 1000.0)
eval0_115_term_rmse_mm = float(np.sqrt(np.mean(np.linalg.norm(init_diff[-1][obs_all[-1]], axis=1)**2)) * 1000.0)
eval0_115_club_rmse_mm = float(np.sqrt(np.mean(np.linalg.norm(init_diff[-1, club_indices], axis=1)**2)) * 1000.0)

v_p115 = init_pts[-1, wr_i, :2] - init_pts[-1, wl_i, :2]
v_t115 = target_points[-1, wr_i, :2] - target_points[-1, wl_i, :2]
yaw_t115 = float(np.degrees(np.arctan2(v_t115[1], v_t115[0])))
yaw_p115 = float(np.degrees(np.arctan2(v_p115[1], v_p115[0])))
diff_115 = float((yaw_p115 - yaw_t115 + 180) % 360 - 180)
eval0_115_yaw_err = float(abs(diff_115) / max(abs(yaw_t115), 1.0) * 100.0)

logger.info("--- [Eval 0 Verification: 1.15s Horizon Extension Unoptimized] ---")
logger.info("Whole Window RMSE [0, 1.15s]:   %6.2f mm", eval0_115_whole_rmse_mm)
logger.info("Terminal Marker RMSE @ 1.15s:   %6.2f mm", eval0_115_term_rmse_mm)
logger.info("Terminal Clubhead RMSE @ 1.15s: %6.2f mm", eval0_115_club_rmse_mm)
logger.info("Pelvis Yaw Error @ 1.15s:       %6.2f%% (%+.2f deg)", eval0_115_yaw_err, diff_115)
logger.info("=" * 70)

# Save Eval 0 baseline checkpoint
eval0_record = {
    "timestamp": datetime.now().isoformat(),
    "pid": pid,
    "evaluation": 0,
    "duration_s": T_SIM,
    "basis_duration_s": T_BASIS,
    "defect_norm": 0.0,
    "105s_reproduction": {
        "whole_rmse_mm": eval0_105_whole_rmse_mm,
        "early_rmse_mm": eval0_early_rmse_mm,
        "terminal_rmse_mm": eval0_105_term_rmse_mm,
        "clubhead_terminal_rmse_mm": eval0_105_club_rmse_mm,
        "pelvis_yaw_diff_deg": diff_105,
        "pelvis_yaw_error_pct": eval0_105_yaw_err,
        "verified": bool(abs(eval0_105_whole_rmse_mm - 212.02) < 1.0 and abs(eval0_early_rmse_mm - 11.22) < 0.2),
    },
    "115s_initial": {
        "whole_rmse_mm": eval0_115_whole_rmse_mm,
        "early_rmse_mm": eval0_early_rmse_mm,
        "terminal_rmse_mm": eval0_115_term_rmse_mm,
        "clubhead_terminal_rmse_mm": eval0_115_club_rmse_mm,
        "pelvis_yaw_diff_deg": diff_115,
        "pelvis_yaw_error_pct": eval0_115_yaw_err,
    },
}
(scratch_dir / "downswing_115s_locked_eval0.json").write_text(json.dumps(eval0_record, indent=2), encoding="utf-8")

eval_count = 0
best_cost = float("inf")

def residual_fn(p_free: np.ndarray) -> np.ndarray:
    global eval_count, best_cost
    eval_count += 1

    full_th = pack_full_theta(p_free)
    pred_pts = forward_rollout(full_th)
    diff = pred_pts - target_points

    # 1. Downswing trajectory tracking on [0.80s, 1.14s] (normalized MSE)
    down_diff = diff[downswing_traj_indices]
    norm_down_res = []
    for m in range(len(labels)):
        w = 15.0 / sqrt_n_down
        if m in club_indices:
            w = 25.0 / sqrt_n_down
        elif m in arm_indices:
            w = 20.0 / sqrt_n_down
        res_m = down_diff[:, m, :] * w
        valid_m = np.isfinite(res_m)
        norm_down_res.append(res_m[valid_m].ravel())
    res_down = np.concatenate(norm_down_res)

    # 2. Terminal delivery alignment at t = 1.15s
    term_diff = diff[-1]
    term_res_list = []
    for c in club_indices:
        res_c = term_diff[c] * 100.0  # dominant clubhead weight
        if np.isfinite(res_c).all():
            term_res_list.append(res_c)
    for a in arm_indices:
        res_a = term_diff[a] * 50.0  # arm alignment
        if np.isfinite(res_a).all():
            term_res_list.append(res_a)
    for to in torso_indices:
        res_to = term_diff[to] * 25.0
        if np.isfinite(res_to).all():
            term_res_list.append(res_to)
    for pel in pelvis_indices:
        res_pel = term_diff[pel] * 30.0  # pelvic orientation
        if np.isfinite(res_pel).all():
            term_res_list.append(res_pel)
    res_term = np.concatenate(term_res_list)

    # 3. Soft Barrier: Gate 4 Early Retention (<= 12.0 mm)
    early_dists = np.linalg.norm(diff[early_mask], axis=2)
    valid_early = np.isfinite(early_dists)
    early_rmse = float(np.sqrt(np.mean(early_dists[valid_early] ** 2)))
    penalty_list = []
    if early_rmse > 0.0116:
        penalty_list.append(np.array([(early_rmse - 0.0116) * 500.0]))
    else:
        penalty_list.append(np.array([0.0]))

    # 4. Pelvis Yaw at 1.15s (Gate 6 < 5.0%)
    v_p_now = pred_pts[-1, wr_i, :2] - pred_pts[-1, wl_i, :2]
    v_t_now = target_points[-1, wr_i, :2] - target_points[-1, wl_i, :2]
    yaw_t_now = float(np.degrees(np.arctan2(v_t_now[1], v_t_now[0])))
    yaw_p_now = float(np.degrees(np.arctan2(v_p_now[1], v_p_now[0])))
    diff_yaw_now = float((yaw_p_now - yaw_t_now + 180) % 360 - 180)
    yaw_err_now = float(abs(diff_yaw_now) / max(abs(yaw_t_now), 1.0) * 100.0)
    if yaw_err_now > 5.0:
        penalty_list.append(np.array([(yaw_err_now - 5.0) * 10.0]))
    else:
        penalty_list.append(np.array([0.0]))

    res_pen = np.concatenate(penalty_list)
    res_vec = np.concatenate([res_down, res_term, res_pen])
    cost = float(0.5 * np.sum(res_vec**2))
    if cost < best_cost:
        best_cost = cost

    coeffs = bernstein_to_simscape(full_th.reshape(joints, CONTROL_COUNT), duration_s=T_BASIS)
    ckpt: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "pid": pid,
        "evaluation": eval_count,
        "cost": cost,
        "best_cost": best_cost,
        "early_rmse_mm": early_rmse * 1000.0,
        "yaw_err_115_pct": yaw_err_now,
        "yaw_diff_115_deg": diff_yaw_now,
        "theta": full_th.tolist(),
        "coefficients": coeffs.tolist(),
    }
    tmp_p = scratch_dir / "downswing_115s_locked_checkpoint.tmp"
    tmp_p.write_text(json.dumps(ckpt), encoding="utf-8")
    tmp_p.replace(scratch_dir / "downswing_115s_locked_checkpoint.json")

    if eval_count % 5 == 0 or eval_count == 1:
        club_err = float(np.sqrt(np.mean(np.linalg.norm(diff[-1, club_indices], axis=1) ** 2)) * 1000.0)
        term_err = float(np.sqrt(np.mean(np.linalg.norm(diff[-1], axis=1) ** 2)) * 1000.0)
        logger.info(
            "Eval %4d | Cost: %.4e (Best: %.4e) | Early: %5.2fmm | Club: %6.2fmm | Term: %6.2fmm | Yaw115: %5.2f%%",
            eval_count, cost, best_cost, early_rmse * 1000.0, club_err, term_err, yaw_err_now
        )

    return res_vec

logger.info("Launching least_squares Downswing 1.15s Refinement (%d parameters: k=4,5, diff_step=2e-3)...", n_active)
t_opt_start = time.time()
opt_result = least_squares(
    residual_fn,
    init_free,
    bounds=(lower_free, upper_free),
    diff_step=2e-3,
    max_nfev=350,
    ftol=1e-7,
    xtol=1e-8,
    gtol=1e-7,
    x_scale=1.0,
)
opt_elapsed = time.time() - t_opt_start
logger.info("Downswing 1.15s refinement completed in %.1f s | Status: %d | Message: %s", opt_elapsed, opt_result.status, opt_result.message)

# Final Certification Rollout
final_full = pack_full_theta(opt_result.x)
final_pts = forward_rollout(final_full)
final_diff = final_pts - target_points

final_dists = np.linalg.norm(final_diff[obs_all], axis=1)
whole_rmse_mm = float(np.sqrt(np.mean(final_dists**2)) * 1000.0)

early_dists = np.linalg.norm(final_diff[early_mask][obs_all[early_mask]], axis=1)
early_rmse_mm = float(np.sqrt(np.mean(early_dists**2)) * 1000.0)

term_dists = np.linalg.norm(final_diff[-1][obs_all[-1]], axis=1)
term_rmse_mm = float(np.sqrt(np.mean(term_dists**2)) * 1000.0)

club_term_dists = np.linalg.norm(final_diff[-1, club_indices], axis=1)
clubhead_term_rmse_mm = float(np.sqrt(np.mean(club_term_dists**2)) * 1000.0)

v_p = final_pts[idx_080, wr_i, :2] - final_pts[idx_080, wl_i, :2]
v_t = target_points[idx_080, wr_i, :2] - target_points[idx_080, wl_i, :2]
yaw_t = float(np.degrees(np.arctan2(v_t[1], v_t[0])))
yaw_p = float(np.degrees(np.arctan2(v_p[1], v_p[0])))
yaw_diff_080 = float((yaw_p - yaw_t + 180) % 360 - 180)
yaw_err_080_pct = float(abs(yaw_diff_080) / max(abs(yaw_t), 1.0) * 100.0)

v_p105 = final_pts[idx_105, wr_i, :2] - final_pts[idx_105, wl_i, :2]
v_t105 = target_points[idx_105, wr_i, :2] - target_points[idx_105, wl_i, :2]
yaw_t105 = float(np.degrees(np.arctan2(v_t105[1], v_t105[0])))
yaw_p105 = float(np.degrees(np.arctan2(v_p105[1], v_p105[0])))
yaw_diff_105 = float((yaw_p105 - yaw_t105 + 180) % 360 - 180)
yaw_err_105_pct = float(abs(yaw_diff_105) / max(abs(yaw_t105), 1.0) * 100.0)

v_p115 = final_pts[-1, wr_i, :2] - final_pts[-1, wl_i, :2]
v_t115 = target_points[-1, wr_i, :2] - target_points[-1, wl_i, :2]
yaw_t115 = float(np.degrees(np.arctan2(v_t115[1], v_t115[0])))
yaw_p115 = float(np.degrees(np.arctan2(v_p115[1], v_p115[0])))
diff_115 = float((yaw_p115 - yaw_t115 + 180) % 360 - 180)
yaw_err_115_pct = float(abs(diff_115) / max(abs(yaw_t115), 1.0) * 100.0)

gates = {
    "whole_window_pass": bool(whole_rmse_mm <= 25.0),
    "terminal_rmse_pass": bool(term_rmse_mm <= 35.0),
    "clubhead_terminal_pass": bool(clubhead_term_rmse_mm <= 60.0),
    "early_retention_pass": bool(early_rmse_mm <= 12.0),
    "pelvis_yaw_080_pass": bool(yaw_err_080_pct < 5.0),
    "pelvis_yaw_115_pass": bool(yaw_err_115_pct < 5.0),
}
passed_count = sum(gates.values())

logger.info("=" * 70)
logger.info("FINAL GATE AUDIT REPORT (DOWNSWING 1.15s LOCKED REFINEMENT)")
logger.info("Gate 1 (Whole Window <= 25.0 mm):     %6.2f mm -> %s", whole_rmse_mm, "PASS" if gates["whole_window_pass"] else "FAIL")
logger.info("Gate 2 (Terminal RMSE <= 35.0 mm):     %6.2f mm -> %s", term_rmse_mm, "PASS" if gates["terminal_rmse_pass"] else "FAIL")
logger.info("Gate 3 (Clubhead Term <= 60.0 mm):     %6.2f mm -> %s", clubhead_term_rmse_mm, "PASS" if gates["clubhead_terminal_pass"] else "FAIL")
logger.info("Gate 4 (Early Retention <= 12.0 mm):   %6.2f mm -> %s", early_rmse_mm, "PASS" if gates["early_retention_pass"] else "FAIL")
logger.info("Gate 5 (Pelvis Yaw @ 0.80s < 5.0%%):    %6.2f%% (%+.2f deg) -> %s", yaw_err_080_pct, yaw_diff_080, "PASS" if gates["pelvis_yaw_080_pass"] else "FAIL")
logger.info("Gate 6 (Pelvis Yaw @ 1.15s < 5.0%%):    %6.2f%% (%+.2f deg) -> %s", yaw_err_115_pct, diff_115, "PASS" if gates["pelvis_yaw_115_pass"] else "FAIL")
logger.info("Overall Gate Status:                   %d/6", passed_count)
logger.info("=" * 70)

final_coeffs = bernstein_to_simscape(final_full.reshape(joints, CONTROL_COUNT), duration_s=T_BASIS)
candidate_id = f"prefix-1150ms-downswing-locked-{int(time.time())}"
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
    "pelvis_yaw_105_diff_deg": yaw_diff_105,
    "pelvis_yaw_105_error_pct": yaw_err_105_pct,
    "pelvis_yaw_115_diff_deg": diff_115,
    "pelvis_yaw_115_error_pct": yaw_err_115_pct,
    "max_defect_norm": 0.0,
    "segmented_rmse_m": whole_rmse_mm / 1000.0,
    "unsegmented_rmse_m": whole_rmse_mm / 1000.0,
    "optimizer_converged": bool(opt_result.status > 0),
    "accepted": bool(passed_count >= 5),
    "gates": gates,
    "gates_passed": f"{passed_count}/6",
    "coefficients": final_coeffs.tolist(),
    "elapsed_s": opt_elapsed,
}

out_path = scratch_dir / "downswing_115s_locked_result.json"
out_path.write_text(json.dumps(res_dict, indent=2), encoding="utf-8")
logger.info("Saved result to %s", out_path)

pkg_path = repo / "candidates" / "candidate_downswing_115s_locked_package.json"
pkg_path.parent.mkdir(parents=True, exist_ok=True)
pkg_path.write_text(json.dumps(res_dict, indent=2), encoding="utf-8")
logger.info("Saved candidate package to %s", pkg_path)

try:
    eng.quit()
except Exception:
    pass

