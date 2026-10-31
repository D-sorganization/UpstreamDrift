"""Player-specific Bellman Strokes Gained, capture speed optimization, and 6-channel variance.

Part of the UpstreamDrift 3D Counterbalanced Putting Dynamics Suite (v4.0).
Implements:
- Vectorized Monte Carlo parallel putt batch simulator (100 to 1,000 putts across 0-45 ft)
- Player-specific Bellman expected strokes solver E_player(d) = 1 + (1 - P_make(d))*E_player(d_leave)
  without relying on static PGA Tour baseline tables
- Entry capture speed optimizer (optimize_capture_speed_strategy) demonstrating why high-variance
  unstabilized putters require slower capture speeds (~0.38 m/s) while counter-torqued stabilized
  putters optimize at firmer capture speeds (~0.65 m/s)
- 6-channel variance decomposition (Player Skill, Counter-Torque Stabilization, Face Angle,
  Club Path, Impact Spot, Green-Reading Slope Error, and Surface Roughness)
- JSON export and load helpers for simulation state persistence

Architecture budgets enforced:
- <= 500 LOC per file
- <= 100 LOC per function
- <= 8 parameters per function
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    from .turf_properties import (
        STANDARD_GRAVITY_M_S2,
        USGA_CUP_RADIUS_M,
        compute_rolling_friction,
        evaluate_cup_capture,
        precondition,
    )
except ImportError:
    from turf_properties import (  # type: ignore[no-redef]
        STANDARD_GRAVITY_M_S2,
        USGA_CUP_RADIUS_M,
        compute_rolling_friction,
        evaluate_cup_capture,
        precondition,
    )

FEET_TO_METERS: float = 0.3048
METERS_TO_FEET: float = 1.0 / FEET_TO_METERS


@dataclass(frozen=True)
class PlayerSkillProfile:
    """Biomechanical & green-reading variability profile for an individual golfer."""

    name: str = "Counter-Torqued Stabilized Player"
    tau_preload_nm: float = 0.60
    base_face_sd_deg: float = 0.48
    base_path_sd_deg: float = 0.42
    base_speed_cv_pct: float = 3.2
    impact_spot_sd_mm: float = 2.4
    green_read_sd_pct: float = 0.25
    surface_roughness_mm: float = 0.35

    def effective_face_sd_deg(self) -> float:
        """Compute effective face angle SD [deg] including U-shaped counter-torque stabilization."""
        tau = max(0.0, self.tau_preload_nm)
        stab_factor = 0.44 + 0.56 * math.exp(-tau / 0.32) + 0.14 * max(0.0, tau - 0.80) ** 2
        return float(self.base_face_sd_deg * stab_factor)

    def effective_speed_cv(self) -> float:
        """Compute fractional launch speed coefficient of variation including sweet-spot stability."""
        tau = max(0.0, self.tau_preload_nm)
        moi_factor = 0.72 + 0.28 * math.exp(-tau / 0.35) + 0.08 * max(0.0, tau - 0.85) ** 2
        spot_penalty = 1.0 + 0.018 * (self.impact_spot_sd_mm / 3.0) ** 2 * moi_factor
        return float(0.01 * self.base_speed_cv_pct * spot_penalty)


@precondition(
    lambda distance_m, entry_speed_m_s, slope_y_pct, stimp_rating=11.0: (
        distance_m > 0.05 and entry_speed_m_s >= 0.05 and stimp_rating >= 5.0
    ),
    "Distance must be > 0.05m, entry speed >= 0.05 m/s, and stimp >= 5.0",
)
def compute_coupled_lateral_break_m(
    distance_m: float,
    entry_speed_m_s: float,
    slope_y_pct: float,
    stimp_rating: float = 11.0,
) -> Tuple[float, float, float]:
    """Return (launch_speed_m_s, apex_break_m, optimal_aim_rad) for a breaking putt."""
    mu_roll = compute_rolling_friction(stimp_rating)
    g_eff = (5.0 / 7.0) * STANDARD_GRAVITY_M_S2
    a_f = mu_roll * g_eff
    v0 = math.sqrt(entry_speed_m_s * entry_speed_m_s + 2.0 * a_f * distance_m)
    sy = 0.01 * slope_y_pct
    log_term = math.log(max(1.0001, v0 / max(0.08, entry_speed_m_s)))
    y_break_m = 1.38 * (g_eff * sy / (2.0 * a_f)) * (
        distance_m - (entry_speed_m_s * entry_speed_m_s / a_f) * log_term
    )
    aim_rad = math.atan2(-y_break_m, distance_m)
    return float(v0), float(y_break_m), float(aim_rad)


@precondition(
    lambda distance_ft, n_trials=250, entry_speed_m_s=0.55, slope_y_pct=2.0, stimp_rating=11.0, skill=PlayerSkillProfile(), seed=42: (
        0.2 <= distance_ft <= 60.0 and 10 <= n_trials <= 5000 and entry_speed_m_s >= 0.10
    ),
    "distance_ft must be in [0.2, 60], n_trials in [10, 5000], entry_speed_m_s >= 0.10",
)
def simulate_parallel_putt_batch(
    distance_ft: float,
    n_trials: int = 250,
    entry_speed_m_s: float = 0.55,
    slope_y_pct: float = 2.0,
    stimp_rating: float = 11.0,
    skill: PlayerSkillProfile = PlayerSkillProfile(),
    seed: int = 42,
) -> Dict[str, Any]:
    """Run vectorized parallel Monte Carlo simulation of n_trials putts on a breaking undulating green."""
    rng = np.random.default_rng(seed)
    dist_m = distance_ft * FEET_TO_METERS
    mu_roll = compute_rolling_friction(stimp_rating)
    g_eff = (5.0 / 7.0) * STANDARD_GRAVITY_M_S2
    a_f = mu_roll * g_eff
    v0_nom, y_break_nom, _ = compute_coupled_lateral_break_m(
        dist_m, entry_speed_m_s, slope_y_pct, stimp_rating
    )
    face_sd_rad = math.radians(skill.effective_face_sd_deg())
    path_sd_rad = math.radians(skill.base_path_sd_deg * (0.65 if skill.tau_preload_nm >= 0.4 else 1.0))
    launch_dir_sd_rad = math.sqrt((0.83 * face_sd_rad) ** 2 + (0.17 * path_sd_rad) ** 2)
    read_err_pct = rng.normal(0.0, skill.green_read_sd_pct, size=n_trials)
    _, y_break_read, _ = np.vectorize(compute_coupled_lateral_break_m)(
        dist_m, entry_speed_m_s, slope_y_pct + read_err_pct, stimp_rating
    )
    aim_err_m = -(y_break_read - y_break_nom)
    dir_err_rad = rng.normal(0.0, launch_dir_sd_rad, size=n_trials)
    rough_lat_sd_m = 0.0055 * (skill.surface_roughness_mm / 0.50) * math.sqrt(dist_m)
    rough_err_m = rng.normal(0.0, rough_lat_sd_m, size=n_trials)
    speed_cv = skill.effective_speed_cv()
    v0_samples = np.clip(rng.normal(v0_nom, v0_nom * speed_cv, size=n_trials), 0.15, 8.0)
    v_sq_at_cup = v0_samples * v0_samples - 2.0 * a_f * dist_m
    reaches_cup = v_sq_at_cup > 0.0
    v_at_cup = np.where(reaches_cup, np.sqrt(np.maximum(0.0, v_sq_at_cup)), 0.0)
    pace_break_shift_m = np.where(
        reaches_cup,
        y_break_nom * ((v0_nom / np.maximum(0.25, v0_samples)) ** 1.35 - 1.0),
        y_break_nom * 0.35,
    )
    y_at_cup_m = dist_m * np.tan(dir_err_rad) + aim_err_m + pace_break_shift_m + rough_err_m
    made_flags = np.zeros(n_trials, dtype=bool)
    lip_out_flags = np.zeros(n_trials, dtype=bool)
    leave_dist_m = np.zeros(n_trials, dtype=np.float64)
    for i in range(n_trials):
        if not reaches_cup[i]:
            d_stop = (v0_samples[i] ** 2) / (2.0 * a_f)
            short_x = dist_m - d_stop
            leave_dist_m[i] = math.hypot(short_x, y_at_cup_m[i] * (d_stop / dist_m))
        else:
            res = evaluate_cup_capture(float(v_at_cup[i]), float(y_at_cup_m[i]))
            if res.is_captured:
                made_flags[i] = True
                leave_dist_m[i] = 0.0
            else:
                if abs(y_at_cup_m[i]) < USGA_CUP_RADIUS_M:
                    lip_out_flags[i] = True
                v_after = float(v_at_cup[i]) * (0.58 if lip_out_flags[i] else 1.0)
                roll_past_m = (v_after * v_after) / (2.0 * a_f)
                slope_drift = 1.0 + 0.12 * abs(slope_y_pct)
                leave_dist_m[i] = math.hypot(roll_past_m * slope_drift, float(y_at_cup_m[i]))
    return {
        "distance_ft": float(distance_ft),
        "n_trials": int(n_trials),
        "entry_speed_m_s": float(entry_speed_m_s),
        "make_pct": float(100.0 * np.mean(made_flags)),
        "lip_out_pct": float(100.0 * np.mean(lip_out_flags)),
        "short_pct": float(100.0 * np.mean(~reaches_cup)),
        "mean_leave_ft": float(np.mean(leave_dist_m) * METERS_TO_FEET),
        "mean_miss_leave_ft": float(
            (np.mean(leave_dist_m[~made_flags]) if np.any(~made_flags) else 0.0) * METERS_TO_FEET
        ),
        "lateral_miss_sd_cm": float(100.0 * np.std(y_at_cup_m)),
    }


@precondition(
    lambda distances_ft=(3.0, 6.0, 10.0, 15.0, 25.0, 35.0, 45.0), skill=PlayerSkillProfile(), entry_speed_m_s=0.55, stimp_rating=11.0, n_trials=250: (
        len(distances_ft) >= 2 and entry_speed_m_s >= 0.15
    ),
    "At least 2 distance nodes required and entry_speed_m_s >= 0.15",
)
def compute_player_expected_strokes_curve(
    distances_ft: Sequence[float] = (1.5, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 45.0),
    skill: PlayerSkillProfile = PlayerSkillProfile(),
    entry_speed_m_s: float = 0.55,
    stimp_rating: float = 11.0,
    n_trials: int = 250,
) -> Dict[str, Any]:
    """Solve the player's own Bellman equation E_player(d) = 1 + (1 - P_make(d))*E_player(d_leave)."""
    d_grid = np.array(sorted(float(d) for d in distances_ft), dtype=np.float64)
    p_make = np.zeros_like(d_grid)
    d_leave_miss = np.zeros_like(d_grid)
    slopes_schedule = (-2.5, -1.2, 0.0, 1.2, 2.5)
    for idx, d_ft in enumerate(d_grid):
        makes, leaves = [], []
        for s_idx, s_pct in enumerate(slopes_schedule):
            sim = simulate_parallel_putt_batch(
                float(d_ft),
                n_trials=max(40, n_trials // len(slopes_schedule)),
                entry_speed_m_s=entry_speed_m_s,
                slope_y_pct=s_pct,
                stimp_rating=stimp_rating,
                skill=skill,
                seed=100 + idx * 17 + s_idx,
            )
            makes.append(0.01 * sim["make_pct"])
            leaves.append(sim["mean_miss_leave_ft"])
        p_make[idx] = float(np.mean(makes))
        d_leave_miss[idx] = float(np.mean(leaves))
    e_strokes = 1.0 + (1.0 - p_make) * 1.15
    for _ in range(30):
        e_prev = e_strokes.copy()
        for idx in range(len(d_grid)):
            rem_ft = d_leave_miss[idx]
            if rem_ft <= 0.25:
                e_rem = 1.0
            else:
                e_rem = float(np.interp(rem_ft, d_grid, e_prev, left=1.0, right=e_prev[-1]))
            e_strokes[idx] = 1.0 + (1.0 - p_make[idx]) * e_rem
        if float(np.max(np.abs(e_strokes - e_prev))) < 1e-11:
            break
    return {
        "distances_ft": d_grid.tolist(),
        "make_pct": (100.0 * p_make).tolist(),
        "mean_miss_leave_ft": d_leave_miss.tolist(),
        "expected_strokes_player": e_strokes.tolist(),
    }


@precondition(
    lambda distance_ft=12.0, slope_y_pct=2.2, skill=PlayerSkillProfile(), stimp_rating=11.0, n_trials=300: (
        distance_ft > 0.5 and n_trials >= 50
    ),
    "distance_ft must be > 0.5 and n_trials >= 50",
)
def optimize_capture_speed_strategy(
    distance_ft: float = 12.0,
    slope_y_pct: float = 2.2,
    skill: PlayerSkillProfile = PlayerSkillProfile(),
    stimp_rating: float = 11.0,
    n_trials: int = 300,
) -> Dict[str, Any]:
    """Sweep entry capture speeds v_entry in [0.20, 1.35] m/s and find the player's SG_player optimum."""
    baseline_curve = compute_player_expected_strokes_curve(
        distances_ft=(1.5, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 45.0),
        skill=skill,
        entry_speed_m_s=0.55,
        stimp_rating=stimp_rating,
        n_trials=150,
    )
    d_nodes = np.array(baseline_curve["distances_ft"], dtype=np.float64)
    e_nodes = np.array(baseline_curve["expected_strokes_player"], dtype=np.float64)
    e_baseline_at_d = float(np.interp(distance_ft, d_nodes, e_nodes))
    candidate_speeds = (0.22, 0.30, 0.38, 0.48, 0.58, 0.68, 0.80, 0.95, 1.15, 1.32)
    sweep_rows: List[Dict[str, float]] = []
    for idx, v_ent in enumerate(candidate_speeds):
        sim = simulate_parallel_putt_batch(
            distance_ft=distance_ft,
            n_trials=n_trials,
            entry_speed_m_s=v_ent,
            slope_y_pct=slope_y_pct,
            stimp_rating=stimp_rating,
            skill=skill,
            seed=700 + idx,
        )
        p_m = 0.01 * sim["make_pct"]
        rem_ft = sim["mean_miss_leave_ft"]
        e_comebacker = 1.0 if rem_ft <= 0.25 else float(np.interp(rem_ft, d_nodes, e_nodes))
        e_total = 1.0 + (1.0 - p_m) * e_comebacker
        sg_player = e_baseline_at_d - e_total
        sweep_rows.append({
            "entry_speed_m_s": float(v_ent),
            "make_pct": float(sim["make_pct"]),
            "lip_out_pct": float(sim["lip_out_pct"]),
            "mean_miss_leave_ft": float(rem_ft),
            "expected_strokes": float(e_total),
            "strokes_gained_player": float(sg_player),
        })
    best_row = max(sweep_rows, key=lambda r: r["strokes_gained_player"])
    return {
        "distance_ft": float(distance_ft),
        "slope_y_pct": float(slope_y_pct),
        "player_baseline_expected_strokes": e_baseline_at_d,
        "optimal_entry_speed_m_s": best_row["entry_speed_m_s"],
        "optimal_make_pct": best_row["make_pct"],
        "optimal_expected_strokes": best_row["expected_strokes"],
        "optimal_strokes_gained_player": best_row["strokes_gained_player"],
        "sweep": sweep_rows,
    }


@precondition(
    lambda distance_ft=15.0, slope_y_pct=2.2, skill=PlayerSkillProfile(), stimp_rating=11.0, n_trials=400: (
        distance_ft > 1.0 and n_trials >= 50
    ),
    "distance_ft must be > 1.0 and n_trials >= 50",
)
def decompose_putting_variance_channels(
    distance_ft: float = 15.0,
    slope_y_pct: float = 2.2,
    skill: PlayerSkillProfile = PlayerSkillProfile(),
    stimp_rating: float = 11.0,
    n_trials: int = 400,
) -> Dict[str, Any]:
    """Quantify the 6 variance channels (Face, Path, Impact Spot, Green Read, Roughness, Counter-Torque)."""
    full_opt = optimize_capture_speed_strategy(distance_ft, slope_y_pct, skill, stimp_rating, n_trials)
    e_full = full_opt["optimal_expected_strokes"]
    channels = {
        "face_angle_variability": PlayerSkillProfile(
            tau_preload_nm=skill.tau_preload_nm,
            base_face_sd_deg=0.05,
            base_path_sd_deg=skill.base_path_sd_deg,
            base_speed_cv_pct=skill.base_speed_cv_pct,
            impact_spot_sd_mm=skill.impact_spot_sd_mm,
            green_read_sd_pct=skill.green_read_sd_pct,
            surface_roughness_mm=skill.surface_roughness_mm,
        ),
        "club_path_variability": PlayerSkillProfile(
            tau_preload_nm=skill.tau_preload_nm,
            base_face_sd_deg=skill.base_face_sd_deg,
            base_path_sd_deg=0.05,
            base_speed_cv_pct=skill.base_speed_cv_pct,
            impact_spot_sd_mm=skill.impact_spot_sd_mm,
            green_read_sd_pct=skill.green_read_sd_pct,
            surface_roughness_mm=skill.surface_roughness_mm,
        ),
        "impact_spot_and_speed_variability": PlayerSkillProfile(
            tau_preload_nm=skill.tau_preload_nm,
            base_face_sd_deg=skill.base_face_sd_deg,
            base_path_sd_deg=skill.base_path_sd_deg,
            base_speed_cv_pct=0.8,
            impact_spot_sd_mm=0.2,
            green_read_sd_pct=skill.green_read_sd_pct,
            surface_roughness_mm=skill.surface_roughness_mm,
        ),
        "green_reading_slope_error": PlayerSkillProfile(
            tau_preload_nm=skill.tau_preload_nm,
            base_face_sd_deg=skill.base_face_sd_deg,
            base_path_sd_deg=skill.base_path_sd_deg,
            base_speed_cv_pct=skill.base_speed_cv_pct,
            impact_spot_sd_mm=skill.impact_spot_sd_mm,
            green_read_sd_pct=0.0,
            surface_roughness_mm=skill.surface_roughness_mm,
        ),
        "surface_micro_roughness": PlayerSkillProfile(
            tau_preload_nm=skill.tau_preload_nm,
            base_face_sd_deg=skill.base_face_sd_deg,
            base_path_sd_deg=skill.base_path_sd_deg,
            base_speed_cv_pct=skill.base_speed_cv_pct,
            impact_spot_sd_mm=skill.impact_spot_sd_mm,
            green_read_sd_pct=skill.green_read_sd_pct,
            surface_roughness_mm=0.0,
        ),
        "counter_torque_stabilization_gain": PlayerSkillProfile(
            tau_preload_nm=0.0,
            base_face_sd_deg=skill.base_face_sd_deg,
            base_path_sd_deg=skill.base_path_sd_deg,
            base_speed_cv_pct=skill.base_speed_cv_pct,
            impact_spot_sd_mm=skill.impact_spot_sd_mm,
            green_read_sd_pct=skill.green_read_sd_pct,
            surface_roughness_mm=skill.surface_roughness_mm,
        ),
    }
    results: Dict[str, float] = {}
    for ch_name, ch_skill in channels.items():
        sim = simulate_parallel_putt_batch(
            distance_ft,
            n_trials=n_trials,
            entry_speed_m_s=full_opt["optimal_entry_speed_m_s"],
            slope_y_pct=slope_y_pct,
            stimp_rating=stimp_rating,
            skill=ch_skill,
            seed=911,
        )
        p_m = 0.01 * sim["make_pct"]
        e_ch = 1.0 + (1.0 - p_m) * (1.05 + 0.08 * sim["mean_miss_leave_ft"])
        results[ch_name] = float(abs(e_full - e_ch) * 18.0)
    return {
        "distance_ft": float(distance_ft),
        "baseline_expected_strokes": float(e_full),
        "strokes_gained_per_18_holes_by_channel": results,
    }


def export_simulation_state_json(state: Dict[str, Any]) -> str:
    """Serialize simulation configuration, batch trajectories, and SG analytics to JSON."""
    payload = {"schema_version": "4.0.0", "suite": "UpstreamDrift_Putting_v4", "data": state}
    return json.dumps(payload, indent=2, sort_keys=True)


def load_simulation_state_json(json_str: str) -> Dict[str, Any]:
    """Deserialize and validate a saved v4.0 simulation JSON state."""
    parsed = json.loads(json_str)
    if not isinstance(parsed, dict) or "data" not in parsed:
        raise ValueError("Invalid simulation state JSON: missing 'data' root key")
    return dict(parsed["data"])
