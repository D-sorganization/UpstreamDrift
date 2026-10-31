"""Virtual USGA Stimpmeter V-groove ramp rollout and optimal player-specific green reader.

Part of the UpstreamDrift 3D Counterbalanced Putting Dynamics Suite (v4.0).
Implements:
- Virtual USGA Stimpmeter V-groove ramp release (v_release = 1.83 m/s at 20 deg notch release)
  integrated via 4th-order Runge-Kutta (RK4) across non-planar undulating green topography
- Bidirectional Stimpmeter green calibration (uphill/downhill/cross-slope effective Stimp)
- Player-specific optimal green reading solver combining Stimpmeter calibration with the
  golfer's Bellman Strokes Gained pace optimizer

Architecture budgets enforced:
- <= 500 LOC per file
- <= 100 LOC per function
- <= 8 parameters per function
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from .green_surface_undulation import (
        GreenUndulationProfile,
        evaluate_green_elevation_m,
        evaluate_green_gradient,
        sample_surface_roughness_acceleration,
    )
    from .strokes_gained_analytics import (
        FEET_TO_METERS,
        METERS_TO_FEET,
        PlayerSkillProfile,
        compute_coupled_lateral_break_m,
        optimize_capture_speed_strategy,
    )
    from .turf_properties import (
        STANDARD_GRAVITY_M_S2,
        compute_rolling_friction,
        precondition,
    )
except ImportError:
    from green_surface_undulation import (  # type: ignore[no-redef]
        GreenUndulationProfile,
        evaluate_green_elevation_m,
        evaluate_green_gradient,
        sample_surface_roughness_acceleration,
    )
    from strokes_gained_analytics import (  # type: ignore[no-redef]
        FEET_TO_METERS,
        METERS_TO_FEET,
        PlayerSkillProfile,
        compute_coupled_lateral_break_m,
        optimize_capture_speed_strategy,
    )
    from turf_properties import (  # type: ignore[no-redef]
        STANDARD_GRAVITY_M_S2,
        compute_rolling_friction,
        precondition,
    )

USGA_STIMPMETER_RELEASE_SPEED_M_S: float = 1.83


@dataclass(frozen=True)
class StimpmeterRolloutResult:
    """Result of rolling a calibration ball off a USGA Stimpmeter ramp."""

    release_speed_m_s: float
    launch_angle_deg: float
    stopping_distance_m: float
    stopping_distance_ft: float
    lateral_deflection_m: float
    trajectory_xy_m: Tuple[Tuple[float, float], ...]


@precondition(
    lambda launch_angle_deg=0.0, stimp_rating=11.0, profile=GreenUndulationProfile(), release_speed_m_s=1.83, dt_s=0.01: (
        stimp_rating >= 5.0 and release_speed_m_s > 0.1 and dt_s > 0.0
    ),
    "stimp_rating >= 5.0, release_speed_m_s > 0.1, and dt_s > 0.0",
)
def roll_stimpmeter_ball(
    launch_angle_deg: float = 0.0,
    stimp_rating: float = 11.0,
    profile: GreenUndulationProfile = GreenUndulationProfile(),
    release_speed_m_s: float = USGA_STIMPMETER_RELEASE_SPEED_M_S,
    dt_s: float = 0.01,
) -> StimpmeterRolloutResult:
    """Integrate a pure-rolling ball released from a USGA Stimpmeter V-groove ramp via RK4."""
    mu_roll = compute_rolling_friction(stimp_rating)
    g_eff = (5.0 / 7.0) * STANDARD_GRAVITY_M_S2
    ang_rad = math.radians(launch_angle_deg)
    state = np.array(
        [0.0, 0.0, release_speed_m_s * math.cos(ang_rad), release_speed_m_s * math.sin(ang_rad)],
        dtype=np.float64,
    )
    pts: List[Tuple[float, float]] = [(0.0, 0.0)]

    def deriv(s: np.ndarray) -> np.ndarray:
        x_p, y_p, vx, vy = float(s[0]), float(s[1]), float(s[2]), float(s[3])
        spd = math.hypot(vx, vy)
        if spd < 1e-3:
            return np.zeros(4, dtype=np.float64)
        dz_dx, dz_dy = evaluate_green_gradient(x_p, y_p, profile)
        ax_rough, ay_rough = sample_surface_roughness_acceleration(
            x_p, y_p, spd, profile.roughness_rms_mm, seed=0
        )
        ax = -mu_roll * g_eff * (vx / spd) - g_eff * dz_dx + ax_rough
        ay = -mu_roll * g_eff * (vy / spd) - g_eff * dz_dy + ay_rough
        return np.array([vx, vy, ax, ay], dtype=np.float64)

    for _ in range(1200):
        spd = math.hypot(float(state[2]), float(state[3]))
        if spd < 0.015:
            break
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt_s * k1)
        k3 = deriv(state + 0.5 * dt_s * k2)
        k4 = deriv(state + dt_s * k3)
        state = state + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        pts.append((float(state[0]), float(state[1])))
    stop_m = math.hypot(float(state[0]), float(state[1]))
    return StimpmeterRolloutResult(
        release_speed_m_s=float(release_speed_m_s),
        launch_angle_deg=float(launch_angle_deg),
        stopping_distance_m=float(stop_m),
        stopping_distance_ft=float(stop_m * METERS_TO_FEET),
        lateral_deflection_m=float(state[1]),
        trajectory_xy_m=tuple(pts[::4]),
    )


@precondition(
    lambda distance_ft=12.0, stimp_rating=11.0, profile=GreenUndulationProfile(), skill=PlayerSkillProfile(): (
        distance_ft > 0.5 and stimp_rating >= 5.0
    ),
    "distance_ft > 0.5 and stimp_rating >= 5.0",
)
def solve_player_optimal_green_read(
    distance_ft: float = 12.0,
    stimp_rating: float = 11.0,
    profile: GreenUndulationProfile = GreenUndulationProfile(),
    skill: PlayerSkillProfile = PlayerSkillProfile(),
) -> Dict[str, Any]:
    """Use bidirectional Stimpmeter rollouts + SG_player optimization to prescribe optimal read."""
    fwd_roll = roll_stimpmeter_ball(0.0, stimp_rating, profile)
    bwd_roll = roll_stimpmeter_ball(180.0, stimp_rating, profile)
    eff_stimp_ft = 2.0 * (fwd_roll.stopping_distance_ft * bwd_roll.stopping_distance_ft) / max(
        0.1, fwd_roll.stopping_distance_ft + bwd_roll.stopping_distance_ft
    )
    dist_m = distance_ft * FEET_TO_METERS
    gx_mid, gy_mid = evaluate_green_gradient(0.5 * dist_m, 0.0, profile)
    eff_slope_y_pct = 100.0 * gy_mid
    pace_opt = optimize_capture_speed_strategy(
        distance_ft=distance_ft,
        slope_y_pct=eff_slope_y_pct,
        skill=skill,
        stimp_rating=eff_stimp_ft,
        n_trials=240,
    )
    v_entry_opt = pace_opt["optimal_entry_speed_m_s"]
    v0_opt, y_break_m, aim_rad = compute_coupled_lateral_break_m(
        dist_m, v_entry_opt, eff_slope_y_pct, eff_stimp_ft
    )
    elev_diff_m = evaluate_green_elevation_m(dist_m, 0.0, profile) - evaluate_green_elevation_m(
        0.0, 0.0, profile
    )
    return {
        "distance_ft": float(distance_ft),
        "stimp_forward_ft": fwd_roll.stopping_distance_ft,
        "stimp_backward_ft": bwd_roll.stopping_distance_ft,
        "effective_stimp_ft": float(eff_stimp_ft),
        "elevation_change_cm": float(100.0 * elev_diff_m),
        "effective_cross_slope_pct": float(eff_slope_y_pct),
        "optimal_entry_capture_speed_m_s": float(v_entry_opt),
        "optimal_launch_speed_m_s": float(v0_opt),
        "predicted_apex_break_cm": float(100.0 * abs(y_break_m)),
        "optimal_aim_angle_deg": float(math.degrees(aim_rad)),
        "expected_make_pct": float(pace_opt["optimal_make_pct"]),
        "expected_strokes": float(pace_opt["optimal_expected_strokes"]),
        "strokes_gained_vs_default_pace": float(pace_opt["optimal_strokes_gained_player"]),
    }
