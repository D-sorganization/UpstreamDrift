"""Report and diagnostics helpers for OpenSim barbell lift muscle evaluation.

Provides structured records, coordinate breakdowns, bound pinning checks,
and asymmetry analysis for lift muscle evaluations (OSL-4, #13042).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python import msk_lift_ik_id as ikid
from src.shared.python.lifting.bundle_targets import LiftTargets

IK_ID_RECEIPTS_PATH = (
    Path(__file__).resolve().parents[5]
    / "docs/development/lifting/evidence/opensim_ik_id/receipts.json"
)


@dataclass(frozen=True)
class LiftSolveResults:
    """Frame-by-frame static optimization results across a lift motion window."""

    activations: np.ndarray
    forces: np.ndarray
    reserves: np.ndarray
    tau_muscles: np.ndarray
    tau_targets: np.ndarray
    cap_pos: np.ndarray
    cap_neg: np.ndarray
    passive_tau: np.ndarray
    moment_balance_errors: np.ndarray
    mean_share: float
    peak_share: float
    total_share: float
    classification: str


@dataclass(frozen=True)
class SquatDiagnostics:
    """Diagnostic summaries and kinematic breakdown for back squat muscle fit."""

    major_groups: dict[str, Any]
    phases: dict[str, Any]
    coord_breakdown: dict[str, dict[str, Any]]
    primary_reserve_drivers: list[str]
    bound_violations: dict[str, Any]
    hip_demand: dict[str, Any]
    start_pose: dict[str, float]


def unavailable_lift_record(lift: str, reason: str) -> dict[str, Any]:
    """Return a standard record for lifts lacking an IK/ID bundle or runtime."""
    return {
        "lift": lift,
        "status": "unavailable",
        "reason": reason,
        "classification": "unqualified",
        "reserve_share": None,
        "mean_reserve_share": None,
        "peak_reserve_share": None,
        "residual_status": None,
        "window_start_s": None,
        "window_end_s": None,
        "duration_s": None,
        "start_pose": None,
        "coordinate_breakdown": None,
        "primary_reserve_drivers": None,
        "bound_violations_or_pinned": None,
        "hip_demand_asymmetry": None,
        "forces": None,
        "activations": None,
        "major_groups": None,
        "phases": None,
        "moco": None,
    }


def load_ik_id_residual_status(
    lift: str,
    receipts_path: Path | None = None,
) -> dict[str, Any]:
    """Load measured pelvis residuals and Hicks limits from the IK/ID receipt."""
    path = receipts_path or IK_ID_RECEIPTS_PATH
    if not path.is_file():
        raise FileNotFoundError(f"IK/ID receipts file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    lift_id = data["lifts"][lift]["ik_id"]["id"]
    gate = lift_id["residual_gate"]
    residuals = lift_id["residuals"]
    peak_force = max(
        float(residuals["pelvis_tx"]["max_abs"]),
        float(residuals["pelvis_ty"]["max_abs"]),
        float(residuals["pelvis_tz"]["max_abs"]),
    )
    peak_moment = max(
        float(residuals["pelvis_tilt"]["max_abs"]),
        float(residuals["pelvis_list"]["max_abs"]),
        float(residuals["pelvis_rotation"]["max_abs"]),
    )
    return {
        "forces_within": bool(gate["forces_within"]),
        "moments_within": bool(gate["moments_within"]),
        "measured_pelvis_residual_force_n": round(peak_force, 4),
        "hicks_force_limit_n": round(float(gate["max_force_n"]), 4),
        "measured_pelvis_residual_moment_nm": round(peak_moment, 4),
        "hicks_moment_limit_nm": round(float(gate["max_moment_nm"]), 4),
        "reference": str(
            gate.get("reference", "Hicks et al. 2015, J Biomech Eng 137:020905")
        ),
    }


def compute_coordinate_breakdown(
    leg_coords: Sequence[str],
    solve: LiftSolveResults,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Compute per-coordinate demand, capacity, passive moment, and reserve load."""
    coord_breakdown: dict[str, dict[str, Any]] = {}
    for j, c in enumerate(leg_coords):
        dem = solve.tau_targets[:, j]
        res = solve.reserves[:, j]
        tot_dem = float(np.sum(np.abs(dem)))
        tot_res = float(np.sum(np.abs(res)))
        coord_breakdown[c] = {
            "id_demand_frame0_nm": round(float(dem[0]), 2),
            "id_demand_mean_abs_nm": round(float(np.mean(np.abs(dem))), 2),
            "id_demand_peak_abs_nm": round(float(np.max(np.abs(dem))), 2),
            "active_capacity_pos_mean_nm": round(
                float(np.mean(solve.cap_pos[:, j])), 2
            ),
            "active_capacity_neg_mean_nm": round(
                float(np.mean(solve.cap_neg[:, j])), 2
            ),
            "passive_moment_frame0_nm": round(float(solve.passive_tau[0, j]), 2),
            "passive_moment_mean_nm": round(float(np.mean(solve.passive_tau[:, j])), 2),
            "reserve_frame0_nm": round(float(res[0]), 2),
            "reserve_mean_abs_nm": round(float(np.mean(np.abs(res))), 2),
            "reserve_peak_abs_nm": round(float(np.max(np.abs(res))), 2),
            "reserve_share": round(tot_res / tot_dem, 4) if tot_dem > 1e-6 else 0.0,
        }

    sorted_drivers = sorted(
        leg_coords,
        key=lambda c: float(np.sum(np.abs(solve.reserves[:, leg_coords.index(c)]))),
        reverse=True,
    )
    primary_reserve_drivers = [
        c
        for c in sorted_drivers
        if float(np.sum(np.abs(solve.reserves[:, leg_coords.index(c)]))) > 1.0
    ][:4]

    return coord_breakdown, primary_reserve_drivers


def check_pinned_coordinates(
    model: Any,
    ik: ikid.IKResult,
    leg_coords: Sequence[str],
) -> dict[str, Any]:
    """Check for coordinates resting at or near model range boundaries."""
    coord_set = model.getCoordinateSet()
    pinned_coords: list[str] = []
    bound_details: list[dict[str, Any]] = []
    for name in leg_coords:
        if coord_set.getIndex(name) == -1:
            continue
        coord_obj = coord_set.get(name)
        min_deg = float(np.degrees(coord_obj.getRangeMin()))
        max_deg = float(np.degrees(coord_obj.getRangeMax()))
        q_idx = ik.coordinates.index(name)
        q_series_deg = np.degrees(ik.q[:, q_idx])

        pinned_max = np.abs(q_series_deg - max_deg) <= 1.0
        pinned_min = np.abs(q_series_deg - min_deg) <= 1.0
        pinned_any = pinned_max | pinned_min
        if np.any(pinned_any):
            pinned_coords.append(name)
            bound_type = "max" if np.all(pinned_max[pinned_any]) else "min"
            bound_val = max_deg if bound_type == "max" else min_deg
            bound_details.append(
                {
                    "coordinate": name,
                    "bound": f"{bound_type} ({bound_val:.1f} deg)",
                    "frames_pinned_count": int(np.sum(pinned_any)),
                    "fraction_pinned": round(float(np.mean(pinned_any)), 4),
                    "min_margin_deg": round(
                        float(
                            np.min(
                                np.minimum(
                                    np.abs(q_series_deg - min_deg),
                                    np.abs(q_series_deg - max_deg),
                                )
                            )
                        ),
                        3,
                    ),
                }
            )

    return {
        "has_pinned_coordinates": bool(pinned_coords),
        "warning": "An IK result pinned at a coordinate bound is not a credible posture.",
        "pinned_coordinates": sorted(set(pinned_coords)),
        "details": bound_details,
    }


def compute_hip_demand_asymmetry(
    tau_targets: np.ndarray,
    leg_coords: Sequence[str],
    targets: LiftTargets,
) -> dict[str, Any]:
    """Explain left/right hip demand asymmetry from ground reaction shear."""
    hip_r_idx = leg_coords.index("hip_flexion_r")
    hip_l_idx = leg_coords.index("hip_flexion_l")
    knee_r_idx = leg_coords.index("knee_angle_r")
    knee_l_idx = leg_coords.index("knee_angle_l")
    tau0 = tau_targets[0]
    grf_l = targets.grf_force_n["left"][0]
    grf_r = targets.grf_force_n["right"][0]
    record: dict[str, Any] = {
        "frame0_hip_demand_r_nm": round(float(tau0[hip_r_idx]), 2),
        "frame0_hip_demand_l_nm": round(float(tau0[hip_l_idx]), 2),
        "frame0_knee_demand_r_nm": round(float(tau0[knee_r_idx]), 2),
        "frame0_knee_demand_l_nm": round(float(tau0[knee_l_idx]), 2),
        "frame0_total_sagittal_r_nm": round(
            float(tau0[hip_r_idx] + tau0[knee_r_idx]), 2
        ),
        "frame0_total_sagittal_l_nm": round(
            float(tau0[hip_l_idx] + tau0[knee_l_idx]), 2
        ),
        "frame0_grf_vertical_l_n": round(float(grf_l[1]), 2),
        "frame0_grf_vertical_r_n": round(float(grf_r[1]), 2),
        "frame0_grf_anterior_shear_l_n": round(float(grf_l[0]), 2),
        "frame0_grf_anterior_shear_r_n": round(float(grf_r[0]), 2),
        "shear_differential_n": round(float(grf_l[0] - grf_r[0]), 2),
    }
    record["cause"] = _asymmetry_cause(record)
    return record


def _asymmetry_cause(r: dict[str, float]) -> str:
    """Describe the frame-0 hip/knee demand split from the measured record."""
    tot_r, tot_l = r["frame0_total_sagittal_r_nm"], r["frame0_total_sagittal_l_nm"]
    rel = abs(tot_r - tot_l) / max(abs(tot_r), abs(tot_l), 1e-12)
    return (
        f"Total sagittal leg extension demand at frame 0 is {tot_r:.2f} N m right vs "
        f"{tot_l:.2f} N m left ({100.0 * rel:.2f}% apart), with vertical GRF "
        f"{r['frame0_grf_vertical_r_n']:.2f} N right vs "
        f"{r['frame0_grf_vertical_l_n']:.2f} N left. The hip-vs-knee split "
        f"(hip {r['frame0_hip_demand_r_nm']:.2f} N m right vs "
        f"{r['frame0_hip_demand_l_nm']:.2f} N m left; knee "
        f"{r['frame0_knee_demand_r_nm']:.2f} N m right vs "
        f"{r['frame0_knee_demand_l_nm']:.2f} N m left) follows the "
        f"anterior-posterior ground shear: {r['frame0_grf_anterior_shear_l_n']:+.2f} N "
        f"on the left foot vs {r['frame0_grf_anterior_shear_r_n']:+.2f} N on the right "
        f"(a {r['shear_differential_n']:.2f} N differential), which moves each GRF "
        "line of action and so trades knee extension demand for hip extension demand."
    )


def extract_start_pose(ik: ikid.IKResult) -> dict[str, float]:
    """Extract start pose joint angles in degrees."""
    return {
        "hip_flexion_r_deg": round(
            float(np.degrees(ik.q[0, ik.coordinates.index("hip_flexion_r")])), 2
        ),
        "hip_flexion_l_deg": round(
            float(np.degrees(ik.q[0, ik.coordinates.index("hip_flexion_l")])), 2
        ),
        "knee_angle_r_deg": round(
            float(np.degrees(ik.q[0, ik.coordinates.index("knee_angle_r")])), 2
        ),
        "knee_angle_l_deg": round(
            float(np.degrees(ik.q[0, ik.coordinates.index("knee_angle_l")])), 2
        ),
    }


def compute_squat_diagnostics(
    model: Any,
    ik: ikid.IKResult,
    leg_coords: Sequence[str],
    targets: LiftTargets,
    solve: LiftSolveResults,
    major_groups: dict[str, Any],
    phases: dict[str, Any],
) -> SquatDiagnostics:
    """Consolidate diagnostic breakdowns, pinning flags, and posture metrics."""
    coord_breakdown, primary_reserve_drivers = compute_coordinate_breakdown(
        leg_coords, solve
    )
    bound_violations = check_pinned_coordinates(model, ik, leg_coords)
    hip_demand = compute_hip_demand_asymmetry(solve.tau_targets, leg_coords, targets)
    start_pose = extract_start_pose(ik)
    return SquatDiagnostics(
        major_groups=major_groups,
        phases=phases,
        coord_breakdown=coord_breakdown,
        primary_reserve_drivers=primary_reserve_drivers,
        bound_violations=bound_violations,
        hip_demand=hip_demand,
        start_pose=start_pose,
    )


def format_squat_solution(
    lift: str,
    targets: LiftTargets,
    muscle_names: Sequence[str],
    leg_coords: Sequence[str],
    solve: LiftSolveResults,
    diag: SquatDiagnostics,
    moco_record: dict[str, Any],
    citation: str,
    *,
    residual_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the complete solution dictionary for a barbell lift."""
    status = (
        residual_status
        if residual_status is not None
        else load_ik_id_residual_status(lift)
    )
    return {
        "lift": lift,
        "status": "complete",
        "reason": "",
        "classification": solve.classification,
        "reserve_share": solve.total_share,
        "mean_reserve_share": solve.mean_share,
        "peak_reserve_share": solve.peak_share,
        "residual_status": status,
        "window_start_s": float(targets.time_s[0]),
        "window_end_s": float(targets.time_s[-1]),
        "duration_s": round(float(targets.time_s[-1] - targets.time_s[0]), 4),
        "start_pose": diag.start_pose,
        "coordinate_breakdown": diag.coord_breakdown,
        "primary_reserve_drivers": diag.primary_reserve_drivers,
        "bound_violations_or_pinned": diag.bound_violations,
        "hip_demand_asymmetry": diag.hip_demand,
        "moment_balance_max_error_nm": float(np.max(solve.moment_balance_errors)),
        "times": targets.time_s,
        "muscle_names": list(muscle_names),
        "leg_coordinates": list(leg_coords),
        "activations": solve.activations,
        "forces": solve.forces,
        "reserves": solve.reserves,
        "tau_targets": solve.tau_targets,
        "tau_muscles": solve.tau_muscles,
        "major_groups": diag.major_groups,
        "phases": diag.phases,
        "moco": moco_record,
        "citation": citation,
    }


# Backwards-compatible aliases for lift-agnostic reporting
format_lift_solution = format_squat_solution
compute_lift_diagnostics = compute_squat_diagnostics
LiftDiagnostics = SquatDiagnostics
