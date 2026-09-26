"""Quantitative acceptance gates and verification evaluation helper (FB-3..FB-6, #10960 P0-9).

Defines the single authoritative gate evaluator and documented physical/kinematic
thresholds for full-body verification receipts under issue #10960.
"""

from __future__ import annotations

from collections.abc import Container, Mapping
import math
from typing import Any

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.acceptance import AcceptanceGates

_DEFAULT_GATES = AcceptanceGates()

FB3_DRAKE_THRESHOLDS: dict[str, float] = {
    "gate_a_fk_diff_m": 1e-12,
    "gate_a_mass_matrix_diff": 1e-12,
    "gate_b_fk_diff_m": 1e-12,
    "gate_c_normal_force_diff_n": 1e-12,
    "gate_c_friction_force_diff_n": 1e-12,
    "gate_c_penetration_diff_m": 1e-12,
    "gate_d_position_residual_diff": 1e-12,
    "gate_d_velocity_residual_diff": 1e-12,
    "max_closure_pos_error": _DEFAULT_GATES.max_closure_residual_rad,  # 0.05 rad
    "max_closure_vel_error": _DEFAULT_GATES.max_closure_residual_rad,  # 0.05 rad/s
    "accelerations_all_finite": 1.0,
}

FB4_CALIBRATION_THRESHOLDS: dict[str, float] = {
    "total_rms_m": _DEFAULT_GATES.g3_whole_driver_rmse_m,  # 0.060 m (60 mm)
    "closure_max_error_m": _DEFAULT_GATES.max_closure_residual_m,  # 0.005 m (5 mm)
}

FB5_MATCHING_THRESHOLDS: dict[str, float] = {
    "whole_marker_rmse_m": _DEFAULT_GATES.g3_whole_driver_rmse_m,  # 0.060 m (60 mm)
    "max_normal_force_n": (
        _DEFAULT_GATES.nominal_body_mass_kg
        * _DEFAULT_GATES.gravity_m_s2
        * _DEFAULT_GATES.max_normal_force_bw_multiplier
    ),  # 2354.4 N
    "max_penetration_m": _DEFAULT_GATES.max_penetration_m,  # 0.010 m (10 mm)
    "max_closure_residual_m": _DEFAULT_GATES.max_closure_residual_m,  # 0.005 m (5 mm)
    "max_defect_norm": _DEFAULT_GATES.max_collocation_defect_m,  # 0.005 m (5 mm)
}

FB6_PARITY_THRESHOLDS: dict[str, float] = {
    "whole_marker_rmse_m": _DEFAULT_GATES.g3_whole_driver_rmse_m,  # 0.060 m (60 mm)
    "max_normal_force_n": (
        _DEFAULT_GATES.nominal_body_mass_kg
        * _DEFAULT_GATES.gravity_m_s2
        * _DEFAULT_GATES.max_normal_force_bw_multiplier
    ),  # 2354.4 N
    "max_penetration_m": _DEFAULT_GATES.max_penetration_m,  # 0.010 m (10 mm)
    "max_closure_residual_m": _DEFAULT_GATES.max_closure_residual_m,  # 0.005 m (5 mm)
    "max_q_difference": 0.05,  # 0.05 rad step-size convergence
}

# Pass only on a literal True; their threshold entry just registers the gate.
BOOLEAN_GATES: frozenset[str] = frozenset({"accelerations_all_finite"})

KNOWN_GATES: frozenset[str] = frozenset(
    set(FB3_DRAKE_THRESHOLDS.keys())
    | set(FB4_CALIBRATION_THRESHOLDS.keys())
    | set(FB5_MATCHING_THRESHOLDS.keys())
    | set(FB6_PARITY_THRESHOLDS.keys())
    | {
        "early_marker_rmse_m",
        "terminal_marker_rmse_m",
        "club_marker_rmse_m",
        "pelvis_yaw_rmse_rad",
        "sub_total_rms_m",
        "full_total_rms_m",
        "mean_frame_rms_m",
        "max_frame_rms_m",
        "closure_mean_error_m",
        "max_closure_residual_rad",
        "max_friction_force_n",
        "max_qd_difference",
        "max_marker_rmse_diff_m",
    }
)


@precondition(
    lambda metrics, thresholds, known_gates=None: (
        isinstance(metrics, Mapping) and isinstance(thresholds, Mapping)
    )
)
@postcondition(lambda result: isinstance(result, dict) and "status" in result)
def evaluate_gates(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, float],
    known_gates: Container[str] | None = None,
) -> dict[str, Any]:
    """Evaluate measured metrics against quantitative thresholds.

    Preconditions (DbC):
    - All gate names in `thresholds` must be known.
    - All gate names in `metrics` must be known.
    - All threshold values must be non-negative.

    Returns:
    - Per-gate dictionary mapping each gate to ``{value, threshold, passed, [reason]}``.
    - Top-level ``status``: "PASSED" only if every gate has a finite value within threshold;
      otherwise "FAILED" with diagnostic reasons.
    - If `thresholds` is empty, returns "DIAGNOSTIC" with an explanatory note.
    """
    valid_gates = known_gates if known_gates is not None else KNOWN_GATES

    # Validate gate names and non-negative thresholds
    for gate_name, thresh in thresholds.items():
        if gate_name not in valid_gates:
            raise ValueError(f"Unknown gate name in thresholds: {gate_name}")
        if not isinstance(thresh, (int, float)) or thresh < 0.0:
            raise ValueError(
                f"Threshold for {gate_name} cannot be negative, got {thresh}"
            )

    for gate_name in metrics:
        if gate_name not in valid_gates:
            raise ValueError(f"Unknown gate name in metrics: {gate_name}")

    if not thresholds:
        return {
            "status": "DIAGNOSTIC",
            "passed": False,
            "gates": {},
            "thresholds": {},
            "note": "No documented thresholds provided; status evaluated as DIAGNOSTIC (#10960)",
        }

    per_gate: dict[str, dict[str, Any]] = {}
    all_passed = True

    for gate_name, thresh in thresholds.items():
        threshold_float = float(thresh)
        if gate_name not in metrics:
            all_passed = False
            per_gate[gate_name] = {
                "value": None,
                "threshold": threshold_float,
                "passed": False,
                "reason": f"missing metric: {gate_name}",
            }
            continue

        raw_val = metrics[gate_name]
        if raw_val is None:
            all_passed = False
            per_gate[gate_name] = {
                "value": None,
                "threshold": threshold_float,
                "passed": False,
                "reason": f"metric {gate_name} is None",
            }
            continue

        if gate_name in BOOLEAN_GATES or isinstance(raw_val, bool):
            gate_ok = raw_val is True
            val_f = 1.0 if gate_ok else 0.0
            if not gate_ok:
                all_passed = False
            per_gate[gate_name] = {
                "value": val_f,
                "threshold": threshold_float,
                "passed": gate_ok,
                **({} if gate_ok else {"reason": f"metric {gate_name} is not True"}),
            }
            continue

        try:
            val_f = float(raw_val)
        except (ValueError, TypeError):
            all_passed = False
            per_gate[gate_name] = {
                "value": None,
                "threshold": threshold_float,
                "passed": False,
                "reason": f"metric {gate_name} is not numeric: {raw_val}",
            }
            continue

        if not math.isfinite(val_f) or math.isnan(val_f):
            all_passed = False
            per_gate[gate_name] = {
                "value": val_f,
                "threshold": threshold_float,
                "passed": False,
                "reason": f"metric {gate_name} is non-finite: {val_f}",
            }
        elif val_f > threshold_float:
            all_passed = False
            per_gate[gate_name] = {
                "value": val_f,
                "threshold": threshold_float,
                "passed": False,
                "reason": f"{gate_name} {val_f} > threshold {threshold_float}",
            }
        else:
            per_gate[gate_name] = {
                "value": val_f,
                "threshold": threshold_float,
                "passed": True,
            }

    status = "PASSED" if all_passed else "FAILED"
    return {
        "status": status,
        "passed": all_passed,
        "gates": per_gate,
        "thresholds": dict(thresholds),
    }
