"""Physical acceptance contract and evaluation engine for tour motion matching (MS-01, #10322).

Defines the single source of truth for physical and kinematic acceptance across all
engines and horizons (G1, G2, G3) under the Matched Swing Program.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import enum
import math
from typing import Any

from src.shared.python.contracts import precondition, postcondition


class Horizon(enum.Enum):
    """Evaluation horizon for the matched swing program."""

    G1 = "G1"  # 0 to 0.85 s (top of backswing / early transition)
    G2 = "G2"  # through impact (~1.20 s)
    G3 = "G3"  # full capture (1.814 s driver / 1.827 s iron)


class GateStatus(enum.Enum):
    """Evaluation outcome for an individual gate."""

    PASSED = "passed"
    FAILED = "failed"
    MISSING = "missing"


@dataclass(frozen=True)
class GateResult:
    """Outcome of evaluating one quantitative gate."""

    name: str
    status: GateStatus
    threshold: float
    measured: float | None = None
    unit: str = "m"
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "threshold": self.threshold,
            "measured": self.measured,
            "unit": self.unit,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AcceptanceGates:
    """Frozen thresholds defining physical and kinematic acceptance per horizon."""

    # Kinematic thresholds (RMSE in metres, yaw in rad or pct)
    # G1 (0 to 0.85 s)
    g1_whole_rmse_m: float = 0.025
    g1_early_rmse_m: float = 0.012
    g1_terminal_rmse_m: float = 0.035
    g1_club_rmse_m: float = 0.060
    g1_pelvis_yaw_rmse_rad: float = 0.05236  # < 3 deg (~0.0524 rad)
    g1_pelvis_yaw_error_pct: float = 5.0

    # G2 (through impact)
    g2_whole_rmse_m: float = 0.040
    g2_early_rmse_m: float = 0.015
    g2_terminal_rmse_m: float = 0.050
    g2_club_rmse_m: float = 0.075
    g2_pelvis_yaw_rmse_rad: float = 0.08727  # < 5 deg

    # G3 (full swing: driver <= 60 mm, iron <= 95 mm)
    g3_whole_driver_rmse_m: float = 0.060
    g3_whole_iron_rmse_m: float = 0.095
    g3_early_rmse_m: float = 0.020
    g3_terminal_rmse_m: float = 0.080
    g3_club_rmse_m: float = 0.100
    g3_pelvis_yaw_rmse_rad: float = 0.10472  # < 6 deg

    # Physical limits (shared across horizons unless specified)
    version: str = "v1.1"
    max_normal_force_bw_multiplier: float = 3.0
    nominal_body_mass_kg: float = 80.0
    gravity_m_s2: float = 9.81
    max_penetration_m: float = 0.010  # 10 mm
    max_closure_residual_m: float = 0.005  # 5 mm
    max_closure_residual_rad: float = 0.05  # 0.05 rad
    max_friction_coefficient: float = 0.8
    max_root_force_n: float = 0.1  # 0.1 N maximum allowed ungrounded root assistance
    min_duration_g1_s: float = 0.80
    min_duration_g2_s: float = 1.15
    min_duration_g3_s: float = 1.75
    weight_fraction_min: float = 0.20
    weight_fraction_max: float = 3.00
    min_inside_support_polygon_fraction: float = 0.85  # 85 % of frames


@dataclass(frozen=True)
class AcceptanceVerdict:
    """Comprehensive acceptance decision for a receipt or comparison outcome."""

    horizon: Horizon
    is_physically_accepted: bool
    status: str
    gates: tuple[GateResult, ...]
    qualification_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon.value,
            "is_physically_accepted": self.is_physically_accepted,
            "status": self.status,
            "gates": [g.as_dict() for g in self.gates],
            "qualification_note": self.qualification_note,
        }


def _extract_metric(data: Mapping[str, Any], *keys: str) -> float | None:
    """Recursively search for metric value under candidate keys or nested dicts."""
    for key in keys:
        if key in data:
            val = data[key]
            if isinstance(val, (int, float)) and not math.isnan(val):
                return float(val)
    # Check nested containers: shared_metrics, restart_uninterrupted_metrics, zero_displacement_parity
    for sub in (
        "shared_metrics",
        "restart_uninterrupted_metrics",
        "zero_displacement_parity",
        "dynamics",
        "ik",
        "forward_rollout",
    ):
        nested = data.get(sub)
        if isinstance(nested, Mapping):
            res = _extract_metric(nested, *keys)
            if res is not None:
                return res
    return None


def _check_rmse_gate(
    name: str,
    measured: float | None,
    threshold: float,
    label: str,
) -> GateResult | None:
    if measured is None:
        return None
    if measured <= threshold:
        return GateResult(
            name=name,
            status=GateStatus.PASSED,
            threshold=threshold,
            measured=measured,
        )
    return GateResult(
        name=name,
        status=GateStatus.FAILED,
        threshold=threshold,
        measured=measured,
        reason=f"{label} RMSE {measured * 1e3:.2f} mm > {threshold * 1e3:.2f} mm",
    )


def _evaluate_marker_rmse(
    receipt: Mapping[str, Any],
    horizon: Horizon,
    gates: AcceptanceGates,
) -> list[GateResult]:
    results: list[GateResult] = []
    capture = str(receipt.get("capture", "driver")).lower()
    if horizon == Horizon.G1:
        thresh_whole = gates.g1_whole_rmse_m
    elif horizon == Horizon.G2:
        thresh_whole = gates.g2_whole_rmse_m
    else:
        thresh_whole = (
            gates.g3_whole_iron_rmse_m
            if "iron" in capture
            else gates.g3_whole_driver_rmse_m
        )

    val_whole = _extract_metric(
        receipt, "whole_marker_rmse_m", "whole_rms_m", "marker_rms_m"
    )
    if val_whole is None:
        results.append(
            GateResult(
                name="whole_marker_rmse_m",
                status=GateStatus.MISSING,
                threshold=thresh_whole,
                reason="missing whole marker RMSE",
            )
        )
    elif not math.isfinite(val_whole):
        results.append(
            GateResult(
                name="whole_marker_rmse_m",
                status=GateStatus.FAILED,
                threshold=thresh_whole,
                measured=None,
                reason="invalid predictions or empty valid marker population (non-finite RMSE)",
            )
        )
    else:
        r_whole = _check_rmse_gate(
            "whole_marker_rmse_m", val_whole, thresh_whole, "whole"
        )
        if r_whole is not None:
            results.append(r_whole)

    thresh_early = (
        gates.g1_early_rmse_m
        if horizon == Horizon.G1
        else (gates.g2_early_rmse_m if horizon == Horizon.G2 else gates.g3_early_rmse_m)
    )
    val_early = _extract_metric(receipt, "early_marker_rmse_m", "early_rms_m")
    r_early = _check_rmse_gate("early_marker_rmse_m", val_early, thresh_early, "early")
    if r_early is not None:
        results.append(r_early)

    thresh_terminal = (
        gates.g1_terminal_rmse_m
        if horizon == Horizon.G1
        else (
            gates.g2_terminal_rmse_m
            if horizon == Horizon.G2
            else gates.g3_terminal_rmse_m
        )
    )
    val_terminal = _extract_metric(receipt, "terminal_marker_rmse_m", "terminal_rms_m")
    r_terminal = _check_rmse_gate(
        "terminal_marker_rmse_m", val_terminal, thresh_terminal, "terminal"
    )
    if r_terminal is not None:
        results.append(r_terminal)

    thresh_club = (
        gates.g1_club_rmse_m
        if horizon == Horizon.G1
        else (gates.g2_club_rmse_m if horizon == Horizon.G2 else gates.g3_club_rmse_m)
    )
    val_club = _extract_metric(receipt, "club_marker_rmse_m", "club_cluster_rms_m")
    r_club = _check_rmse_gate("club_marker_rmse_m", val_club, thresh_club, "club")
    if r_club is not None:
        results.append(r_club)

    return results


def _evaluate_pelvis_yaw(
    receipt: Mapping[str, Any],
    horizon: Horizon,
    gates: AcceptanceGates,
) -> list[GateResult]:
    results: list[GateResult] = []
    thresh_yaw_rad = (
        gates.g1_pelvis_yaw_rmse_rad
        if horizon == Horizon.G1
        else (
            gates.g2_pelvis_yaw_rmse_rad
            if horizon == Horizon.G2
            else gates.g3_pelvis_yaw_rmse_rad
        )
    )
    val_yaw_rad = _extract_metric(receipt, "pelvis_yaw_rmse_rad")
    val_yaw_pct = _extract_metric(receipt, "pelvis_yaw_error_pct")
    val_yaw_deg = _extract_metric(receipt, "pelvis_yaw_diff_deg")
    if val_yaw_rad is not None:
        if val_yaw_rad <= thresh_yaw_rad:
            results.append(
                GateResult(
                    name="pelvis_yaw_rmse_rad",
                    status=GateStatus.PASSED,
                    threshold=thresh_yaw_rad,
                    measured=val_yaw_rad,
                    unit="rad",
                )
            )
        else:
            results.append(
                GateResult(
                    name="pelvis_yaw_rmse_rad",
                    status=GateStatus.FAILED,
                    threshold=thresh_yaw_rad,
                    measured=val_yaw_rad,
                    unit="rad",
                    reason=f"pelvis yaw RMSE {val_yaw_rad:.4f} rad > {thresh_yaw_rad:.4f} rad",
                )
            )
    elif val_yaw_pct is not None:
        thresh_pct = gates.g1_pelvis_yaw_error_pct
        if val_yaw_pct <= thresh_pct:
            results.append(
                GateResult(
                    name="pelvis_yaw_error_pct",
                    status=GateStatus.PASSED,
                    threshold=thresh_pct,
                    measured=val_yaw_pct,
                    unit="%",
                )
            )
        else:
            results.append(
                GateResult(
                    name="pelvis_yaw_error_pct",
                    status=GateStatus.FAILED,
                    threshold=thresh_pct,
                    measured=val_yaw_pct,
                    unit="%",
                    reason=f"pelvis yaw error {val_yaw_pct:.2f}% > {thresh_pct:.2f}%",
                )
            )
    elif val_yaw_deg is not None:
        thresh_deg = math.degrees(thresh_yaw_rad)
        if abs(val_yaw_deg) <= thresh_deg:
            results.append(
                GateResult(
                    name="pelvis_yaw_diff_deg",
                    status=GateStatus.PASSED,
                    threshold=thresh_deg,
                    measured=val_yaw_deg,
                    unit="deg",
                )
            )
        else:
            results.append(
                GateResult(
                    name="pelvis_yaw_diff_deg",
                    status=GateStatus.FAILED,
                    threshold=thresh_deg,
                    measured=val_yaw_deg,
                    unit="deg",
                    reason=f"pelvis yaw diff {val_yaw_deg:.2f} deg > {thresh_deg:.2f} deg",
                )
            )
    return results


def _evaluate_normal_contact_force(
    receipt: Mapping[str, Any],
    gates: AcceptanceGates,
    contact_audit: Any,
) -> list[GateResult]:
    results: list[GateResult] = []
    mass_kg = gates.nominal_body_mass_kg
    if (
        isinstance(receipt.get("anthropometric"), (list, tuple))
        and len(receipt["anthropometric"]) >= 2
    ):
        mass_kg = float(receipt["anthropometric"][1])
    thresh_max_force = (
        mass_kg * gates.gravity_m_s2 * gates.max_normal_force_bw_multiplier
    )

    val_force = None
    if isinstance(contact_audit, Mapping):
        val_force = _extract_metric(contact_audit, "max_normal_force_n")
    if val_force is None:
        val_force = _extract_metric(receipt, "max_normal_force_n")

    is_native_crossval = "native" in str(
        receipt.get("lane", "")
    ) or "two_window_fit" in str(receipt)
    if val_force is None:
        if not is_native_crossval and "contact_audit" in receipt:
            results.append(
                GateResult(
                    name="max_normal_force_n",
                    status=GateStatus.FAILED,
                    threshold=thresh_max_force,
                    unit="N",
                    reason="missing max normal force in contact audit",
                )
            )
        elif not is_native_crossval:
            dyn = receipt.get("dynamics")
            if isinstance(dyn, Mapping) and "controller" in dyn:
                results.append(
                    GateResult(
                        name="max_normal_force_n",
                        status=GateStatus.FAILED,
                        threshold=thresh_max_force,
                        unit="N",
                        reason="missing contact forces in dynamics receipt",
                    )
                )
            else:
                results.append(
                    GateResult(
                        name="max_normal_force_n",
                        status=GateStatus.MISSING,
                        threshold=thresh_max_force,
                        unit="N",
                        reason="missing dynamics.max_normal_force_n",
                    )
                )
    else:
        if val_force <= thresh_max_force:
            results.append(
                GateResult(
                    name="max_normal_force_n",
                    status=GateStatus.PASSED,
                    threshold=thresh_max_force,
                    measured=val_force,
                    unit="N",
                )
            )
        else:
            results.append(
                GateResult(
                    name="max_normal_force_n",
                    status=GateStatus.FAILED,
                    threshold=thresh_max_force,
                    measured=val_force,
                    unit="N",
                    reason=f"max normal force {val_force:.1f} N exceeds {thresh_max_force:.1f} N (3x BW)",
                )
            )
    return results


def _evaluate_penetration(
    receipt: Mapping[str, Any],
    gates: AcceptanceGates,
    contact_audit: Any,
) -> list[GateResult]:
    results: list[GateResult] = []
    thresh_penetration = gates.max_penetration_m
    val_penetration = None
    if isinstance(contact_audit, Mapping):
        val_penetration = _extract_metric(contact_audit, "max_penetration_m")
    if val_penetration is None:
        val_penetration = _extract_metric(receipt, "max_penetration_m")
    if val_penetration is not None:
        if val_penetration <= thresh_penetration:
            results.append(
                GateResult(
                    name="max_penetration_m",
                    status=GateStatus.PASSED,
                    threshold=thresh_penetration,
                    measured=val_penetration,
                    unit="m",
                )
            )
        else:
            results.append(
                GateResult(
                    name="max_penetration_m",
                    status=GateStatus.FAILED,
                    threshold=thresh_penetration,
                    measured=val_penetration,
                    unit="m",
                    reason=f"penetration {val_penetration * 1e3:.1f} mm > {thresh_penetration * 1e3:.1f} mm",
                )
            )
    return results


def _evaluate_ground_and_closure(
    receipt: Mapping[str, Any],
    gates: AcceptanceGates,
    contact_audit: Any,
) -> list[GateResult]:
    results = _evaluate_penetration(receipt, gates, contact_audit)
    thresh_closure_m = gates.max_closure_residual_m
    val_closure_m = _extract_metric(
        receipt,
        "max_closure_residual_m",
        "closure_error_max_m",
        "max_closure_translation_m",
        "reference_closure_max_abs",
    )
    if val_closure_m is not None:
        if val_closure_m <= thresh_closure_m:
            results.append(
                GateResult(
                    name="max_closure_residual_m",
                    status=GateStatus.PASSED,
                    threshold=thresh_closure_m,
                    measured=val_closure_m,
                    unit="m",
                )
            )
        else:
            results.append(
                GateResult(
                    name="max_closure_residual_m",
                    status=GateStatus.FAILED,
                    threshold=thresh_closure_m,
                    measured=val_closure_m,
                    unit="m",
                    reason=f"closure residual {val_closure_m * 1e3:.2f} mm > {thresh_closure_m * 1e3:.2f} mm",
                )
            )

    thresh_rot = gates.max_closure_residual_rad
    val_rot = None
    closure_dict = receipt.get("closure")
    if isinstance(closure_dict, Mapping):
        val_rot = _extract_metric(
            closure_dict, "max_closure_rotation_rad", "closure_rotation_max_rad"
        )
    if val_rot is None:
        val_rot = _extract_metric(
            receipt, "max_closure_rotation_rad", "closure_rotation_max_rad"
        )

    if val_rot is not None:
        if val_rot <= thresh_rot:
            results.append(
                GateResult(
                    name="closure_rotation_rad",
                    status=GateStatus.PASSED,
                    threshold=thresh_rot,
                    measured=val_rot,
                    unit="rad",
                )
            )
        else:
            results.append(
                GateResult(
                    name="closure_rotation_rad",
                    status=GateStatus.FAILED,
                    threshold=thresh_rot,
                    measured=val_rot,
                    unit="rad",
                    reason=f"closure rotation {val_rot:.4f} rad > {thresh_rot:.4f} rad",
                )
            )
    return results


def _evaluate_weight_fraction(
    receipt: Mapping[str, Any],
    gates: AcceptanceGates,
) -> list[GateResult]:
    results: list[GateResult] = []
    weight_frac = (
        receipt.get("dynamics", {}).get("weight_fraction")
        if isinstance(receipt.get("dynamics"), Mapping)
        else None
    )
    if isinstance(weight_frac, Mapping):
        w_min = weight_frac.get("min")
        w_max = weight_frac.get("max")
        if w_min is not None and w_max is not None:
            if (
                float(w_min) >= gates.weight_fraction_min
                and float(w_max) <= gates.weight_fraction_max
            ):
                results.append(
                    GateResult(
                        name="weight_fraction",
                        status=GateStatus.PASSED,
                        threshold=gates.weight_fraction_min,
                        measured=float(w_min),
                        unit="BW",
                    )
                )
            else:
                results.append(
                    GateResult(
                        name="weight_fraction",
                        status=GateStatus.FAILED,
                        threshold=gates.weight_fraction_min,
                        measured=float(w_min),
                        unit="BW",
                        reason=f"weight fraction range [{w_min:.2f}, {w_max:.2f}] outside [{gates.weight_fraction_min}, {gates.weight_fraction_max}]",
                    )
                )
    return results


def _evaluate_friction_cone(
    receipt: Mapping[str, Any],
    gates: AcceptanceGates,
    contact_audit: Any,
) -> list[GateResult]:
    results: list[GateResult] = []
    val_ratio = None
    if isinstance(contact_audit, Mapping):
        val_ratio = _extract_metric(contact_audit, "max_friction_ratio")
    if val_ratio is None:
        val_ratio = _extract_metric(receipt, "max_friction_ratio")

    if val_ratio is not None:
        if val_ratio <= gates.max_friction_coefficient:
            results.append(
                GateResult(
                    name="friction_cone",
                    status=GateStatus.PASSED,
                    threshold=gates.max_friction_coefficient,
                    measured=val_ratio,
                    unit="ratio",
                )
            )
        else:
            results.append(
                GateResult(
                    name="friction_cone",
                    status=GateStatus.FAILED,
                    threshold=gates.max_friction_coefficient,
                    measured=val_ratio,
                    unit="ratio",
                    reason=f"friction ratio {val_ratio:.2f} exceeds mu={gates.max_friction_coefficient:.2f}",
                )
            )
    return results


def _evaluate_root_assistance(
    receipt: Mapping[str, Any],
    gates: AcceptanceGates,
) -> list[GateResult]:
    results: list[GateResult] = []
    dyn = receipt.get("dynamics")
    has_dynamics = isinstance(dyn, Mapping)
    val_root = None
    if isinstance(dyn, Mapping):
        val_root = _extract_metric(dyn, "max_root_force_n", "delta_tau_root_max_n")
    if val_root is None:
        val_root = _extract_metric(receipt, "max_root_force_n", "delta_tau_root_max_n")

    if has_dynamics:
        if val_root is None:
            results.append(
                GateResult(
                    name="root_assistance",
                    status=GateStatus.MISSING,
                    threshold=gates.max_root_force_n,
                    unit="N",
                    reason="missing root assistance history (delta_tau_root / max_root_force_n)",
                )
            )
        elif val_root <= gates.max_root_force_n:
            results.append(
                GateResult(
                    name="root_assistance",
                    status=GateStatus.PASSED,
                    threshold=gates.max_root_force_n,
                    measured=val_root,
                    unit="N",
                )
            )
        else:
            results.append(
                GateResult(
                    name="root_assistance",
                    status=GateStatus.FAILED,
                    threshold=gates.max_root_force_n,
                    measured=val_root,
                    unit="N",
                    reason=f"phantom root assistance {val_root:.2f} N > {gates.max_root_force_n:.2f} N",
                )
            )
    return results


def _evaluate_horizon_duration(
    receipt: Mapping[str, Any],
    horizon: Horizon,
    gates: AcceptanceGates,
) -> list[GateResult]:
    results: list[GateResult] = []
    dur = _extract_metric(receipt, "duration_s")
    if dur is None:
        hor_dict = receipt.get("horizon")
        if isinstance(hor_dict, Mapping):
            t_end = hor_dict.get("t_end_s")
            t_start = hor_dict.get("t_start_s", 0.0)
            if isinstance(t_end, (int, float)) and isinstance(t_start, (int, float)):
                dur = float(t_end - t_start)

    if dur is not None:
        thresh = (
            gates.min_duration_g3_s
            if horizon == Horizon.G3
            else (
                gates.min_duration_g2_s
                if horizon == Horizon.G2
                else gates.min_duration_g1_s
            )
        )
        if dur >= thresh:
            results.append(
                GateResult(
                    name="horizon_duration_s",
                    status=GateStatus.PASSED,
                    threshold=thresh,
                    measured=dur,
                    unit="s",
                )
            )
        else:
            results.append(
                GateResult(
                    name="horizon_duration_s",
                    status=GateStatus.FAILED,
                    threshold=thresh,
                    measured=dur,
                    unit="s",
                    reason=f"duration {dur:.3f} s is truncated for horizon {horizon.value} (minimum required: {thresh:.3f} s)",
                )
            )
    return results


@precondition(
    lambda receipt, horizon=Horizon.G1, gates=None: isinstance(horizon, Horizon),
    "horizon must be Horizon enum",
)
@postcondition(
    lambda verdict: all(g.measured is not None or g.reason for g in verdict.gates),
    "every gate must have measured or reason",
)
def evaluate(
    receipt: Mapping[str, Any],
    *,
    horizon: Horizon = Horizon.G1,
    gates: AcceptanceGates | None = None,
) -> AcceptanceVerdict:
    """Pure evaluation function that returns physical & kinematic acceptance verdict."""
    if gates is None:
        gates = AcceptanceGates()

    contact_audit = receipt.get("contact_audit")
    gate_results: list[GateResult] = []
    gate_results.extend(_evaluate_marker_rmse(receipt, horizon, gates))
    gate_results.extend(_evaluate_pelvis_yaw(receipt, horizon, gates))
    gate_results.extend(_evaluate_normal_contact_force(receipt, gates, contact_audit))
    gate_results.extend(_evaluate_ground_and_closure(receipt, gates, contact_audit))
    gate_results.extend(_evaluate_weight_fraction(receipt, gates))
    gate_results.extend(_evaluate_friction_cone(receipt, gates, contact_audit))
    gate_results.extend(_evaluate_root_assistance(receipt, gates))
    gate_results.extend(_evaluate_horizon_duration(receipt, horizon, gates))

    # Overall verdict
    is_accepted = len(gate_results) > 0 and all(
        g.status == GateStatus.PASSED for g in gate_results
    )
    status_str = "PASSED" if is_accepted else "REJECTED"

    return AcceptanceVerdict(
        horizon=horizon,
        is_physically_accepted=is_accepted,
        status=status_str,
        gates=tuple(gate_results),
        qualification_note=(
            "Physical acceptance criteria passed"
            if is_accepted
            else "Physical or kinematic thresholds violated"
        ),
    )
