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
    max_normal_force_bw_multiplier: float = 3.0
    nominal_body_mass_kg: float = 80.0
    gravity_m_s2: float = 9.81
    max_penetration_m: float = 0.010  # 10 mm
    max_closure_residual_m: float = 0.005  # 5 mm
    max_closure_residual_rad: float = 0.05  # 0.05 rad
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

    gate_results: list[GateResult] = []

    # 1. Whole-marker RMSE
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
        gate_results.append(
            GateResult(
                name="whole_marker_rmse_m",
                status=GateStatus.MISSING,
                threshold=thresh_whole,
                reason="missing whole marker RMSE",
            )
        )
    elif val_whole <= thresh_whole:
        gate_results.append(
            GateResult(
                name="whole_marker_rmse_m",
                status=GateStatus.PASSED,
                threshold=thresh_whole,
                measured=val_whole,
            )
        )
    else:
        gate_results.append(
            GateResult(
                name="whole_marker_rmse_m",
                status=GateStatus.FAILED,
                threshold=thresh_whole,
                measured=val_whole,
                reason=f"whole RMSE {val_whole * 1e3:.2f} mm > {thresh_whole * 1e3:.2f} mm",
            )
        )

    # 2. Early marker RMSE
    thresh_early = (
        gates.g1_early_rmse_m
        if horizon == Horizon.G1
        else (gates.g2_early_rmse_m if horizon == Horizon.G2 else gates.g3_early_rmse_m)
    )
    val_early = _extract_metric(receipt, "early_marker_rmse_m", "early_rms_m")
    if val_early is not None:
        if val_early <= thresh_early:
            gate_results.append(
                GateResult(
                    name="early_marker_rmse_m",
                    status=GateStatus.PASSED,
                    threshold=thresh_early,
                    measured=val_early,
                )
            )
        else:
            gate_results.append(
                GateResult(
                    name="early_marker_rmse_m",
                    status=GateStatus.FAILED,
                    threshold=thresh_early,
                    measured=val_early,
                    reason=f"early RMSE {val_early * 1e3:.2f} mm > {thresh_early * 1e3:.2f} mm",
                )
            )

    # 3. Terminal marker RMSE
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
    if val_terminal is not None:
        if val_terminal <= thresh_terminal:
            gate_results.append(
                GateResult(
                    name="terminal_marker_rmse_m",
                    status=GateStatus.PASSED,
                    threshold=thresh_terminal,
                    measured=val_terminal,
                )
            )
        else:
            gate_results.append(
                GateResult(
                    name="terminal_marker_rmse_m",
                    status=GateStatus.FAILED,
                    threshold=thresh_terminal,
                    measured=val_terminal,
                    reason=f"terminal RMSE {val_terminal * 1e3:.2f} mm > {thresh_terminal * 1e3:.2f} mm",
                )
            )

    # 4. Club marker RMSE
    thresh_club = (
        gates.g1_club_rmse_m
        if horizon == Horizon.G1
        else (gates.g2_club_rmse_m if horizon == Horizon.G2 else gates.g3_club_rmse_m)
    )
    val_club = _extract_metric(receipt, "club_marker_rmse_m", "club_cluster_rms_m")
    if val_club is not None:
        if val_club <= thresh_club:
            gate_results.append(
                GateResult(
                    name="club_marker_rmse_m",
                    status=GateStatus.PASSED,
                    threshold=thresh_club,
                    measured=val_club,
                )
            )
        else:
            gate_results.append(
                GateResult(
                    name="club_marker_rmse_m",
                    status=GateStatus.FAILED,
                    threshold=thresh_club,
                    measured=val_club,
                    reason=f"club RMSE {val_club * 1e3:.2f} mm > {thresh_club * 1e3:.2f} mm",
                )
            )

    # 5. Pelvis yaw
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
            gate_results.append(
                GateResult(
                    name="pelvis_yaw_rmse_rad",
                    status=GateStatus.PASSED,
                    threshold=thresh_yaw_rad,
                    measured=val_yaw_rad,
                    unit="rad",
                )
            )
        else:
            gate_results.append(
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
            gate_results.append(
                GateResult(
                    name="pelvis_yaw_error_pct",
                    status=GateStatus.PASSED,
                    threshold=thresh_pct,
                    measured=val_yaw_pct,
                    unit="%",
                )
            )
        else:
            gate_results.append(
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
            gate_results.append(
                GateResult(
                    name="pelvis_yaw_diff_deg",
                    status=GateStatus.PASSED,
                    threshold=thresh_deg,
                    measured=val_yaw_deg,
                    unit="deg",
                )
            )
        else:
            gate_results.append(
                GateResult(
                    name="pelvis_yaw_diff_deg",
                    status=GateStatus.FAILED,
                    threshold=thresh_deg,
                    measured=val_yaw_deg,
                    unit="deg",
                    reason=f"pelvis yaw diff {val_yaw_deg:.2f} deg > {thresh_deg:.2f} deg",
                )
            )

    # 6. Physical limit: Max normal contact force <= 3x body weight (~2354 N for 80kg)
    # Extract subject mass if present
    mass_kg = gates.nominal_body_mass_kg
    if (
        isinstance(receipt.get("anthropometric"), (list, tuple))
        and len(receipt["anthropometric"]) >= 2
    ):
        mass_kg = float(receipt["anthropometric"][1])
    thresh_max_force = (
        mass_kg * gates.gravity_m_s2 * gates.max_normal_force_bw_multiplier
    )

    # Normal contact force lookup (contact_audit, dynamics, or direct)
    val_force = None
    contact_audit = receipt.get("contact_audit")
    if isinstance(contact_audit, Mapping):
        val_force = _extract_metric(contact_audit, "max_normal_force_n")
    if val_force is None:
        val_force = _extract_metric(receipt, "max_normal_force_n")

    # Native fits without explicit feet contact forces (e.g. Simscape native model)
    is_native_crossval = "native" in str(
        receipt.get("lane", "")
    ) or "two_window_fit" in str(receipt)
    if val_force is None:
        if not is_native_crossval and "contact_audit" in receipt:
            gate_results.append(
                GateResult(
                    name="max_normal_force_n",
                    status=GateStatus.FAILED,
                    threshold=thresh_max_force,
                    unit="N",
                    reason="missing max normal force in contact audit",
                )
            )
        elif not is_native_crossval:
            # Check dynamics controller
            dyn = receipt.get("dynamics")
            if isinstance(dyn, Mapping) and "controller" in dyn:
                gate_results.append(
                    GateResult(
                        name="max_normal_force_n",
                        status=GateStatus.FAILED,
                        threshold=thresh_max_force,
                        unit="N",
                        reason="missing contact forces in dynamics receipt",
                    )
                )
            else:
                gate_results.append(
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
            gate_results.append(
                GateResult(
                    name="max_normal_force_n",
                    status=GateStatus.PASSED,
                    threshold=thresh_max_force,
                    measured=val_force,
                    unit="N",
                )
            )
        else:
            gate_results.append(
                GateResult(
                    name="max_normal_force_n",
                    status=GateStatus.FAILED,
                    threshold=thresh_max_force,
                    measured=val_force,
                    unit="N",
                    reason=f"max normal force {val_force:.1f} N exceeds {thresh_max_force:.1f} N (3x BW)",
                )
            )

    # 7. Physical limit: Ground penetration <= 10 mm
    thresh_penetration = gates.max_penetration_m
    val_penetration = None
    if isinstance(contact_audit, Mapping):
        val_penetration = _extract_metric(contact_audit, "max_penetration_m")
    if val_penetration is None:
        val_penetration = _extract_metric(receipt, "max_penetration_m")
    if val_penetration is not None:
        if val_penetration <= thresh_penetration:
            gate_results.append(
                GateResult(
                    name="max_penetration_m",
                    status=GateStatus.PASSED,
                    threshold=thresh_penetration,
                    measured=val_penetration,
                    unit="m",
                )
            )
        else:
            gate_results.append(
                GateResult(
                    name="max_penetration_m",
                    status=GateStatus.FAILED,
                    threshold=thresh_penetration,
                    measured=val_penetration,
                    unit="m",
                    reason=f"penetration {val_penetration * 1e3:.1f} mm > {thresh_penetration * 1e3:.1f} mm",
                )
            )

    # 8. Physical limit: Loop closure residual <= 5 mm / 0.05 rad
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
            gate_results.append(
                GateResult(
                    name="max_closure_residual_m",
                    status=GateStatus.PASSED,
                    threshold=thresh_closure_m,
                    measured=val_closure_m,
                    unit="m",
                )
            )
        else:
            gate_results.append(
                GateResult(
                    name="max_closure_residual_m",
                    status=GateStatus.FAILED,
                    threshold=thresh_closure_m,
                    measured=val_closure_m,
                    unit="m",
                    reason=f"closure residual {val_closure_m * 1e3:.2f} mm > {thresh_closure_m * 1e3:.2f} mm",
                )
            )

    # 9. Weight fraction within [0.2, 3.0]
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
                gate_results.append(
                    GateResult(
                        name="weight_fraction",
                        status=GateStatus.PASSED,
                        threshold=gates.weight_fraction_min,
                        measured=float(w_min),
                        unit="BW",
                    )
                )
            else:
                gate_results.append(
                    GateResult(
                        name="weight_fraction",
                        status=GateStatus.FAILED,
                        threshold=gates.weight_fraction_min,
                        measured=float(w_min),
                        unit="BW",
                        reason=f"weight fraction range [{w_min:.2f}, {w_max:.2f}] outside [{gates.weight_fraction_min}, {gates.weight_fraction_max}]",
                    )
                )

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
        qualification_note="Physical acceptance criteria passed"
        if is_accepted
        else "Physical or kinematic thresholds violated",
    )
