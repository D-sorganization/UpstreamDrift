"""Physical acceptance contract and evaluation engine for tour motion matching (MS-01, #10322).

Defines the single source of truth for physical and kinematic acceptance across all
engines and horizons (G1, G2, G3) under the Matched Swing Program.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    NOT_APPLICABLE = "not_applicable"
    DISCLOSED = "disclosed"


# Kinematic native-fit lanes carry no contact audit; matched exactly, never by substring.
NATIVE_FIT_LANES: frozenset[str] = frozenset(
    {"native", "drake_native_fit", "crocoddyl_native_fit"}
)
ACCEPTABLE_GATE_STATUSES: frozenset[GateStatus] = frozenset(
    {GateStatus.PASSED, GateStatus.DISCLOSED, GateStatus.NOT_APPLICABLE}
)


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


def is_verdict_accepted(gate_results: Sequence[GateResult]) -> bool:
    """Return True iff at least one gate passed and all gates are acceptable."""
    return (
        len(gate_results) > 0
        and any(g.status == GateStatus.PASSED for g in gate_results)
        and all(g.status in ACCEPTABLE_GATE_STATUSES for g in gate_results)
    )


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

    # Dynamic well-posedness artifacts (MS-100 / MS-107)
    max_open_loop_drift_m: float | None = (
        None  # None => tied to horizon's whole_marker_rmse_m (e.g. 25 mm for G1)
    )
    max_integrator_rtol: float = 1e-5  # required declared integrator relative tolerance
    max_collocation_defect_m: float = 0.005  # 5 mm dynamical consistency defect
    max_stabilized_marker_rmse_m: float = 0.040  # 40 mm low-gain PD tracking error

    # Negative fixture and physical consistency gates (#10431)
    max_friction_coefficient: float = 0.80
    max_actuator_torque_n_m: float = 200.0
    max_root_residual_n_m: float = 1e-3
    min_club_marker_coverage_fraction: float = 0.80
    g1_min_duration_s: float = 0.85
    g2_min_duration_s: float = 1.20
    g3_min_duration_s: float = 1.80

    @property
    def g3_whole_rmse_m(self) -> float:
        """Alias for g3_whole_driver_rmse_m for uniform horizon whole-RMSE access."""
        return self.g3_whole_driver_rmse_m


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

    if receipt.get("lane") in NATIVE_FIT_LANES:
        results.append(
            GateResult(
                name="max_normal_force_n",
                status=GateStatus.NOT_APPLICABLE,
                threshold=thresh_max_force,
                unit="N",
                reason="native kinematic-fit lane has no contact audit",
            )
        )
    elif val_force is None:
        if "contact_audit" in receipt:
            results.append(
                GateResult(
                    name="max_normal_force_n",
                    status=GateStatus.FAILED,
                    threshold=thresh_max_force,
                    unit="N",
                    reason="missing max normal force in contact audit",
                )
            )
        else:
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


def _evaluate_ground_and_closure(
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


def _extract_solver_and_integrators(
    receipt: Mapping[str, Any],
    replay_data: Mapping[str, Any],
) -> tuple[str | None, str | None, Mapping[str, Any]]:
    raw_solver = receipt.get("solver")
    solver_block: Mapping[str, Any] = (
        raw_solver if isinstance(raw_solver, Mapping) else {}
    )
    node_integ = solver_block.get("node_integrator") or receipt.get("node_integrator")
    replay_integ = (
        solver_block.get("replay_integrator")
        or replay_data.get("integrator")
        or receipt.get("replay_integrator")
    )
    return (
        str(node_integ) if node_integ is not None else None,
        str(replay_integ) if replay_integ is not None else None,
        solver_block,
    )


def _evaluate_integrator_consistency(
    receipt: Mapping[str, Any],
    replay_data: Mapping[str, Any],
) -> list[GateResult]:
    """Verify solver node_integrator and replay_integrator are identical."""
    results: list[GateResult] = []
    node_integ, replay_integ, _ = _extract_solver_and_integrators(receipt, replay_data)

    if node_integ and replay_integ and node_integ.lower() != replay_integ.lower():
        results.append(
            GateResult(
                name="integrator_consistency",
                status=GateStatus.FAILED,
                threshold=1.0,
                measured=0.0,
                unit="match",
                reason=f"solver node_integrator '{node_integ}' != replay_integrator '{replay_integ}'",
            )
        )
    elif node_integ and replay_integ:
        results.append(
            GateResult(
                name="integrator_consistency",
                status=GateStatus.PASSED,
                threshold=1.0,
                measured=1.0,
                unit="match",
            )
        )
    return results


def _evaluate_integrator_tolerance(
    receipt: Mapping[str, Any],
    replay_data: Mapping[str, Any],
    horizon: Horizon,
    gates: AcceptanceGates,
) -> list[GateResult]:
    """Verify declared integrator relative tolerance satisfies strictness threshold."""
    results: list[GateResult] = []
    node_integ, replay_integ, solver_block = _extract_solver_and_integrators(
        receipt, replay_data
    )

    rtol_val = _extract_metric(replay_data, "rtol", "rk45_rtol", "tolerance")
    if rtol_val is None:
        solver_rtol = solver_block.get("rk45_rtol")
        if isinstance(solver_rtol, (int, float)):
            rtol_val = float(solver_rtol)
        else:
            receipt_rtol = receipt.get("rk45_rtol")
            if isinstance(receipt_rtol, (int, float)):
                rtol_val = float(receipt_rtol)

    if rtol_val is not None:
        if float(rtol_val) > gates.max_integrator_rtol:
            results.append(
                GateResult(
                    name="integrator_tolerance",
                    status=GateStatus.FAILED,
                    threshold=gates.max_integrator_rtol,
                    measured=float(rtol_val),
                    reason=f"declared integrator rtol {float(rtol_val):e} exceeds maximum allowable {gates.max_integrator_rtol:e}",
                )
            )
        else:
            results.append(
                GateResult(
                    name="integrator_tolerance",
                    status=GateStatus.PASSED,
                    threshold=gates.max_integrator_rtol,
                    measured=float(rtol_val),
                )
            )
    elif node_integ in ("rk45", "runge_kutta_45") or replay_integ in (
        "rk45",
        "runge_kutta_45",
    ):
        results.append(
            GateResult(
                name="integrator_tolerance",
                status=(
                    GateStatus.FAILED
                    if horizon in (Horizon.G2, Horizon.G3)
                    else GateStatus.MISSING
                ),
                threshold=gates.max_integrator_rtol,
                reason="missing declared rk45_rtol for adaptive RK45 integrator",
            )
        )
    return results


def _evaluate_open_loop_replay(
    receipt: Mapping[str, Any],
    horizon: Horizon,
    gates: AcceptanceGates,
) -> list[GateResult]:
    """Evaluate open-loop forward rollout drift and integrator tolerance."""
    results: list[GateResult] = []

    # Drift threshold is tied to the horizon's whole-RMSE scale (e.g. 25 mm for G1)
    if gates.max_open_loop_drift_m is not None:
        thresh_drift = gates.max_open_loop_drift_m
    elif horizon == Horizon.G1:
        thresh_drift = gates.g1_whole_rmse_m
    elif horizon == Horizon.G2:
        thresh_drift = gates.g2_whole_rmse_m
    elif horizon == Horizon.G3:
        thresh_drift = gates.g3_whole_rmse_m
    else:
        thresh_drift = gates.g1_whole_rmse_m

    replay_data = receipt.get("open_loop_replay")
    if not isinstance(replay_data, Mapping):
        replay_data = receipt.get("forward_rollout")

    if not isinstance(replay_data, Mapping):
        if horizon in (Horizon.G2, Horizon.G3):
            results.append(
                GateResult(
                    name="open_loop_replay",
                    status=GateStatus.MISSING,
                    threshold=thresh_drift,
                    reason="missing open-loop replay artifact",
                )
            )
        return results

    # 1. Integrator consistency and tolerance checks
    results.extend(_evaluate_integrator_consistency(receipt, replay_data))
    results.extend(_evaluate_integrator_tolerance(receipt, replay_data, horizon, gates))

    # 2. Replay drift check against horizon whole-RMSE threshold
    drift_val = _extract_metric(
        replay_data, "drift_m", "max_drift_m", "whole_marker_rmse_m", "max_error_m"
    )

    if drift_val is None:
        results.append(
            GateResult(
                name="open_loop_replay",
                status=GateStatus.FAILED,
                threshold=thresh_drift,
                reason="missing drift metric in open-loop replay",
            )
        )
    elif drift_val > thresh_drift:
        results.append(
            GateResult(
                name="open_loop_replay",
                status=GateStatus.FAILED,
                threshold=thresh_drift,
                measured=drift_val,
                unit="m",
                reason=f"open-loop drift {drift_val * 1e3:.1f} mm exceeds {thresh_drift * 1e3:.1f} mm threshold",
            )
        )
    else:
        results.append(
            GateResult(
                name="open_loop_replay",
                status=GateStatus.PASSED,
                threshold=thresh_drift,
                measured=drift_val,
                unit="m",
            )
        )
    return results


def _evaluate_collocation_defect(
    receipt: Mapping[str, Any],
    horizon: Horizon,
    gates: AcceptanceGates,
) -> list[GateResult]:
    """Evaluate per-node dynamical consistency collocation defect."""
    results: list[GateResult] = []
    defect_data = receipt.get("collocation_defect")
    if not isinstance(defect_data, Mapping):
        defect_data = receipt.get("defects")

    if not isinstance(defect_data, Mapping):
        if horizon in (Horizon.G2, Horizon.G3):
            results.append(
                GateResult(
                    name="collocation_defect",
                    status=GateStatus.MISSING,
                    threshold=gates.max_collocation_defect_m,
                    reason="missing per-node collocation defect artifact",
                )
            )
        return results

    defect_val = _extract_metric(
        defect_data, "max_defect_m", "collocation_defect_m", "defect_max_m"
    )
    if defect_val is None:
        results.append(
            GateResult(
                name="collocation_defect",
                status=GateStatus.FAILED,
                threshold=gates.max_collocation_defect_m,
                reason="missing max defect measurement in collocation defect artifact",
            )
        )
    elif defect_val > gates.max_collocation_defect_m:
        results.append(
            GateResult(
                name="collocation_defect",
                status=GateStatus.FAILED,
                threshold=gates.max_collocation_defect_m,
                measured=defect_val,
                unit="m",
                reason=f"max collocation defect {defect_val * 1e3:.2f} mm > {gates.max_collocation_defect_m * 1e3:.2f} mm",
            )
        )
    else:
        results.append(
            GateResult(
                name="collocation_defect",
                status=GateStatus.PASSED,
                threshold=gates.max_collocation_defect_m,
                measured=defect_val,
                unit="m",
            )
        )
    return results


def _evaluate_stabilized_replay(
    receipt: Mapping[str, Any],
    horizon: Horizon,
    gates: AcceptanceGates,
) -> list[GateResult]:
    """Evaluate cross-engine stabilized replay under low-gain PD tracking."""
    results: list[GateResult] = []
    stab_data = receipt.get("stabilized_replay")
    if not isinstance(stab_data, Mapping):
        stab_data = receipt.get("stabilized_tracking")

    if not isinstance(stab_data, Mapping):
        if horizon in (Horizon.G2, Horizon.G3):
            results.append(
                GateResult(
                    name="stabilized_replay",
                    status=GateStatus.MISSING,
                    threshold=gates.max_stabilized_marker_rmse_m,
                    reason="missing stabilized replay artifact",
                )
            )
        return results

    rmse_val = _extract_metric(
        stab_data, "whole_marker_rmse_m", "marker_rms_m", "rmse_m"
    )
    if rmse_val is None:
        results.append(
            GateResult(
                name="stabilized_replay",
                status=GateStatus.FAILED,
                threshold=gates.max_stabilized_marker_rmse_m,
                reason="missing tracking RMSE in stabilized replay artifact",
            )
        )
    elif rmse_val > gates.max_stabilized_marker_rmse_m:
        results.append(
            GateResult(
                name="stabilized_replay",
                status=GateStatus.FAILED,
                threshold=gates.max_stabilized_marker_rmse_m,
                measured=rmse_val,
                unit="m",
                reason=f"stabilized tracking RMSE {rmse_val * 1e3:.2f} mm > {gates.max_stabilized_marker_rmse_m * 1e3:.2f} mm",
            )
        )
    else:
        results.append(
            GateResult(
                name="stabilized_replay",
                status=GateStatus.PASSED,
                threshold=gates.max_stabilized_marker_rmse_m,
                measured=rmse_val,
                unit="m",
            )
        )
    return results


def _evaluate_calibration_provenance(
    receipt: Mapping[str, Any],
    gates: AcceptanceGates,
    declared_capture: str | None = None,
) -> list[GateResult]:
    """Verify that capture and calibration/attachment provenance agree."""
    results: list[GateResult] = []
    capture = (
        declared_capture
        or receipt.get("capture")
        or receipt.get("receipt_path")
        or receipt.get("path")
        or receipt.get("source_receipt")
        or receipt.get("name")
        or ""
    )
    capture = str(capture).lower()
    attachments_source = str(receipt.get("attachments_source", "")).lower()

    if not capture or not attachments_source:
        return results

    is_iron_capture = "iron" in capture
    is_driver_capture = "driver" in capture
    is_iron_calib = "iron" in attachments_source
    is_driver_calib = "driver" in attachments_source

    if is_iron_capture and is_driver_calib and not is_iron_calib:
        results.append(
            GateResult(
                name="calibration_provenance",
                status=GateStatus.FAILED,
                threshold=1.0,
                measured=0.0,
                unit="match",
                reason=(
                    f"capture and calibration provenance disagree: iron capture reused driver calibration '{receipt.get('attachments_source')}'"
                ),
            )
        )
    elif is_driver_capture and is_iron_calib and not is_driver_calib:
        results.append(
            GateResult(
                name="calibration_provenance",
                status=GateStatus.FAILED,
                threshold=1.0,
                measured=0.0,
                unit="match",
                reason=(
                    f"capture and calibration provenance disagree: driver capture reused iron calibration '{receipt.get('attachments_source')}'"
                ),
            )
        )
    elif (is_iron_capture and is_iron_calib) or (is_driver_capture and is_driver_calib):
        results.append(
            GateResult(
                name="calibration_provenance",
                status=GateStatus.PASSED,
                threshold=1.0,
                measured=1.0,
                unit="match",
                reason="capture and calibration provenance match",
            )
        )
    return results


def _evaluate_friction_cone(
    receipt: Mapping[str, Any],
    gates: AcceptanceGates,
    contact_audit: Any,
) -> list[GateResult]:
    """Evaluate Coulomb friction cone ratio against physical bound."""
    results: list[GateResult] = []
    f_ratio = None
    if isinstance(contact_audit, Mapping):
        f_ratio = _extract_metric(contact_audit, "max_friction_ratio", "friction_ratio")
    if f_ratio is None:
        f_ratio = _extract_metric(receipt, "max_friction_ratio", "friction_ratio")

    if f_ratio is not None:
        thresh = gates.max_friction_coefficient
        if f_ratio <= thresh:
            results.append(
                GateResult(
                    name="friction_cone",
                    status=GateStatus.PASSED,
                    threshold=thresh,
                    measured=f_ratio,
                    unit="ratio",
                )
            )
        else:
            results.append(
                GateResult(
                    name="friction_cone",
                    status=GateStatus.FAILED,
                    threshold=thresh,
                    measured=f_ratio,
                    unit="ratio",
                    reason=f"max friction ratio {f_ratio:.2f} exceeds allowable coefficient {thresh:.2f}",
                )
            )
    return results


def _evaluate_torque_bounds(
    receipt: Mapping[str, Any],
    gates: AcceptanceGates,
) -> list[GateResult]:
    """Evaluate actuator torque limits and verify no torque bound overwrites."""
    results: list[GateResult] = []
    dyn = receipt.get("dynamics")
    torque_val = None
    overwrites = None
    if isinstance(dyn, Mapping):
        torque_val = _extract_metric(dyn, "max_actuator_torque_n_m", "max_torque_n_m")
        overwrites = dyn.get("torque_bound_overwrites")
    if torque_val is None:
        torque_val = _extract_metric(
            receipt, "max_actuator_torque_n_m", "max_torque_n_m"
        )
    if overwrites is None:
        overwrites = receipt.get("torque_bound_overwrites")

    if torque_val is not None or overwrites is not None:
        thresh = gates.max_actuator_torque_n_m
        t_val = torque_val if torque_val is not None else 0.0
        ow_val = int(overwrites) if overwrites is not None else 0
        if t_val <= thresh and ow_val == 0:
            results.append(
                GateResult(
                    name="torque_bounds",
                    status=GateStatus.PASSED,
                    threshold=thresh,
                    measured=t_val,
                    unit="N*m",
                )
            )
        else:
            reasons: list[str] = []
            if t_val > thresh:
                reasons.append(
                    f"max actuator torque {t_val:.1f} N*m > {thresh:.1f} N*m"
                )
            if ow_val > 0:
                reasons.append(f"{ow_val} torque bound overwrites detected")
            results.append(
                GateResult(
                    name="torque_bounds",
                    status=GateStatus.FAILED,
                    threshold=thresh,
                    measured=t_val,
                    unit="N*m",
                    reason="; ".join(reasons),
                )
            )
    return results


def _evaluate_root_force_history(
    receipt: Mapping[str, Any],
    gates: AcceptanceGates,
) -> list[GateResult]:
    """Verify presence of root force histories and that root residual forces vanish."""
    results: list[GateResult] = []
    dyn = receipt.get("dynamics")
    has_root = None
    root_res = None
    if isinstance(dyn, Mapping):
        has_root = dyn.get("has_root_histories")
        root_res = _extract_metric(dyn, "max_root_residual_n_m", "root_residual_n_m")
    if has_root is None:
        has_root = receipt.get("has_root_histories")
    if root_res is None:
        root_res = _extract_metric(
            receipt, "max_root_residual_n_m", "root_residual_n_m"
        )

    if has_root is not None or root_res is not None:
        thresh = gates.max_root_residual_n_m
        r_val = root_res if root_res is not None else 0.0
        if has_root is False:
            results.append(
                GateResult(
                    name="root_force_history",
                    status=GateStatus.FAILED,
                    threshold=thresh,
                    measured=r_val,
                    unit="N*m",
                    reason="root force history missing when dynamics declared",
                )
            )
        elif r_val > thresh:
            results.append(
                GateResult(
                    name="root_force_history",
                    status=GateStatus.FAILED,
                    threshold=thresh,
                    measured=r_val,
                    unit="N*m",
                    reason=f"max root residual {r_val:.2f} N*m exceeds tolerance {thresh:.2e} N*m",
                )
            )
        else:
            results.append(
                GateResult(
                    name="root_force_history",
                    status=GateStatus.PASSED,
                    threshold=thresh,
                    measured=r_val,
                    unit="N*m",
                )
            )
    return results


def _evaluate_coordinate_dimension(
    receipt: Mapping[str, Any],
    gates: AcceptanceGates,
) -> list[GateResult]:
    """Verify coordinate dimension matches expected model dimension."""
    results: list[GateResult] = []
    exp_nv = receipt.get("expected_nv")
    meas_nv = receipt.get("measured_nv")
    if exp_nv is not None and meas_nv is not None:
        if int(exp_nv) == int(meas_nv):
            results.append(
                GateResult(
                    name="coordinate_dimension",
                    status=GateStatus.PASSED,
                    threshold=float(exp_nv),
                    measured=float(meas_nv),
                    unit="dim",
                )
            )
        else:
            results.append(
                GateResult(
                    name="coordinate_dimension",
                    status=GateStatus.FAILED,
                    threshold=float(exp_nv),
                    measured=float(meas_nv),
                    unit="dim",
                    reason=f"coordinate dimension mismatch: expected {exp_nv} nv but measured {meas_nv} nv",
                )
            )
    return results


def _evaluate_club_coverage(
    receipt: Mapping[str, Any],
    gates: AcceptanceGates,
) -> list[GateResult]:
    """Verify club marker coverage fraction exceeds minimum required fraction."""
    results: list[GateResult] = []
    cov = receipt.get("coverage")
    club_cov = None
    if isinstance(cov, Mapping):
        club_cov = _extract_metric(
            cov, "club_marker_coverage_fraction", "club_coverage"
        )
    if club_cov is None:
        club_cov = _extract_metric(
            receipt, "club_marker_coverage_fraction", "club_coverage"
        )

    if club_cov is not None:
        thresh = gates.min_club_marker_coverage_fraction
        if club_cov >= thresh:
            results.append(
                GateResult(
                    name="club_coverage",
                    status=GateStatus.PASSED,
                    threshold=thresh,
                    measured=club_cov,
                    unit="fraction",
                )
            )
        else:
            results.append(
                GateResult(
                    name="club_coverage",
                    status=GateStatus.FAILED,
                    threshold=thresh,
                    measured=club_cov,
                    unit="fraction",
                    reason=f"club marker coverage fraction {club_cov:.2%} < threshold {thresh:.2%}",
                )
            )
    return results


def _evaluate_horizon_truncation(
    receipt: Mapping[str, Any],
    horizon: Horizon,
    gates: AcceptanceGates,
) -> list[GateResult]:
    """Verify trajectory duration is not truncated below horizon requirements."""
    results: list[GateResult] = []
    duration = _extract_metric(
        receipt, "duration_s", "time_span_s", "trajectory_duration_s"
    )
    if duration is not None:
        thresh = (
            gates.g1_min_duration_s
            if horizon == Horizon.G1
            else (
                gates.g2_min_duration_s
                if horizon == Horizon.G2
                else gates.g3_min_duration_s
            )
        )
        if duration >= thresh:
            results.append(
                GateResult(
                    name="horizon_truncation",
                    status=GateStatus.PASSED,
                    threshold=thresh,
                    measured=duration,
                    unit="s",
                )
            )
        else:
            results.append(
                GateResult(
                    name="horizon_truncation",
                    status=GateStatus.FAILED,
                    threshold=thresh,
                    measured=duration,
                    unit="s",
                    reason=f"horizon duration {duration:.2f} s is truncated below required minimum {thresh:.2f} s",
                )
            )
    return results


def _evaluate_synthetic_engine(
    receipt: Mapping[str, Any],
    gates: AcceptanceGates,
) -> list[GateResult]:
    """Verify synthetic engine is not claiming native qualification falsely."""
    results: list[GateResult] = []
    is_nat_qual = receipt.get("is_native_qualified")
    engine = str(receipt.get("engine", "")).lower()

    if is_nat_qual is not None or "synthetic" in engine:
        if is_nat_qual is False or "synthetic" in engine:
            results.append(
                GateResult(
                    name="synthetic_engine",
                    status=GateStatus.FAILED,
                    threshold=1.0,
                    measured=0.0,
                    unit="match",
                    reason=f"synthetic analytical engine '{engine or 'unknown'}' not natively qualified",
                )
            )
        else:
            results.append(
                GateResult(
                    name="synthetic_engine",
                    status=GateStatus.PASSED,
                    threshold=1.0,
                    measured=1.0,
                    unit="match",
                )
            )
    return results


def _evaluate_full_body_profile(receipt: Mapping[str, Any]) -> list[GateResult]:
    """Reject reduced-model profiles that claim full-body acceptance (MS-61)."""
    profile = str(receipt.get("model_profile", "")).strip().lower()
    if not profile:
        nested = receipt.get("reduced_model_body_only")
        if isinstance(nested, Mapping):
            profile = str(nested.get("profile", "")).strip().lower()
    if profile in {
        "reduced_body_excluding_head",
        "body_only",
        "body-excluding-head",
        "body_excluding_head",
        "exclude_head",
        "head_excluded",
    }:
        return [
            GateResult(
                name="full_body_profile",
                status=GateStatus.FAILED,
                threshold=1.0,
                measured=0.0,
                unit="match",
                reason=(
                    f"model_profile={profile!r} cannot satisfy full-body G1; "
                    "body-excluding-head is diagnostic only (MS-61 / #10348)"
                ),
            )
        ]
    return []


def _evaluate_dual_terminal_disclosure(
    receipt: Mapping[str, Any],
) -> list[GateResult]:
    """Require both terminal metrics when the receipt opts into dual disclosure."""
    required = bool(receipt.get("require_dual_terminal_metrics"))
    full_g1 = receipt.get("full_marker_g1")
    if isinstance(full_g1, Mapping) and full_g1.get("require_dual_terminal_metrics"):
        required = True
    if not required:
        return []

    full_t = _extract_metric(
        receipt,
        "terminal_full_marker_rmse_m",
        "terminal_marker_rmse_m",
        "terminal_rms_m",
    )
    body_t = _extract_metric(receipt, "terminal_body_excluding_head_rmse_m")
    head_t = _extract_metric(receipt, "terminal_head_cluster_rmse_m")
    if full_t is None or body_t is None or head_t is None:
        return [
            GateResult(
                name="dual_terminal_disclosure",
                status=GateStatus.MISSING,
                threshold=1.0,
                reason=(
                    "require_dual_terminal_metrics needs terminal_full_marker_rmse_m, "
                    "terminal_body_excluding_head_rmse_m, and "
                    "terminal_head_cluster_rmse_m"
                ),
            )
        ]
    return [
        GateResult(
            name="dual_terminal_disclosure",
            status=GateStatus.DISCLOSED,
            threshold=1.0,
            measured=None,
            unit="match",
            reason="terminal full/body/head metrics disclosed",
        )
    ]


@precondition(
    lambda receipt, horizon=Horizon.G1, gates=None, capture=None: isinstance(
        horizon, Horizon
    ),
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
    capture: str | None = None,
) -> AcceptanceVerdict:
    """Pure evaluation function that returns physical & kinematic acceptance verdict."""
    if gates is None:
        gates = AcceptanceGates()

    contact_audit = receipt.get("contact_audit")
    gate_results: list[GateResult] = []
    from src.shared.python.motion_matching.evidence_integrity import (
        evaluate_evidence_integrity,
    )

    gate_results.extend(evaluate_evidence_integrity(receipt))
    gate_results.extend(_evaluate_marker_rmse(receipt, horizon, gates))
    # MS-61 (#10348): never hide head-cluster terminal; body-only cannot pass full-body.
    from src.shared.python.motion_matching.full_marker_terminal import (
        evaluate_full_marker_terminal_disclosure,
    )

    gate_results.extend(evaluate_full_marker_terminal_disclosure(receipt))
    gate_results.extend(_evaluate_pelvis_yaw(receipt, horizon, gates))
    gate_results.extend(_evaluate_normal_contact_force(receipt, gates, contact_audit))
    gate_results.extend(_evaluate_ground_and_closure(receipt, gates, contact_audit))
    gate_results.extend(_evaluate_weight_fraction(receipt, gates))
    gate_results.extend(_evaluate_calibration_provenance(receipt, gates, capture))
    gate_results.extend(_evaluate_open_loop_replay(receipt, horizon, gates))
    gate_results.extend(_evaluate_collocation_defect(receipt, horizon, gates))
    gate_results.extend(_evaluate_stabilized_replay(receipt, horizon, gates))
    gate_results.extend(_evaluate_friction_cone(receipt, gates, contact_audit))
    gate_results.extend(_evaluate_torque_bounds(receipt, gates))
    gate_results.extend(_evaluate_root_force_history(receipt, gates))
    gate_results.extend(_evaluate_coordinate_dimension(receipt, gates))
    gate_results.extend(_evaluate_club_coverage(receipt, gates))
    gate_results.extend(_evaluate_horizon_truncation(receipt, horizon, gates))
    gate_results.extend(_evaluate_synthetic_engine(receipt, gates))
    gate_results.extend(_evaluate_full_body_profile(receipt))
    gate_results.extend(_evaluate_dual_terminal_disclosure(receipt))

    # Overall verdict
    is_accepted = is_verdict_accepted(gate_results)
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


def evaluate_baseline_package_acceptance(
    package: Any,
    *,
    horizon: Horizon | None = None,
    gates: AcceptanceGates | None = None,
) -> AcceptanceVerdict:
    """Evaluate acceptance for a baseline package or manifest under the Matched Swing Program (TB-02 #10587)."""
    if hasattr(package, "to_dict"):
        pkg_dict = package.to_dict()
    elif isinstance(package, Mapping):
        pkg_dict = dict(package)
    else:
        raise TypeError("package must be BaselinePackage or Mapping")

    ident = pkg_dict.get("identity")
    ident_dict = ident if isinstance(ident, Mapping) else {}
    raw_h = ident_dict.get("horizon", "G1") if ident_dict else "G1"
    h = horizon if horizon is not None else Horizon(raw_h)
    return evaluate(pkg_dict, horizon=h, gates=gates)


def reject_visual_override_of_physical_failure(package: Any) -> bool:
    """CO-08 path-anchor: visual attractiveness cannot override physical failure.

    Delegates to club-only matrix qualification so tour acceptance and the club
    matrix share one fail-closed rule.
    """
    from src.shared.python.motion_matching.club_only.matrix_qualification import (
        ExportedCandidatePackage,
        physical_overrides_visual,
    )

    if not isinstance(package, ExportedCandidatePackage):
        raise TypeError("package must be ExportedCandidatePackage")
    return physical_overrides_visual(package)


def qualify_tour_baseline(
    package: Any,
    profile_version: str | None = None,
) -> Any:
    """Independently qualify tour baseline package against canonical criteria (TB-09 #10594).

    Recomputes all metrics and verifies artifact integrity, dynamic rollout,
    physical/geometric constraints, and endpoint criteria.
    """
    from src.shared.python.tour_baselines.qualification import (
        QUALIFICATION_PROFILE_VERSION,
        IndependentBaselineQualifier,
    )

    ver = (
        profile_version
        if profile_version is not None
        else QUALIFICATION_PROFILE_VERSION
    )
    qualifier = IndependentBaselineQualifier()
    return qualifier.qualify(package, profile_version=ver)
