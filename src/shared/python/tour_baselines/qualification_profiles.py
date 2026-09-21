"""Versioned qualification profiles and scientific gate evaluation for golf baselines (TB-02 #10587).

Authoritative full-body G1/G2/G3 contracts remain binding; reduced-model educational
baselines require explicitly separate, frozen numeric qualification profiles based on
their attainable fixed-geometry residuals without relaxing full-body thresholds.

Defines "best" as the best feasible candidate within a declared model class, objective,
observation set, horizon, and computation budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np

from src.shared.python.tour_baselines.baseline_package import (
    BaselinePackage,
    DynamicFeasibilityStatus,
    KinematicAccuracyStatus,
    ProductPromotionStatus,
    ScientificQualificationStatus,
    SolverConvergenceStatus,
    StatusBundle,
)
from src.shared.python.tour_baselines.models import ModelTopology

logger = logging.getLogger(__name__)

QUALIFICATION_PROFILE_VERSION = "tour-qualification-profile/1.0.0"


@dataclass(frozen=True)
class QualificationGateResult:
    """Outcome of evaluating an individual numeric or physical gate."""

    name: str
    threshold: float
    measured: float | None
    passed: bool
    unit: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "measured": self.measured,
            "passed": self.passed,
            "unit": self.unit,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class QualificationVerdict:
    """Consolidated qualification decision for a baseline package."""

    profile_name: str
    profile_version: str
    passed: bool
    statuses: StatusBundle
    gates: tuple[QualificationGateResult, ...]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "passed": self.passed,
            "statuses": self.statuses.to_dict(),
            "gates": [g.to_dict() for g in self.gates],
            "notes": self.notes,
        }


@dataclass(frozen=True)
class AuthoritativeFullBodyProfile:
    """Authoritative full-body multi-body qualification profile (G1/G2/G3)."""

    name: str = "AuthoritativeFullBodyProfile"
    version: str = QUALIFICATION_PROFILE_VERSION
    topology: ModelTopology = ModelTopology.FULL_BODY_MULTIBODY
    rationale: str = (
        "Authoritative clinical/tour full-body acceptance gates under G1/G2/G3 horizons "
        "(MS-01 #10322, AcceptanceGates). Requires full-body marker coverage."
    )

    # G1 thresholds (metres, radians)
    g1_whole_rmse_m: float = 0.025
    g1_early_rmse_m: float = 0.012
    g1_terminal_rmse_m: float = 0.035
    g1_club_rmse_m: float = 0.060
    g1_pelvis_yaw_rmse_rad: float = 0.05236

    # G2 thresholds
    g2_whole_rmse_m: float = 0.040
    g2_early_rmse_m: float = 0.015
    g2_terminal_rmse_m: float = 0.050
    g2_club_rmse_m: float = 0.075
    g2_pelvis_yaw_rmse_rad: float = 0.08727

    # G3 thresholds
    g3_whole_driver_rmse_m: float = 0.060
    g3_whole_iron_rmse_m: float = 0.095
    g3_early_rmse_m: float = 0.020
    g3_terminal_rmse_m: float = 0.080
    g3_club_rmse_m: float = 0.100
    g3_pelvis_yaw_rmse_rad: float = 0.10472

    # Physical limits
    max_normal_force_bw_mult: float = 3.0
    max_penetration_m: float = 0.010
    max_closure_residual_m: float = 0.005


@dataclass(frozen=True)
class PlanarDrivenPendulumProfile:
    """Reduced educational baseline profile for planar 2-DOF driven pendulum."""

    name: str = "PlanarDrivenPendulumProfile"
    version: str = QUALIFICATION_PROFILE_VERSION
    topology: ModelTopology = ModelTopology.PLANAR_DRIVEN_PENDULUM
    rationale: str = (
        "Planar 2-DOF driven pendulum educational baseline. Attainable fixed-geometry "
        "residuals are bounded by planarity; out-of-plane deformation is zero by topology. "
        "Observable landmarks restricted to clubhead and grip. Does NOT alter full-body thresholds."
    )
    max_club_rmse_m: float = 0.150  # 150 mm attainable clubhead tracking
    max_out_of_plane_residual_m: float = 0.050  # 50 mm planarity consistency
    observable_landmarks: tuple[str, ...] = ("Grip", "Marker_2", "Marker_3")


@dataclass(frozen=True)
class UpperBodyGolferProfile:
    """Reduced baseline profile for constrained 5-DOF upper-body golfer."""

    name: str = "UpperBodyGolferProfile"
    version: str = QUALIFICATION_PROFILE_VERSION
    topology: ModelTopology = ModelTopology.CONSTRAINED_UPPER_BODY
    rationale: str = (
        "Constrained upper-body golfer baseline (5 independent DOFs, closed kinematic loop). "
        "Evaluates attainable torso/arm/club marker geometry and weld closure without "
        "lower extremity tracking requirements."
    )
    max_whole_marker_rmse_m: float = 0.055  # 55 mm attainable upper body tracking
    max_closure_residual_m: float = 0.005  # 5 mm weld closure
    observable_landmarks: tuple[str, ...] = (
        "Clavicle",
        "ShoulderLeft",
        "ShoulderRight",
        "Marker_2",
        "Marker_3",
    )


@dataclass(frozen=True)
class TriplePendulumProfile:
    """Reduced kinematic reconstruction baseline profile for 3-DOF triple pendulum."""

    name: str = "TriplePendulumProfile"
    version: str = QUALIFICATION_PROFILE_VERSION
    topology: ModelTopology = ModelTopology.KINEMATIC_RECONSTRUCTION
    rationale: str = (
        "Triple pendulum kinematic reconstruction baseline for torso, lead arm, and club shaft. "
        "Evaluates kinematic swing timing and club path against planar projection."
    )
    max_club_rmse_m: float = 0.120  # 120 mm attainable club tracking
    observable_landmarks: tuple[str, ...] = (
        "Spine",
        "Arm",
        "Club",
        "Marker_2",
        "Marker_3",
    )


def get_qualification_profile(topology: ModelTopology) -> Any:
    """Return the frozen qualification profile corresponding to a model topology."""
    if topology == ModelTopology.FULL_BODY_MULTIBODY:
        return AuthoritativeFullBodyProfile()
    if topology == ModelTopology.PLANAR_DRIVEN_PENDULUM:
        return PlanarDrivenPendulumProfile()
    if topology == ModelTopology.CONSTRAINED_UPPER_BODY:
        return UpperBodyGolferProfile()
    if topology == ModelTopology.KINEMATIC_RECONSTRUCTION:
        return TriplePendulumProfile()
    return AuthoritativeFullBodyProfile()


def _extract_club_rmse(metrics: Any) -> float:
    """Extract average club marker RMSE or fallback to whole marker RMSE."""
    club_errors = [
        m_sum.rmse_m
        for name, m_sum in metrics.per_marker.items()
        if "Marker_" in name or "Club" in name
    ]
    return (
        float(np.mean(club_errors))
        if club_errors
        else float(metrics.whole_marker_rmse_m)
    )


def _eval_closure_gate(
    package: BaselinePackage,
    max_closure_residual_m: float,
) -> QualificationGateResult | None:
    """Evaluate closure residual gate if reported in package."""
    closure_meas = package.reports.get("closure_residual_m")
    if closure_meas is None:
        return None
    cl_pass = closure_meas <= max_closure_residual_m
    reason = (
        ""
        if cl_pass
        else f"closure {closure_meas * 1e3:.1f} mm > {max_closure_residual_m * 1e3:.1f} mm"
    )
    return QualificationGateResult(
        name="closure_residual_m",
        threshold=max_closure_residual_m,
        measured=float(closure_meas),
        passed=cl_pass,
        unit="m",
        reason=reason,
    )


def _eval_full_body_gates(
    package: BaselinePackage,
    profile: AuthoritativeFullBodyProfile,
) -> list[QualificationGateResult]:
    """Evaluate authoritative full-body gates."""
    gates: list[QualificationGateResult] = []
    ident = package.identity
    metrics = package.metrics
    horizon = ident.horizon

    if horizon == "G1":
        whole_th = profile.g1_whole_rmse_m
        club_th = profile.g1_club_rmse_m
    elif horizon == "G2":
        whole_th = profile.g2_whole_rmse_m
        club_th = profile.g2_club_rmse_m
    else:
        whole_th = (
            profile.g3_whole_iron_rmse_m
            if ident.capture == "iron"
            else profile.g3_whole_driver_rmse_m
        )
        club_th = profile.g3_club_rmse_m

    # Whole marker RMSE
    w_meas = metrics.whole_marker_rmse_m
    w_pass = w_meas <= whole_th
    gates.append(
        QualificationGateResult(
            name="whole_marker_rmse_m",
            threshold=whole_th,
            measured=w_meas,
            passed=w_pass,
            unit="m",
            reason=""
            if w_pass
            else f"whole marker RMSE {w_meas * 1e3:.1f} mm > {whole_th * 1e3:.1f} mm",
        )
    )

    # Club marker RMSE
    c_meas = _extract_club_rmse(metrics)
    c_pass = c_meas <= club_th
    gates.append(
        QualificationGateResult(
            name="club_marker_rmse_m",
            threshold=club_th,
            measured=c_meas,
            passed=c_pass,
            unit="m",
            reason=""
            if c_pass
            else f"club marker RMSE {c_meas * 1e3:.1f} mm > {club_th * 1e3:.1f} mm",
        )
    )

    # Closure residual if reported
    cl_gate = _eval_closure_gate(package, profile.max_closure_residual_m)
    if cl_gate is not None:
        gates.append(cl_gate)

    return gates


def _eval_planar_pendulum_gates(
    package: BaselinePackage,
    profile: PlanarDrivenPendulumProfile,
) -> list[QualificationGateResult]:
    """Evaluate planar driven pendulum educational gates."""
    gates: list[QualificationGateResult] = []
    metrics = package.metrics

    # Club RMSE
    c_meas = _extract_club_rmse(metrics)
    c_pass = c_meas <= profile.max_club_rmse_m
    gates.append(
        QualificationGateResult(
            name="club_marker_rmse_m",
            threshold=profile.max_club_rmse_m,
            measured=c_meas,
            passed=c_pass,
            unit="m",
            reason=""
            if c_pass
            else f"planar club RMSE {c_meas * 1e3:.1f} mm > {profile.max_club_rmse_m * 1e3:.1f} mm",
        )
    )

    # Out of plane residual
    out_meas = metrics.out_of_plane_residual_m
    if out_meas is not None:
        out_pass = out_meas <= profile.max_out_of_plane_residual_m
        gates.append(
            QualificationGateResult(
                name="out_of_plane_residual_m",
                threshold=profile.max_out_of_plane_residual_m,
                measured=out_meas,
                passed=out_pass,
                unit="m",
                reason=""
                if out_pass
                else f"out-of-plane {out_meas * 1e3:.1f} mm > {profile.max_out_of_plane_residual_m * 1e3:.1f} mm",
            )
        )

    return gates


def _eval_upper_body_gates(
    package: BaselinePackage,
    profile: UpperBodyGolferProfile,
) -> list[QualificationGateResult]:
    """Evaluate constrained upper body golfer gates."""
    gates: list[QualificationGateResult] = []
    metrics = package.metrics

    # Whole marker RMSE for upper body
    w_meas = metrics.whole_marker_rmse_m
    w_pass = w_meas <= profile.max_whole_marker_rmse_m
    gates.append(
        QualificationGateResult(
            name="upper_body_marker_rmse_m",
            threshold=profile.max_whole_marker_rmse_m,
            measured=w_meas,
            passed=w_pass,
            unit="m",
            reason=""
            if w_pass
            else f"upper body RMSE {w_meas * 1e3:.1f} mm > {profile.max_whole_marker_rmse_m * 1e3:.1f} mm",
        )
    )

    # Closure residual
    cl_gate = _eval_closure_gate(package, profile.max_closure_residual_m)
    if cl_gate is not None:
        gates.append(cl_gate)

    return gates


def _eval_triple_pendulum_gates(
    package: BaselinePackage,
    profile: TriplePendulumProfile,
) -> list[QualificationGateResult]:
    """Evaluate triple pendulum kinematic reconstruction gates."""
    gates: list[QualificationGateResult] = []
    metrics = package.metrics
    c_meas = _extract_club_rmse(metrics)
    c_pass = c_meas <= profile.max_club_rmse_m
    gates.append(
        QualificationGateResult(
            name="club_marker_rmse_m",
            threshold=profile.max_club_rmse_m,
            measured=c_meas,
            passed=c_pass,
            unit="m",
            reason=""
            if c_pass
            else f"triple pendulum club RMSE {c_meas * 1e3:.1f} mm > {profile.max_club_rmse_m * 1e3:.1f} mm",
        )
    )
    return gates


def evaluate_baseline_qualification(
    package: BaselinePackage,
    profile: Any | None = None,
) -> QualificationVerdict:
    """Evaluate scientific qualification of a baseline package against a profile.

    Fail-closed:
    - Missing native replay -> scientific_qualification CANNOT be QUALIFIED (UNVERIFIED).
    - Failed numeric gates -> kinematic_accuracy = EXCEEDS_THRESHOLD and scientific_qualification = DISQUALIFIED.
    - All passed + native replay -> scientific_qualification = QUALIFIED.
    """
    if profile is None:
        profile = get_qualification_profile(package.identity.topology)

    p_name = getattr(profile, "name", "UnknownProfile")
    p_ver = getattr(profile, "version", QUALIFICATION_PROFILE_VERSION)

    if isinstance(profile, AuthoritativeFullBodyProfile):
        gate_results = _eval_full_body_gates(package, profile)
    elif isinstance(profile, PlanarDrivenPendulumProfile):
        gate_results = _eval_planar_pendulum_gates(package, profile)
    elif isinstance(profile, UpperBodyGolferProfile):
        gate_results = _eval_upper_body_gates(package, profile)
    elif isinstance(profile, TriplePendulumProfile):
        gate_results = _eval_triple_pendulum_gates(package, profile)
    else:
        gate_results = []

    all_gates_passed = len(gate_results) > 0 and all(g.passed for g in gate_results)
    st = package.statuses
    has_replay = st.has_native_replay

    # Determine updated statuses
    if all_gates_passed:
        kin_status = KinematicAccuracyStatus.WITHIN_TOLERANCE
        if has_replay:
            sci_status = ScientificQualificationStatus.QUALIFIED
        else:
            sci_status = ScientificQualificationStatus.UNVERIFIED
    else:
        kin_status = KinematicAccuracyStatus.EXCEEDS_THRESHOLD
        sci_status = ScientificQualificationStatus.DISQUALIFIED

    updated_statuses = StatusBundle(
        solver_convergence=st.solver_convergence,
        kinematic_accuracy=kin_status,
        dynamic_feasibility=st.dynamic_feasibility,
        scientific_qualification=sci_status,
        product_promotion=st.product_promotion,
        has_native_replay=has_replay,
    )

    verdict_passed = (
        all_gates_passed
        and has_replay
        and sci_status == ScientificQualificationStatus.QUALIFIED
    )

    note = (
        f"Qualified under {p_name} ({p_ver})"
        if verdict_passed
        else (
            "Missing native replay evidence"
            if not has_replay
            else f"Numeric gate violation under {p_name}"
        )
    )

    return QualificationVerdict(
        profile_name=p_name,
        profile_version=p_ver,
        passed=verdict_passed,
        statuses=updated_statuses,
        gates=tuple(gate_results),
        notes=note,
    )
