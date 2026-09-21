"""Club-only acceptance evaluation with separated status lanes (CO-02 #10606)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.acceptance import AcceptanceGates, Horizon
from src.shared.python.motion_matching.club_only.ambiguity import (
    AmbiguityVerdict,
    CandidateScore,
    assess_ambiguity,
)
from src.shared.python.motion_matching.club_only.observation import (
    orientation_residual_so3,
)
from src.shared.python.motion_matching.club_only.profiles import ClubOnlyProfile
from src.shared.python.tour_baselines.qualification_profiles import (
    QualificationGateResult,
)

# Re-export Horizon so club-only consumers keep G3 vocabulary without forking gates.
__all__ = [
    "ClubOnlyResidualReport",
    "ClubOnlyStatuses",
    "ClubOnlyAcceptanceVerdict",
    "evaluate_club_only_acceptance",
    "normalized_position_error",
    "normalized_orientation_error",
    "Horizon",
]


@dataclass(frozen=True)
class ClubOnlyResidualReport:
    """Measured club residuals and unweighted physical diagnostics."""

    grip_position_rmse_m: float
    face_position_rmse_m: float
    grip_orientation_rmse_rad: float | None
    face_orientation_rmse_rad: float | None
    native_coverage_fraction: float
    speed_error_m_s: float | None
    phase_error_s: float | None
    unweighted_physical: Mapping[str, float]

    def __post_init__(self) -> None:
        for name, value in (
            ("grip_position_rmse_m", self.grip_position_rmse_m),
            ("face_position_rmse_m", self.face_position_rmse_m),
            ("native_coverage_fraction", self.native_coverage_fraction),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if not 0.0 <= self.native_coverage_fraction <= 1.0:
            raise ValueError("coverage fraction must be in [0, 1]")
        for name, optional in (
            ("grip_orientation_rmse_rad", self.grip_orientation_rmse_rad),
            ("face_orientation_rmse_rad", self.face_orientation_rmse_rad),
            ("speed_error_m_s", self.speed_error_m_s),
            ("phase_error_s", self.phase_error_s),
        ):
            if optional is not None and (not math.isfinite(optional) or optional < 0.0):
                raise ValueError(f"{name} must be finite and >= 0 when set")
        for key, value in self.unweighted_physical.items():
            if not math.isfinite(value):
                raise ValueError(f"unweighted_physical[{key!r}] must be finite")


@dataclass(frozen=True)
class ClubOnlyStatuses:
    """Separated kinematic / torque / scientific / product lanes."""

    kinematic_preview: str
    torque_replay: str
    scientific: str
    product: str

    def as_dict(self) -> dict[str, str]:
        return {
            "kinematic_preview": self.kinematic_preview,
            "torque_replay": self.torque_replay,
            "scientific": self.scientific,
            "product": self.product,
        }


@dataclass(frozen=True)
class ClubOnlyAcceptanceVerdict:
    """Measured-vs-overall acceptance under a frozen club-only profile."""

    measured_accepted: bool
    overall_accepted: bool
    gates: tuple[QualificationGateResult, ...]
    statuses: ClubOnlyStatuses
    ambiguity: AmbiguityVerdict
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "measured_accepted": self.measured_accepted,
            "overall_accepted": self.overall_accepted,
            "gates": [g.to_dict() for g in self.gates],
            "statuses": self.statuses.as_dict(),
            "ambiguity": self.ambiguity.as_dict(),
            "limitations": list(self.limitations),
        }


def normalized_position_error(
    predicted_xyz_m: NDArray[np.floating],
    measured_xyz_m: NDArray[np.floating],
    *,
    sigma_m: float,
) -> float:
    """RMSE of per-frame Euclidean residuals scaled by position uncertainty."""
    if not math.isfinite(sigma_m) or sigma_m <= 0.0:
        raise ValueError("sigma_m must be finite and > 0")
    pred = np.asarray(predicted_xyz_m, dtype=np.float64)
    meas = np.asarray(measured_xyz_m, dtype=np.float64)
    if pred.shape != meas.shape or pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError("position arrays must share shape (N, 3)")
    if pred.shape[0] == 0:
        raise ValueError("position arrays must be non-empty")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(meas)):
        raise ValueError("position arrays must be finite")
    norms = np.linalg.norm(pred - meas, axis=1) / sigma_m
    return float(np.sqrt(np.mean(np.square(norms))))


def normalized_orientation_error(
    predicted_quat_wxyz: NDArray[np.floating],
    measured_quat_wxyz: NDArray[np.floating],
    *,
    sigma_rad: float,
) -> float:
    """RMSE of SO(3) geodesic residuals scaled by orientation uncertainty."""
    if not math.isfinite(sigma_rad) or sigma_rad <= 0.0:
        raise ValueError("sigma_rad must be finite and > 0")
    pred = np.asarray(predicted_quat_wxyz, dtype=np.float64)
    meas = np.asarray(measured_quat_wxyz, dtype=np.float64)
    if pred.shape != meas.shape or pred.ndim != 2 or pred.shape[1] != 4:
        raise ValueError("quaternion arrays must share shape (N, 4)")
    if pred.shape[0] == 0:
        raise ValueError("quaternion arrays must be non-empty")
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(meas)):
        raise ValueError("quaternion arrays must be finite")
    residuals = orientation_residual_so3(pred, meas) / sigma_rad
    return float(np.sqrt(np.mean(np.square(residuals))))


def _gate(
    name: str,
    *,
    threshold: float,
    measured: float | None,
    passed: bool,
    unit: str,
    reason: str = "",
) -> QualificationGateResult:
    return QualificationGateResult(
        name=name,
        threshold=threshold,
        measured=measured,
        passed=passed,
        unit=unit,
        reason=reason,
    )


def _measured_gates(
    residual: ClubOnlyResidualReport,
    profile: ClubOnlyProfile,
) -> list[QualificationGateResult]:
    obs = profile.observation
    measured_club = max(residual.grip_position_rmse_m, residual.face_position_rmse_m)
    club_threshold = max(obs.max_grip_position_rmse_m, obs.max_face_position_rmse_m)
    gates = [
        _gate(
            "measured_club_residual",
            threshold=club_threshold,
            measured=measured_club,
            passed=measured_club <= club_threshold,
            unit="m",
            reason="max grip/face position RMSE vs profile",
        ),
        _gate(
            "grip_position_rmse_m",
            threshold=obs.max_grip_position_rmse_m,
            measured=residual.grip_position_rmse_m,
            passed=residual.grip_position_rmse_m <= obs.max_grip_position_rmse_m,
            unit="m",
        ),
        _gate(
            "face_position_rmse_m",
            threshold=obs.max_face_position_rmse_m,
            measured=residual.face_position_rmse_m,
            passed=residual.face_position_rmse_m <= obs.max_face_position_rmse_m,
            unit="m",
        ),
        _gate(
            "native_coverage_fraction",
            threshold=obs.min_native_coverage_fraction,
            measured=residual.native_coverage_fraction,
            passed=residual.native_coverage_fraction
            >= obs.min_native_coverage_fraction,
            unit="fraction",
        ),
    ]
    if obs.max_grip_orientation_rmse_rad is not None:
        measured = residual.grip_orientation_rmse_rad
        gates.append(
            _gate(
                "grip_orientation_rmse_rad",
                threshold=obs.max_grip_orientation_rmse_rad,
                measured=measured,
                passed=(
                    measured is not None
                    and measured <= obs.max_grip_orientation_rmse_rad
                ),
                unit="rad",
                reason="supported orientation gate",
            )
        )
    if obs.max_face_orientation_rmse_rad is not None:
        measured = residual.face_orientation_rmse_rad
        gates.append(
            _gate(
                "face_orientation_rmse_rad",
                threshold=obs.max_face_orientation_rmse_rad,
                measured=measured,
                passed=(
                    measured is not None
                    and measured <= obs.max_face_orientation_rmse_rad
                ),
                unit="rad",
                reason="supported orientation gate",
            )
        )
    if obs.max_speed_error_m_s is not None and residual.speed_error_m_s is not None:
        gates.append(
            _gate(
                "speed_error_m_s",
                threshold=obs.max_speed_error_m_s,
                measured=residual.speed_error_m_s,
                passed=residual.speed_error_m_s <= obs.max_speed_error_m_s,
                unit="m/s",
            )
        )
    if obs.max_phase_error_s is not None and residual.phase_error_s is not None:
        gates.append(
            _gate(
                "phase_error_s",
                threshold=obs.max_phase_error_s,
                measured=residual.phase_error_s,
                passed=residual.phase_error_s <= obs.max_phase_error_s,
                unit="s",
            )
        )
    return gates


def _physical_gates(
    residual: ClubOnlyResidualReport,
    candidates: Sequence[CandidateScore],
    profile: ClubOnlyProfile,
    *,
    body_markers_present: bool,
    force_labels_synthetic: bool,
) -> list[QualificationGateResult]:
    gates: list[QualificationGateResult] = []
    closure = residual.unweighted_physical.get("closure_residual_m")
    if closure is None:
        closures = [
            c.closure_residual_m for c in candidates if c.closure_residual_m is not None
        ]
        closure = max(closures) if closures else None
    if closure is not None:
        gates.append(
            _gate(
                "closure_residual_m",
                threshold=profile.physical.max_closure_residual_m,
                measured=closure,
                passed=closure <= profile.physical.max_closure_residual_m,
                unit="m",
            )
        )
    if profile.physical.requires_contact_feasibility:
        contact_ok = all(c.contact_feasible for c in candidates)
        gates.append(
            _gate(
                "contact_feasibility",
                threshold=1.0,
                measured=1.0 if contact_ok else 0.0,
                passed=contact_ok,
                unit="bool",
                reason="stance/contact must remain feasible",
            )
        )
    if profile.physical.requires_body_markers:
        gates.append(
            _gate(
                "body_marker_coverage",
                threshold=1.0,
                measured=1.0 if body_markers_present else 0.0,
                passed=body_markers_present,
                unit="bool",
                reason="missing body data never passes a body-marker gate",
            )
        )
    force_claim = any(c.claims_force_measurement for c in candidates)
    if force_claim:
        reason = (
            "synthetic labels cannot claim force measurement"
            if force_labels_synthetic
            else "force-measurement claim rejected without native force evidence"
        )
        gates.append(
            _gate(
                "force_measurement_claim",
                threshold=0.0,
                measured=1.0,
                passed=False,
                unit="bool",
                reason=reason,
            )
        )
    return gates


def _scientific_status(
    *,
    measured_ok: bool,
    physical_ok: bool,
    body_ok: bool,
    torque_replay_validated: bool,
) -> str:
    if not measured_ok or not physical_ok or not body_ok:
        return "disqualified"
    if torque_replay_validated:
        return "qualified"
    return "unverified"


def evaluate_club_only_acceptance(
    residual: ClubOnlyResidualReport,
    candidates: Sequence[CandidateScore],
    profile: ClubOnlyProfile,
    *,
    body_markers_present: bool,
    force_labels_synthetic: bool,
    torque_replay_validated: bool = False,
    kinematic_preview_ok: bool = False,
) -> ClubOnlyAcceptanceVerdict:
    """Evaluate club-only acceptance without collapsing status lanes."""
    if not candidates:
        raise ValueError("candidates must be non-empty")

    # Preserve authoritative G3 constants on the shared AcceptanceGates object.
    _ = AcceptanceGates().g3_club_rmse_m
    _ = Horizon.G3

    measured_gates = _measured_gates(residual, profile)
    physical_gates = _physical_gates(
        residual,
        candidates,
        profile,
        body_markers_present=body_markers_present,
        force_labels_synthetic=force_labels_synthetic,
    )
    gates = tuple(measured_gates + physical_gates)
    measured_ok = all(g.passed for g in measured_gates)
    # Prior score is intentionally unused for measured acceptance.
    physical_ok = all(g.passed for g in physical_gates)
    body_ok = body_markers_present if profile.physical.requires_body_markers else True
    ambiguity = assess_ambiguity(candidates, profile)
    overall = measured_ok and physical_ok and body_ok

    kinematic = "passed" if kinematic_preview_ok else "unevaluated"
    torque = "passed" if torque_replay_validated else "unevaluated"
    scientific = _scientific_status(
        measured_ok=measured_ok,
        physical_ok=physical_ok,
        body_ok=body_ok,
        torque_replay_validated=torque_replay_validated,
    )
    # Club-only software acceptance never promotes product readiness.
    product = "exploratory"

    return ClubOnlyAcceptanceVerdict(
        measured_accepted=measured_ok,
        overall_accepted=overall,
        gates=gates,
        statuses=ClubOnlyStatuses(
            kinematic_preview=kinematic,
            torque_replay=torque,
            scientific=scientific,
            product=product,
        ),
        ambiguity=ambiguity,
        limitations=profile.limitations,
    )
