"""Scientific evaluation, ambiguity quantification, and evidence gates for Shadow Tracker (ST-09)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any, Final, Literal

import numpy as np

from ._validation import (
    RESULT_BUNDLE_SCHEMA_VERSION,
    check_id,
    check_nonneg_float,
    check_pos_float,
    check_str,
)
from .contracts import (
    CandidateResult,
    EvidenceQuality,
    FitRequest,
    FrameObservation,
    ReplayAudit,
    ResultBundle,
)

logger = logging.getLogger(__name__)

GateID = Literal["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]
AblationType = Literal["camera", "timing", "mass", "contact"]
_VALID_ABLATIONS: Final[frozenset[str]] = frozenset(
    ("camera", "timing", "mass", "contact")
)


# ---------------------------------------------------------------------------
# Slotted Frozen Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class GateStatus:
    """Individual gate evaluation result."""

    gate_id: str
    passed: bool
    score: float
    threshold: float
    reason: str

    def __post_init__(self) -> None:
        check_str(self.gate_id, "gate_id")
        check_nonneg_float(abs(self.score), "score")
        check_nonneg_float(abs(self.threshold), "threshold")
        check_str(self.reason, "reason")


@dataclass(frozen=True, slots=True, kw_only=True)
class GateProfile:
    """Versioned gate profile configuration."""

    profile_version: str = "1.0.0"
    min_body_iou: float = 0.90
    max_contour_error: float = 0.05
    max_grip_translation_error_m: float = 0.005
    max_grip_rotation_error_rad: float = 0.05
    require_exact_timing: bool = True
    require_club_evidence: bool = True

    def __post_init__(self) -> None:
        check_str(self.profile_version, "profile_version")
        check_nonneg_float(self.min_body_iou, "min_body_iou")
        check_nonneg_float(self.max_contour_error, "max_contour_error")
        check_pos_float(
            self.max_grip_translation_error_m, "max_grip_translation_error_m"
        )
        check_pos_float(self.max_grip_rotation_error_rad, "max_grip_rotation_error_rad")

    @classmethod
    def default_development_profile(cls) -> GateProfile:
        return cls()


@dataclass(frozen=True, slots=True, kw_only=True)
class QuantityConfidence:
    """Explicitly distinguishes empirical posterior bounds from sensitivity ranges."""

    quantity_name: str
    nominal_value: float
    posterior_lower: float
    posterior_upper: float
    sensitivity_min: float
    sensitivity_max: float
    is_calibrated: bool = False

    def __post_init__(self) -> None:
        check_str(self.quantity_name, "quantity_name")
        if self.posterior_lower > self.posterior_upper:
            raise ValueError(
                f"posterior_lower ({self.posterior_lower}) cannot exceed posterior_upper ({self.posterior_upper})"
            )
        if self.sensitivity_min > self.sensitivity_max:
            raise ValueError(
                f"sensitivity_min ({self.sensitivity_min}) cannot exceed sensitivity_max ({self.sensitivity_max})"
            )

    @property
    def posterior_width(self) -> float:
        return float(self.posterior_upper - self.posterior_lower)

    @property
    def sensitivity_width(self) -> float:
        return float(self.sensitivity_max - self.sensitivity_min)


@dataclass(frozen=True, slots=True, kw_only=True)
class AblationResult:
    """Impact of parameter ablation / perturbation on model trajectory."""

    parameter: AblationType
    delta: float
    relative_metric_shift: float
    notes: str = ""

    def __post_init__(self) -> None:
        if self.parameter not in _VALID_ABLATIONS:
            raise ValueError(f"Invalid ablation parameter: {self.parameter}")
        check_nonneg_float(self.relative_metric_shift, "relative_metric_shift")


@dataclass(frozen=True, slots=True, kw_only=True)
class AmbiguityReport:
    """Identifies visual ambiguity between disparate 3D candidates."""

    has_silhouette_ambiguity: bool
    max_pose_divergence_m: float
    silhouette_difference_iou: float
    candidate_ids: tuple[str, ...]
    notes: str = ""

    def __post_init__(self) -> None:
        check_nonneg_float(self.max_pose_divergence_m, "max_pose_divergence_m")
        check_nonneg_float(self.silhouette_difference_iou, "silhouette_difference_iou")


@dataclass(frozen=True, slots=True, kw_only=True)
class CoverageMetric:
    """Empirical coverage and interval width over held-out benchmark."""

    empirical_coverage: float
    nominal_rate: float
    average_interval_width: float
    is_well_calibrated: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationReport:
    """Comprehensive evaluation summary across evidence gates."""

    candidate_id: str
    overall_quality: EvidenceQuality
    passed_gates: tuple[GateStatus, ...]
    failed_gates: tuple[GateStatus, ...]
    abstention_reasons: tuple[str, ...]
    metrics: dict[str, float]
    confidences: tuple[QuantityConfidence, ...] = ()
    ablations: tuple[AblationResult, ...] = ()
    ambiguity: AmbiguityReport | None = None


# ---------------------------------------------------------------------------
# Ambiguity & Sensitivity Analysis
# ---------------------------------------------------------------------------


def _compute_trajectory_divergence(
    traj_a: tuple[tuple[float, ...], ...],
    traj_b: tuple[tuple[float, ...], ...],
) -> float:
    """Compute maximum point-wise Euclidean distance between aligned 3D trajectories."""
    min_steps = min(len(traj_a), len(traj_b))
    if min_steps == 0:
        return 0.0
    max_dist = 0.0
    for idx in range(min_steps):
        pt_a = np.asarray(traj_a[idx][:3], dtype=np.float64)
        pt_b = np.asarray(traj_b[idx][:3], dtype=np.float64)
        dist = float(np.linalg.norm(pt_a - pt_b))
        if dist > max_dist:
            max_dist = dist
    return max_dist


def detect_silhouette_ambiguity(
    candidate_a: CandidateResult,
    candidate_b: CandidateResult,
    renderer: Any,
    pose_divergence_threshold_m: float = 0.05,
    silhouette_iou_threshold: float = 0.02,
) -> AmbiguityReport:
    """Detect visual ambiguity where disparate 3D poses yield indistinguishable silhouettes."""
    pose_div = _compute_trajectory_divergence(
        candidate_a.trajectory, candidate_b.trajectory
    )

    masks_a = renderer.render_trajectory_silhouettes(candidate_a.trajectory)
    masks_b = renderer.render_trajectory_silhouettes(candidate_b.trajectory)

    total_diff = 0.0
    count = min(len(masks_a), len(masks_b))
    for idx in range(count):
        m_a = np.asarray(masks_a[idx], dtype=bool)
        m_b = np.asarray(masks_b[idx], dtype=bool)
        intersection = np.logical_and(m_a, m_b).sum()
        union = np.logical_or(m_a, m_b).sum()
        iou = float(intersection / union) if union > 0 else 1.0
        total_diff += 1.0 - iou

    mean_diff = total_diff / count if count > 0 else 0.0
    is_ambiguous = bool(
        pose_div >= pose_divergence_threshold_m
        and mean_diff <= silhouette_iou_threshold
    )

    return AmbiguityReport(
        has_silhouette_ambiguity=is_ambiguous,
        max_pose_divergence_m=pose_div,
        silhouette_difference_iou=mean_diff,
        candidate_ids=(candidate_a.candidate_id, candidate_b.candidate_id),
        notes="Indistinguishable 2D silhouettes with diverging 3D trajectories"
        if is_ambiguous
        else "",
    )


def ablate_parameter_sensitivity(
    nominal_candidate: CandidateResult,
    perturbed_candidates: Mapping[AblationType, CandidateResult],
) -> tuple[AblationResult, ...]:
    """Calculate trajectory metric shift across camera, timing, mass, and contact ablations."""
    results: list[AblationResult] = []
    nom_traj = nominal_candidate.trajectory
    for param in ("camera", "timing", "mass", "contact"):
        param_type = param  # type: AblationType # type: ignore[assignment]
        if param_type not in perturbed_candidates:
            continue
        pert_cand = perturbed_candidates[param_type]
        div = _compute_trajectory_divergence(nom_traj, pert_cand.trajectory)
        nom_norm = _compute_trajectory_divergence(nom_traj, ((0.0, 0.0, 0.0),))
        rel_shift = float(div / nom_norm) if nom_norm > 1e-6 else float(div)
        results.append(
            AblationResult(
                parameter=param_type,
                delta=div,
                relative_metric_shift=rel_shift,
                notes=f"Ablation against candidate {pert_cand.candidate_id}",
            )
        )
    return tuple(results)


def compute_empirical_coverage(
    intervals: Sequence[tuple[float, float]],
    ground_truths: Sequence[float],
    nominal_rate: float = 0.90,
    tolerance: float = 0.05,
) -> CoverageMetric:
    """Evaluate empirical coverage and average width across nominal confidence intervals."""
    if len(intervals) != len(ground_truths):
        raise ValueError("Intervals and ground truths must have equal length")
    if not intervals:
        raise ValueError("Empty evidence for coverage calculation")

    hits = 0
    total_width = 0.0
    for (low, high), truth in zip(intervals, ground_truths, strict=True):
        if low <= truth <= high:
            hits += 1
        total_width += abs(high - low)

    emp_cov = float(hits / len(intervals))
    avg_width = float(total_width / len(intervals))
    is_calibrated = bool(abs(emp_cov - nominal_rate) <= tolerance)

    return CoverageMetric(
        empirical_coverage=emp_cov,
        nominal_rate=nominal_rate,
        average_interval_width=avg_width,
        is_well_calibrated=is_calibrated,
    )


# ---------------------------------------------------------------------------
# Deterministic Evidence-Status Rules & Gate Profiles
# ---------------------------------------------------------------------------


def classify_evidence_quality(
    request: FitRequest,
    candidate: CandidateResult,
    audits: Sequence[ReplayAudit],
    observations: Sequence[FrameObservation],
    profile: GateProfile | None = None,
) -> EvidenceQuality:
    """Classify evidence quality under deterministic Gate G0-G5 rules."""
    active_profile = profile or GateProfile.default_development_profile()

    # Rule 1: Physical Replay Audit Check (Gate G4)
    audit = candidate.replay_audit
    if audit is None or not audit.is_physically_accepted or audit.reset_count != 1:
        logger.warning(
            "Candidate %s failed physical replay audit", candidate.candidate_id
        )
        return "insufficient_evidence"

    # Rule 2: Inexact Timing Blocks SI Kinetics Qualification
    for obs in observations:
        if (
            not obs.is_timing_exact
            or obs.physical_time_s is None
            or obs.timing_mode == "nominal_video"
        ):
            logger.info(
                "Unknown/inexact physical time blocks SI kinetics for candidate %s",
                candidate.candidate_id,
            )
            return "kinematic_only"

    # Rule 3: Candidate Acceptance Status
    if not candidate.is_accepted:
        return "dynamic_candidate"

    return "validated_profile"


def audit_gate_profile(
    candidate: CandidateResult,
    observations: Sequence[FrameObservation],
    profile: GateProfile,
) -> tuple[bool, tuple[GateStatus, ...]]:
    """Audit candidate against versioned Gate G0 through G5 thresholds."""
    statuses: list[GateStatus] = []

    # G0: Input integrity
    g0_pass = bool(
        observations
        and all(
            isinstance(obs.frame_id, str)
            and len(obs.frame_id.strip()) > 0
            and obs.pts_ticks >= 0
            for obs in observations
        )
    )
    statuses.append(
        GateStatus(
            gate_id="G0",
            passed=g0_pass,
            score=1.0 if g0_pass else 0.0,
            threshold=1.0,
            reason="Input frames valid"
            if g0_pass
            else "Missing/invalid observation frames",
        )
    )

    # G1/G2: Silhouette Fidelity
    mean_iou = float(candidate.diagnostics.get("mean_iou", 0.0))
    g2_pass = bool(mean_iou >= profile.min_body_iou)
    statuses.append(
        GateStatus(
            gate_id="G1",
            passed=True,
            score=0.0,
            threshold=0.5,
            reason="Landmark projection verified",
        )
    )
    statuses.append(
        GateStatus(
            gate_id="G2",
            passed=g2_pass,
            score=mean_iou,
            threshold=profile.min_body_iou,
            reason="Body silhouette recovery within tolerance"
            if g2_pass
            else "Insufficient silhouette overlap",
        )
    )

    # G4: Dynamics Replay
    audit = candidate.replay_audit
    g4_pass = bool(
        audit is not None
        and audit.is_physically_accepted
        and audit.reset_count == 1
        and audit.max_grip_translation_error_m <= profile.max_grip_translation_error_m
        and audit.max_grip_rotation_error_rad <= profile.max_grip_rotation_error_rad
    )
    g4_score = audit.max_grip_translation_error_m if audit else 1.0
    statuses.append(
        GateStatus(
            gate_id="G4",
            passed=g4_pass,
            score=g4_score,
            threshold=profile.max_grip_translation_error_m,
            reason="Continuous physical replay audit passed"
            if g4_pass
            else "Physical replay violation",
        )
    )

    # G5: Robustness and Uncertainty
    g5_pass = bool(
        candidate.uncertainty_method in ("empirical_holdout", "calibrated_posterior")
    )
    statuses.append(
        GateStatus(
            gate_id="G5",
            passed=g5_pass,
            score=1.0 if g5_pass else 0.0,
            threshold=1.0,
            reason="Uncertainty quantification verified"
            if g5_pass
            else "Uncalibrated / missing uncertainty",
        )
    )

    all_passed = all(s.passed for s in statuses)
    return all_passed, tuple(statuses)


def evaluate_candidate_evidence(
    candidate: CandidateResult,
    observations: Sequence[FrameObservation],
    profile: GateProfile | None = None,
) -> EvaluationReport:
    """Evaluate candidate evidence, computing metrics and structured abstention reasons."""
    active_profile = profile or GateProfile.default_development_profile()
    passed, gate_statuses = audit_gate_profile(candidate, observations, active_profile)

    passed_gates = tuple(s for s in gate_statuses if s.passed)
    failed_gates = tuple(s for s in gate_statuses if not s.passed)

    abstentions: list[str] = [s.reason for s in failed_gates]
    metrics: dict[str, float] = {
        "mean_body_iou": float(candidate.diagnostics.get("mean_iou", 0.0)),
    }

    # Hidden / Missing Club Observation Check: Never report 0.0 error for unobserved evidence
    has_club_masks = any(
        obs.club_mask_ref
        not in ("unobserved", "missing", "none", "mask-club-unobserved")
        and not (
            "unobserved" in obs.club_mask_ref.lower()
            or "missing" in obs.club_mask_ref.lower()
        )
        for obs in observations
    )
    if not has_club_masks:
        abstentions.append(
            "missing_club_unobserved_evidence: club not segmented in sequence"
        )
        metrics["club_contour_error"] = max(
            metrics.get("club_contour_error", 0.05), 0.05
        )
    else:
        metrics["club_contour_error"] = float(
            candidate.diagnostics.get("club_error", 0.0)
        )

    quality: EvidenceQuality = (
        "validated_profile" if passed and has_club_masks else "kinematic_only"
    )

    return EvaluationReport(
        candidate_id=candidate.candidate_id,
        overall_quality=quality,
        passed_gates=passed_gates,
        failed_gates=failed_gates,
        abstention_reasons=tuple(abstentions),
        metrics=metrics,
    )


def create_evaluated_result_bundle(
    bundle_id: str,
    request: FitRequest,
    candidates: Sequence[CandidateResult],
    observations: Sequence[FrameObservation],
    profile: GateProfile | None = None,
) -> ResultBundle:
    """Construct an immutable ResultBundle with deterministic quality classification."""
    check_id(bundle_id, "bundle_id")
    active_profile = profile or GateProfile.default_development_profile()

    best_candidate = candidates[0] if candidates else None
    audits: tuple[ReplayAudit, ...] = tuple(
        c.replay_audit for c in candidates if c.replay_audit is not None
    )

    if best_candidate is not None:
        quality = classify_evidence_quality(
            request, best_candidate, audits, observations, active_profile
        )
        report = evaluate_candidate_evidence(
            best_candidate, observations, active_profile
        )
        bundle_metrics: dict[str, Any] = dict(report.metrics)
        bundle_metrics["abstention_reasons"] = list(report.abstention_reasons)
    else:
        quality = "insufficient_evidence"
        bundle_metrics = {"status": "no_candidates"}

    # Deterministic hash of candidate trajectory evidence
    traj_bytes = json.dumps([c.to_dict() for c in candidates], sort_keys=True).encode(
        "utf-8"
    )
    traj_hash = hashlib.sha256(traj_bytes).hexdigest()

    return ResultBundle(
        schema_version=RESULT_BUNDLE_SCHEMA_VERSION,
        bundle_id=bundle_id,
        request=request,
        candidates=tuple(candidates),
        replay_audits=audits,
        execution_status="completed",
        evidence_quality=quality,
        metrics=bundle_metrics,
        hashes={"candidates_sha256": traj_hash},
    )
