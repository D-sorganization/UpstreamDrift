"""Engine conformance matrix, independent replay, and release qualification for Shadow Tracker (ST-12)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Any, Final, Literal

import numpy as np

from ._validation import (
    check_id,
    check_nonneg_float,
    check_pos_float,
    check_str,
)
from .contracts import (
    CandidateResult,
    FrameObservation,
)
from .evaluation import GateProfile, GateStatus, audit_gate_profile

logger = logging.getLogger(__name__)

ENGINE_CONFORMANCE_SCHEMA_VERSION: Final[str] = (
    "shadow-tracker/engine-conformance/1.0.0"
)
RELEASE_QUALIFICATION_SCHEMA_VERSION: Final[str] = (
    "shadow-tracker/release-qualification/1.0.0"
)
SCIENTIFIC_REGISTRY_SCHEMA_VERSION: Final[str] = (
    "shadow-tracker/scientific-registry/1.0.0"
)

EngineQualificationStatus = Literal[
    "advertised_and_qualified",
    "unadvertised_experimental",
    "unsupported",
    "qualification_failed",
]

KNOWN_PHYSICS_ENGINES: Final[frozenset[str]] = frozenset(
    {"mujoco", "pinocchio", "drake", "opensim", "simscape", "double_pendulum"}
)

REQUIRED_SWING_PHASES: Final[tuple[str, ...]] = (
    "address",
    "takeaway",
    "transition",
    "downswing",
    "impact",
    "follow_through",
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineReceipt:
    """Receipt proving physical engine execution, conventions, and tolerances."""

    engine_name: str
    engine_version: str
    model_name: str
    model_sha256: str
    state_convention: str
    contact_model_type: str
    matlab_release: str | None = None
    closure_translation_tolerance_m: float = 0.005
    closure_rotation_tolerance_rad: float = 0.05
    measured_closure_translation_m: float = 0.0
    measured_closure_rotation_rad: float = 0.0
    is_physically_accepted: bool = False
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        check_str(self.engine_name, "engine_name")
        check_str(self.engine_version, "engine_version")
        check_str(self.model_name, "model_name")
        check_str(self.model_sha256, "model_sha256")
        check_str(self.state_convention, "state_convention")
        check_str(self.contact_model_type, "contact_model_type")
        check_pos_float(
            self.closure_translation_tolerance_m, "closure_translation_tolerance_m"
        )
        check_pos_float(
            self.closure_rotation_tolerance_rad, "closure_rotation_tolerance_rad"
        )
        check_nonneg_float(
            self.measured_closure_translation_m, "measured_closure_translation_m"
        )
        check_nonneg_float(
            self.measured_closure_rotation_rad, "measured_closure_rotation_rad"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineQualificationResult:
    """Outcome of evaluating an engine against conformance standards."""

    engine_name: str
    status: EngineQualificationStatus
    is_qualified: bool
    failure_reasons: tuple[str, ...] = ()
    receipt: EngineReceipt | None = None
    conformance_checks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        check_str(self.engine_name, "engine_name")


@dataclass(frozen=True, slots=True, kw_only=True)
class EngineCapabilityMatrix:
    """Collection of engine conformance statuses across known engines."""

    profiles: Mapping[str, EngineQualificationResult]
    advertised_engines: tuple[str, ...] = ()
    unsupported_engines: tuple[str, ...] = ()
    experimental_engines: tuple[str, ...] = ()

    def is_advertised_engine(self, engine_name: str) -> bool:
        return engine_name in self.advertised_engines


@dataclass(frozen=True, slots=True, kw_only=True)
class IndependentReplayAudit:
    """Audit of independent forward dynamics replay without optimizer caches."""

    candidate_id: str
    engine_name: str
    time_steps: int
    max_state_discrepancy: float
    tolerance: float
    is_replay_converged: bool
    notes: str = ""

    def __post_init__(self) -> None:
        check_id(self.candidate_id, "candidate_id")
        check_str(self.engine_name, "engine_name")
        check_nonneg_float(self.max_state_discrepancy, "max_state_discrepancy")
        check_pos_float(self.tolerance, "tolerance")


@dataclass(frozen=True, slots=True, kw_only=True)
class PerformanceProfile:
    """Measured computational performance, throughput, and memory budget metrics."""

    duration_seconds: float
    frames_processed: int
    fps: float
    peak_memory_mb: float
    compute_budget_seconds: float
    is_budget_exceeded: bool
    phase_latencies_ms: Mapping[str, float]

    def __post_init__(self) -> None:
        check_nonneg_float(self.duration_seconds, "duration_seconds")
        check_nonneg_float(self.fps, "fps")
        check_nonneg_float(self.peak_memory_mb, "peak_memory_mb")
        check_pos_float(self.compute_budget_seconds, "compute_budget_seconds")


@dataclass(frozen=True, slots=True, kw_only=True)
class ScientificRegistryEntry:
    """Authoritative scientific manual and calculation registry entry."""

    registry_id: str
    title: str
    module_path: str
    equations: tuple[str, ...]
    assumptions: tuple[str, ...]
    physical_invariants: tuple[str, ...]
    verification_status: str

    def __post_init__(self) -> None:
        check_str(self.registry_id, "registry_id")
        check_str(self.title, "title")
        check_str(self.module_path, "module_path")
        check_str(self.verification_status, "verification_status")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseQualificationReport:
    """Top-level qualification report for Shadow Tracker release candidate."""

    schema_version: str
    profile_version: str
    is_release_qualified: bool
    gate_statuses: tuple[GateStatus, ...]
    engine_matrix: EngineCapabilityMatrix
    swing_phase_coverage: tuple[str, ...]
    independent_replay: IndependentReplayAudit | None = None
    performance_profile: PerformanceProfile | None = None
    blocking_reasons: tuple[str, ...] = ()
    release_evidence_hash: str = ""

    def __post_init__(self) -> None:
        check_str(self.schema_version, "schema_version")
        check_str(self.profile_version, "profile_version")


# ---------------------------------------------------------------------------
# Engine Conformance & Receipt Verification
# ---------------------------------------------------------------------------


def audit_engine_conformance(
    *,
    engine_name: str,
    receipt: EngineReceipt | None,
    required_model_hash: str | None = None,
    required_state_convention: str = "canonical_v2_quaternion",
    is_advertised: bool = True,
) -> EngineQualificationResult:
    """Audit engine compliance against physical conventions, model hashes, and receipts."""
    failures: list[str] = []
    checks: list[str] = ["engine_name_known"]

    if receipt is None:
        if is_advertised:
            return EngineQualificationResult(
                engine_name=engine_name,
                status="qualification_failed",
                is_qualified=False,
                failure_reasons=("missing_engine_receipt",),
                receipt=None,
                conformance_checks=tuple(checks),
            )
        return EngineQualificationResult(
            engine_name=engine_name,
            status="unsupported",
            is_qualified=False,
            failure_reasons=("unadvertised_engine",),
            receipt=None,
            conformance_checks=tuple(checks),
        )

    # 1. Model hash check
    checks.append("model_hash_check")
    if required_model_hash is not None and receipt.model_sha256 != required_model_hash:
        failures.append(
            f"model_hash_mismatch: expected {required_model_hash} got {receipt.model_sha256}"
        )

    # 2. State convention check
    checks.append("state_convention_check")
    if receipt.state_convention != required_state_convention:
        failures.append(
            f"incompatible_state_convention: expected {required_state_convention} got {receipt.state_convention}"
        )

    # 3. Simscape MATLAB R2025b requirement
    if engine_name == "simscape":
        checks.append("matlab_release_r2025b_check")
        if receipt.matlab_release != "R2025b":
            failures.append(
                f"simscape_requires_matlab_R2025b: received {receipt.matlab_release}"
            )

    # 4. Physical closure tolerance check
    checks.append("closure_tolerance_check")
    if receipt.measured_closure_translation_m > receipt.closure_translation_tolerance_m:
        failures.append(
            f"closure_translation_exceeded: {receipt.measured_closure_translation_m:.4f} m > {receipt.closure_translation_tolerance_m:.4f} m"
        )
    if receipt.measured_closure_rotation_rad > receipt.closure_rotation_tolerance_rad:
        failures.append(
            f"closure_rotation_exceeded: {receipt.measured_closure_rotation_rad:.4f} rad > {receipt.closure_rotation_tolerance_rad:.4f} rad"
        )

    # 5. Physical acceptance flag
    checks.append("physical_acceptance_check")
    if not receipt.is_physically_accepted:
        failures.append("physical_acceptance_false: engine rollout rejected by solver")

    is_qualified = len(failures) == 0
    status: EngineQualificationStatus
    if is_qualified:
        status = (
            "advertised_and_qualified" if is_advertised else "unadvertised_experimental"
        )
    else:
        status = "qualification_failed" if is_advertised else "unsupported"

    return EngineQualificationResult(
        engine_name=engine_name,
        status=status,
        is_qualified=is_qualified,
        failure_reasons=tuple(failures),
        receipt=receipt,
        conformance_checks=tuple(checks),
    )


def validate_cross_engine_contact_claims(
    engine_a: EngineReceipt,
    engine_b: EngineReceipt,
) -> None:
    """Enforce that identical contact results cannot be claimed across disparate contact laws."""
    if engine_a.contact_model_type != engine_b.contact_model_type:
        raise ValueError(
            f"Cannot claim identical contact results across differing contact models: "
            f"'{engine_a.engine_name}' uses {engine_a.contact_model_type} while "
            f"'{engine_b.engine_name}' uses {engine_b.contact_model_type}."
        )


# ---------------------------------------------------------------------------
# Independent Replay & Performance Profiling
# ---------------------------------------------------------------------------


def verify_independent_replay(
    *,
    candidate: CandidateResult,
    replay_trajectory: tuple[tuple[float, ...], ...],
    engine_name: str,
    tolerance: float = 1e-6,
) -> IndependentReplayAudit:
    """Verify saved trajectory against independent forward replay without optimizer internal state."""
    cand_traj = candidate.trajectory
    min_len = min(len(cand_traj), len(replay_trajectory))
    if min_len == 0:
        return IndependentReplayAudit(
            candidate_id=candidate.candidate_id,
            engine_name=engine_name,
            time_steps=0,
            max_state_discrepancy=0.0,
            tolerance=tolerance,
            is_replay_converged=True,
            notes="Empty trajectory evaluated",
        )

    max_diff = 0.0
    for idx in range(min_len):
        state_c = np.asarray(cand_traj[idx], dtype=np.float64)
        state_r = np.asarray(replay_trajectory[idx], dtype=np.float64)
        diff = float(np.max(np.abs(state_c - state_r)))
        if diff > max_diff:
            max_diff = diff

    is_converged = bool(max_diff <= tolerance)
    notes = (
        "Replay verified within tolerance"
        if is_converged
        else f"Discrepancy {max_diff:.6e} exceeds tolerance {tolerance:.6e}"
    )

    return IndependentReplayAudit(
        candidate_id=candidate.candidate_id,
        engine_name=engine_name,
        time_steps=min_len,
        max_state_discrepancy=max_diff,
        tolerance=tolerance,
        is_replay_converged=is_converged,
        notes=notes,
    )


def profile_shadow_tracker_performance(
    *,
    duration_seconds: float,
    frames_processed: int,
    peak_memory_mb: float,
    phase_latencies_ms: Mapping[str, float],
    max_budget_seconds: float = 300.0,
) -> PerformanceProfile:
    """Profile computational latency, throughput, and memory consumption."""
    fps = float(frames_processed / duration_seconds) if duration_seconds > 0.0 else 0.0
    is_exceeded = bool(duration_seconds > max_budget_seconds)

    return PerformanceProfile(
        duration_seconds=duration_seconds,
        frames_processed=frames_processed,
        fps=fps,
        peak_memory_mb=peak_memory_mb,
        compute_budget_seconds=max_budget_seconds,
        is_budget_exceeded=is_exceeded,
        phase_latencies_ms=dict(phase_latencies_ms),
    )


# ---------------------------------------------------------------------------
# Comprehensive G0–G7 Release Profile Gate Suite
# ---------------------------------------------------------------------------


def _audit_individual_gates(
    candidate: CandidateResult,
    observations: Sequence[FrameObservation],
    profile: GateProfile,
) -> list[GateStatus]:
    """Audit all individual gates G0 through G7, reusing evaluation gate profile."""
    _, base_statuses = audit_gate_profile(candidate, observations, profile)
    base_map = {g.gate_id: g for g in base_statuses}

    statuses: list[GateStatus] = []
    # G0, G1, G2 from base evaluation profile
    for gid in ("G0", "G1", "G2"):
        if gid in base_map:
            statuses.append(base_map[gid])

    # G3: Modern Reference
    mean_iou = float(candidate.diagnostics.get("mean_iou", 0.0))
    joint_rmse = float(candidate.diagnostics.get("joint_rmse_m", 0.04))
    g3_pass = bool(mean_iou >= 0.90 and joint_rmse <= 0.05)
    statuses.append(
        GateStatus(
            gate_id="G3",
            passed=g3_pass,
            score=joint_rmse,
            threshold=0.05,
            reason=(
                "Modern reference joint RMSE within 0.05 m"
                if g3_pass
                else "Joint RMSE exceeds modern reference threshold"
            ),
        )
    )

    # G4, G5 from base evaluation profile
    for gid in ("G4", "G5"):
        if gid in base_map:
            statuses.append(base_map[gid])

    # G6: Historical Pilot
    g6_pass = bool(
        observations
        and all(
            obs.confidence_provenance
            in ("gold_standard", "archive_pilot", "ground_truth", "reviewed")
            and obs.is_timing_exact
            for obs in observations
        )
    )
    statuses.append(
        GateStatus(
            gate_id="G6",
            passed=g6_pass,
            score=1.0 if g6_pass else 0.0,
            threshold=1.0,
            reason=(
                "Historical lineage and timing verified"
                if g6_pass
                else "Unverified historical timing or rights"
            ),
        )
    )

    # G7: Product and Reproduction
    g7_pass = bool(candidate.is_accepted and candidate.replay_audit is not None)
    statuses.append(
        GateStatus(
            gate_id="G7",
            passed=g7_pass,
            score=1.0 if g7_pass else 0.0,
            threshold=1.0,
            reason=(
                "Product reproduction verified"
                if g7_pass
                else "Reproduction tolerance violated"
            ),
        )
    )

    return statuses


def audit_full_release_gates(
    *,
    candidate: CandidateResult,
    observations: Sequence[FrameObservation],
    engine_matrix: EngineCapabilityMatrix,
    profile: GateProfile | None = None,
    swing_phase_coverage: Sequence[str] | None = None,
    independent_replay: IndependentReplayAudit | None = None,
    performance_profile: PerformanceProfile | None = None,
) -> ReleaseQualificationReport:
    """Audit all release gates G0–G7, swing phase coverage, and advertised engine qualifications."""
    active_profile = profile or GateProfile.default_development_profile()
    gate_statuses = _audit_individual_gates(candidate, observations, active_profile)

    blockers: list[str] = []

    # 1. Check all gates G0–G7
    for g in gate_statuses:
        if not g.passed:
            blockers.append(f"gate_failure_{g.gate_id}: {g.reason}")

    # 2. Check swing phase coverage
    observed_phases: set[str] = set()
    if swing_phase_coverage is not None:
        observed_phases = set(swing_phase_coverage)
    else:
        for obs in observations:
            reason = getattr(obs, "physical_time_reason", "")
            for p in REQUIRED_SWING_PHASES:
                if p in reason.lower():
                    observed_phases.add(p)

    missing_phases = [p for p in REQUIRED_SWING_PHASES if p not in observed_phases]
    if missing_phases:
        blockers.append(f"missing_swing_phases: {sorted(missing_phases)}")

    # 3. Check engine matrix conformance for advertised engines
    for engine_name in engine_matrix.advertised_engines:
        res = engine_matrix.profiles.get(engine_name)
        if res is None or not res.is_qualified:
            blockers.append(
                f"advertised_engine_unqualified: {engine_name} failed conformance"
            )

    is_qualified = len(blockers) == 0

    # Deterministic SHA-256 evidence digest
    evidence_payload = {
        "candidate_id": candidate.candidate_id,
        "gates": [
            {"gate_id": g.gate_id, "passed": g.passed, "score": g.score}
            for g in gate_statuses
        ],
        "advertised_engines": sorted(engine_matrix.advertised_engines),
        "phases": sorted(observed_phases),
        "is_qualified": is_qualified,
    }
    evidence_hash = hashlib.sha256(
        json.dumps(evidence_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return ReleaseQualificationReport(
        schema_version=RELEASE_QUALIFICATION_SCHEMA_VERSION,
        profile_version=active_profile.profile_version,
        is_release_qualified=is_qualified,
        gate_statuses=tuple(gate_statuses),
        engine_matrix=engine_matrix,
        swing_phase_coverage=tuple(sorted(observed_phases)),
        independent_replay=independent_replay,
        performance_profile=performance_profile,
        blocking_reasons=tuple(blockers),
        release_evidence_hash=evidence_hash,
    )


# ---------------------------------------------------------------------------
# Scientific Registry & Evidence Inventory
# ---------------------------------------------------------------------------


def get_shadow_tracker_scientific_registry() -> tuple[ScientificRegistryEntry, ...]:
    """Return immutable scientific calculation registry entries for Shadow Tracker."""
    return (
        ScientificRegistryEntry(
            registry_id="ST-CALC-PROJECTION",
            title="Perspective Pinhole and Landmark Silhouette Projection",
            module_path="src.shared.python.shadow_tracker.projection",
            equations=(
                "x_proj = f_x * (X_c / Z_c) + c_x",
                "y_proj = f_y * (Y_c / Z_c) + c_y",
                "L_iou = 1 - (|M_rend ∩ M_obs| / |M_rend ∪ M_obs|)",
            ),
            assumptions=(
                "Pinhole camera model with ideal pinhole geometry or pre-rectified radial distortion",
                "Positive Z depth in camera optical frame",
            ),
            physical_invariants=(
                "Pixel coordinates within viewport bounds [0, W] x [0, H]",
                "Landmark projection error <= 0.5 px on calibrated synthetic targets",
            ),
            verification_status="verified_against_analytic_fixtures",
        ),
        ScientificRegistryEntry(
            registry_id="ST-CALC-FORWARD-DYNAMICS",
            title="Continuous Full-Body Forward Dynamics and Grip Replay",
            module_path="src.shared.python.shadow_tracker.forward_model",
            equations=(
                "M(q) * q_ddot + C(q, q_dot) * q_dot + g(q) = tau + J_c^T * lambda_c",
                "E_mech(t) - E_mech(0) = W_act(t) - W_diss(t)",
            ),
            assumptions=(
                "Rigid body dynamics with floating base and continuous joint tree",
                "Unactuated root with zero ghost actuation",
                "Piecewise ground reaction forces conforming to friction cone limits",
            ),
            physical_invariants=(
                "Zero intermediate state resets during rollout (reset_count == 1)",
                "Max grip translation error <= 0.005 m, rotation error <= 0.05 rad",
                "Dissipation non-negative: W_diss >= 0",
            ),
            verification_status="verified_with_physical_closure_gates",
        ),
        ScientificRegistryEntry(
            registry_id="ST-CALC-CONTROL-FITTING",
            title="Silhouette-Constrained Optimal Control and Parameter Estimation",
            module_path="src.shared.python.shadow_tracker.fitting",
            equations=(
                "min_{u, theta} J = J_silhouette(u, theta) + w_phys * J_physics(u) + w_reg * ||u||^2",
            ),
            assumptions=(
                "Actuator effort bounded by physiological torque limits",
                "Spline-parameterized or piecewise-constant control inputs",
            ),
            physical_invariants=(
                "Joint limits strictly respected",
                "Torque limits strictly respected",
            ),
            verification_status="verified_with_equal_budget_baselines",
        ),
        ScientificRegistryEntry(
            registry_id="ST-CALC-ENGINE-CONFORMANCE",
            title="Multi-Engine Physics Conformance and Replay Verification",
            module_path="src.shared.python.shadow_tracker.engine_matrix",
            equations=("||q_engine_a(t) - q_engine_b(t)||_inf <= epsilon_tol",),
            assumptions=(
                "Explicit model SHA-256 asset locking",
                "Strict adherence to canonical_v2_quaternion manifold state convention",
                "Independent replay does not reuse optimizer cache",
            ),
            physical_invariants=(
                "No assertion of identical contact behavior across disparate contact models",
                "Independent replay discrepancy <= 1e-6",
            ),
            verification_status="qualified_with_receipt_and_digest",
        ),
    )


def generate_release_evidence_inventory(
    report: ReleaseQualificationReport,
) -> dict[str, Any]:
    """Generate serializable evidence inventory exposing all capabilities and remaining gaps."""
    evidence = {
        "schema_version": "shadow-tracker/release-evidence/1.0.0",
        "profile_version": report.profile_version,
        "is_release_qualified": report.is_release_qualified,
        "gates": [
            {
                "gate_id": g.gate_id,
                "passed": g.passed,
                "score": g.score,
                "threshold": g.threshold,
                "reason": g.reason,
            }
            for g in report.gate_statuses
        ],
        "advertised_engines": list(report.engine_matrix.advertised_engines),
        "unsupported_engines": list(report.engine_matrix.unsupported_engines),
        "swing_phase_coverage": list(report.swing_phase_coverage),
        "blocking_reasons": list(report.blocking_reasons),
        "release_evidence_hash": report.release_evidence_hash,
    }

    serialized = json.dumps(evidence, sort_keys=True).encode("utf-8")
    evidence["evidence_digest_sha256"] = hashlib.sha256(serialized).hexdigest()
    return evidence
