"""Pinocchio driver/iron G2–G3 continuation and independent replay (MS-111, #10385).

Pure software contracts for:
- G2/G3 horizon continuation schedules (driver and iron)
- Same-integrator solve/replay parity (reject integrator-specific solutions)
- Declared armature propagation to equivalent-model replay
- Failed continuation evidence preservation
- Open-loop control feed from q0/v0 (no per-frame pose prescription)
- Timestep / contact-transition / initial-state perturbation check roster
- Independent replay package save/reopen

Native ControlTower qualification is out of scope here. Status payloads always
report ``claims_native_success=False`` until a linked MS-100 receipt proves
otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from src.shared.python.motion_matching.named_state import (
        CaptureAttachmentDeclaration,
    )

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition, require
from src.shared.python.motion_matching.acceptance import Horizon

SCHEMA_VERSION = "pinocchio-g2-g3-continuation/1.0.0"
DEFAULT_ARMATURE_KG_M2 = 5e-3
DEFAULT_RK45_RTOL = 1e-6
QUALIFIED_INTEGRATOR = "rk45"
G1_END_S = 0.85
G2_END_S = 1.20
DRIVER_G3_END_S = 1.814
IRON_G3_END_S = 1.827

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_ARMATURE_KG_M2",
    "ArmaturePlantDeclaration",
    "ClubKind",
    "ContinuationSchedule",
    "FailedContinuationEvidence",
    "IndependentReplayPackage",
    "IntegratorConfig",
    "IntegratorParityVerdict",
    "OpenLoopFeedRequest",
    "OpenLoopFeedVerdict",
    "QualificationClaim",
    "RobustnessCheck",
    "RobustnessCheckKind",
    "build_continuation_schedule",
    "CandidatePromotionRequest",
    "CandidatePromotionVerdict",
    "CrossEngineTransferMatrix",
    "EngineTransferVerdict",
    "TorqueTransferTolerances",
    "evaluate_candidate_promotion",
    "evaluate_cross_engine_torque_transfer",
    "evaluate_integrator_parity",
    "export_independent_replay_package",
    "import_independent_replay_package",
    "ms111_contract_status",
    "propagate_armature_to_replay",
    "record_failed_continuation_stage",
    "required_robustness_checks",
    "validate_open_loop_control_feed",
]


class ClubKind(str, Enum):
    """Tour capture club family for Pinocchio G2/G3 rows."""

    DRIVER = "driver"
    IRON = "iron"


class QualificationClaim(str, Enum):
    """Honest qualification ladder; native success requires receipt evidence."""

    CONTRACT_READY = "contract_ready"
    AWAITING_NATIVE_EVIDENCE = "awaiting_native_evidence"
    NATIVE_QUALIFIED = "native_qualified"


class RobustnessCheckKind(str, Enum):
    """PF-07 / MS-111 robustness probes required on driver and iron."""

    TIMESTEP_REFINEMENT = "timestep_refinement"
    CONTACT_TRANSITION = "contact_transition"
    INITIAL_STATE_PERTURBATION = "initial_state_perturbation"


@dataclass(frozen=True, slots=True)
class IntegratorConfig:
    """Declared integrator identity for solve or independent replay."""

    name: str
    rtol: float
    fixed_step: bool

    def __post_init__(self) -> None:
        require(bool(self.name.strip()), "integrator name required", self.name)
        require(self.rtol > 0.0, "rtol must be positive", self.rtol)


@dataclass(frozen=True, slots=True)
class IntegratorParityVerdict:
    """Fail-closed comparison of solve vs independent-replay integrators."""

    accepted: bool
    reason: str
    claims_native_success: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "claims_native_success": self.claims_native_success,
        }


@dataclass(frozen=True, slots=True)
class ContinuationSchedule:
    """G2/G3 stage ends continuing from a qualified G1 checkpoint."""

    club: ClubKind
    target_horizon: Horizon
    g1_end_s: float
    target_end_s: float
    stage_ends_s: tuple[float, ...]
    declared_armature_kg_m2: float
    node_integrator: str
    replay_integrator: str
    rk45_rtol: float
    document_id: str
    claims_native_success: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "club": self.club.value,
            "target_horizon": self.target_horizon.value,
            "g1_end_s": self.g1_end_s,
            "target_end_s": self.target_end_s,
            "stage_ends_s": list(self.stage_ends_s),
            "declared_armature_kg_m2": self.declared_armature_kg_m2,
            "node_integrator": self.node_integrator,
            "replay_integrator": self.replay_integrator,
            "rk45_rtol": self.rk45_rtol,
            "document_id": self.document_id,
            "claims_native_success": self.claims_native_success,
        }


@dataclass(frozen=True, slots=True)
class ArmaturePlantDeclaration:
    """Armature/inertial plant identity that equivalent-model replay must apply."""

    armature_kg_m2: float
    actuated_dof_count: int
    model_sha256: str
    source_engine: str
    target_engine: str
    equivalent_model_required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FailedContinuationEvidence:
    """Immutable record of a failed continuation stage (never upgraded to pass)."""

    club: ClubKind
    stage_end_s: float
    reason: str
    solver_cost: float
    rollout_whole_rmse_m: float
    replay_whole_rmse_m: float
    integrator: IntegratorConfig
    preserved: bool = True
    accepted: bool = False
    claims_native_success: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "club": self.club.value,
            "stage_end_s": self.stage_end_s,
            "reason": self.reason,
            "solver_cost": self.solver_cost,
            "rollout_whole_rmse_m": self.rollout_whole_rmse_m,
            "replay_whole_rmse_m": self.replay_whole_rmse_m,
            "integrator": {
                "name": self.integrator.name,
                "rtol": self.integrator.rtol,
                "fixed_step": self.integrator.fixed_step,
            },
            "preserved": self.preserved,
            "accepted": self.accepted,
            "claims_native_success": self.claims_native_success,
        }


@dataclass(frozen=True, slots=True)
class OpenLoopFeedRequest:
    """Controls allocated once from q0/v0; optional pose prescription is banned."""

    q0: NDArray[np.float64]
    v0: NDArray[np.float64]
    controls_u: NDArray[np.float64]
    timestamps_s: NDArray[np.float64]
    prescribed_poses_q: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True)
class OpenLoopFeedVerdict:
    """Open-loop feed validation outcome."""

    accepted: bool
    reason: str
    used_measured_state_reset: bool
    claims_native_success: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "used_measured_state_reset": self.used_measured_state_reset,
            "claims_native_success": self.claims_native_success,
        }


@dataclass(frozen=True, slots=True)
class RobustnessCheck:
    """Named robustness probe required before G2/G3 native qualification."""

    kind: RobustnessCheckKind
    club: ClubKind
    required_for_g2_g3: bool = True
    claims_native_success: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "club": self.club.value,
            "required_for_g2_g3": self.required_for_g2_g3,
            "claims_native_success": self.claims_native_success,
        }


@dataclass(frozen=True, slots=True)
class IndependentReplayPackage:
    """Saved independent-replay package (save/reopen identity)."""

    root: Path
    schedule: ContinuationSchedule
    armature: ArmaturePlantDeclaration
    candidate_sha256: str
    controls_sha256: str
    q0_sha256: str
    v0_sha256: str
    package_sha256: str
    qualification: QualificationClaim = QualificationClaim.CONTRACT_READY
    claims_native_success: bool = False


def _club_document_id(club: ClubKind) -> str:
    return "anthro_driver" if club is ClubKind.DRIVER else "anthro_iron"


def _target_end_s(club: ClubKind, horizon: Horizon) -> float:
    if horizon is Horizon.G2:
        return G2_END_S
    if horizon is Horizon.G3:
        return DRIVER_G3_END_S if club is ClubKind.DRIVER else IRON_G3_END_S
    raise ValueError("MS-111 continuation targets G2 or G3 only")


def _stage_ends_between(g1_end: float, target_end: float) -> tuple[float, ...]:
    """Build strictly increasing stage ends after G1 through the target horizon."""
    require(target_end > g1_end, "target must extend past G1", target_end)
    candidates: tuple[float, ...]
    if target_end <= G2_END_S + 1e-9:
        candidates = (1.00, G2_END_S)
    else:
        candidates = (1.00, G2_END_S, 1.40, 1.60, target_end)
    ends = [t for t in candidates if g1_end < t < target_end - 1e-9]
    ends.append(float(target_end))
    return tuple(ends)


@precondition(lambda club, horizon: isinstance(club, ClubKind), "club must be ClubKind")
@postcondition(lambda result: result.claims_native_success is False)
def build_continuation_schedule(
    club: ClubKind,
    horizon: Horizon,
) -> ContinuationSchedule:
    """Return the Pinocchio G2/G3 continuation schedule for ``club``."""
    require(
        horizon in (Horizon.G2, Horizon.G3),
        "MS-111 continuation targets G2 or G3 only",
        horizon,
    )
    target_end = _target_end_s(club, horizon)
    return ContinuationSchedule(
        club=club,
        target_horizon=horizon,
        g1_end_s=G1_END_S,
        target_end_s=target_end,
        stage_ends_s=_stage_ends_between(G1_END_S, target_end),
        declared_armature_kg_m2=DEFAULT_ARMATURE_KG_M2,
        node_integrator=QUALIFIED_INTEGRATOR,
        replay_integrator=QUALIFIED_INTEGRATOR,
        rk45_rtol=DEFAULT_RK45_RTOL,
        document_id=_club_document_id(club),
        claims_native_success=False,
    )


@precondition(
    lambda solve, replay: isinstance(solve, IntegratorConfig),
    "solve must be IntegratorConfig",
)
@postcondition(lambda result: result.claims_native_success is False)
def evaluate_integrator_parity(
    solve: IntegratorConfig,
    replay: IntegratorConfig,
) -> IntegratorParityVerdict:
    """Reject integrator-specific solutions (solve ≠ independent replay)."""
    if solve.name.strip().lower() != replay.name.strip().lower():
        return IntegratorParityVerdict(
            accepted=False,
            reason=(
                f"integrator mismatch: solve={solve.name!r} replay={replay.name!r}"
            ),
        )
    if not np.isclose(solve.rtol, replay.rtol, rtol=0.0, atol=0.0):
        return IntegratorParityVerdict(
            accepted=False,
            reason=f"rtol mismatch: solve={solve.rtol} replay={replay.rtol}",
        )
    if solve.fixed_step != replay.fixed_step:
        return IntegratorParityVerdict(
            accepted=False,
            reason="fixed_step mismatch between solve and replay",
        )
    return IntegratorParityVerdict(accepted=True, reason="")


def propagate_armature_to_replay(
    *,
    armature_kg_m2: float,
    actuated_dof_count: int,
    model_sha256: str,
    source_engine: str,
    target_engine: str,
) -> ArmaturePlantDeclaration:
    """Declare armature that equivalent-model replay must apply unchanged."""
    require(armature_kg_m2 >= 0.0, "armature must be nonnegative", armature_kg_m2)
    require(
        actuated_dof_count > 0,
        "actuated_dof_count must be positive",
        actuated_dof_count,
    )
    require(bool(model_sha256.strip()), "model_sha256 required", model_sha256)
    require(bool(source_engine.strip()), "source_engine required", source_engine)
    require(bool(target_engine.strip()), "target_engine required", target_engine)
    return ArmaturePlantDeclaration(
        armature_kg_m2=float(armature_kg_m2),
        actuated_dof_count=int(actuated_dof_count),
        model_sha256=model_sha256,
        source_engine=source_engine,
        target_engine=target_engine,
        equivalent_model_required=True,
    )


def record_failed_continuation_stage(
    *,
    club: ClubKind,
    stage_end_s: float,
    reason: str,
    solver_cost: float,
    rollout_whole_rmse_m: float,
    replay_whole_rmse_m: float,
    integrator: IntegratorConfig,
) -> FailedContinuationEvidence:
    """Preserve a failed continuation stage without claiming acceptance."""
    require(stage_end_s > 0.0, "stage_end_s must be positive", stage_end_s)
    require(bool(reason.strip()), "failure reason required", reason)
    require(np.isfinite(solver_cost), "solver_cost must be finite", solver_cost)
    require(
        rollout_whole_rmse_m >= 0.0 and np.isfinite(rollout_whole_rmse_m),
        "rollout_whole_rmse_m must be finite and >= 0",
        rollout_whole_rmse_m,
    )
    require(
        replay_whole_rmse_m >= 0.0 and np.isfinite(replay_whole_rmse_m),
        "replay_whole_rmse_m must be finite and >= 0",
        replay_whole_rmse_m,
    )
    return FailedContinuationEvidence(
        club=club,
        stage_end_s=float(stage_end_s),
        reason=reason.strip(),
        solver_cost=float(solver_cost),
        rollout_whole_rmse_m=float(rollout_whole_rmse_m),
        replay_whole_rmse_m=float(replay_whole_rmse_m),
        integrator=integrator,
        preserved=True,
        accepted=False,
        claims_native_success=False,
    )


def validate_open_loop_control_feed(
    request: OpenLoopFeedRequest,
) -> OpenLoopFeedVerdict:
    """Accept controls fed once from q0/v0; reject per-frame pose prescription."""
    q0 = np.asarray(request.q0, dtype=np.float64)
    v0 = np.asarray(request.v0, dtype=np.float64)
    controls = np.asarray(request.controls_u, dtype=np.float64)
    times = np.asarray(request.timestamps_s, dtype=np.float64)
    if q0.ndim != 1 or v0.ndim != 1 or q0.size != v0.size:
        return OpenLoopFeedVerdict(
            accepted=False,
            reason="q0 and v0 must be 1D and same length",
            used_measured_state_reset=False,
        )
    if not np.all(np.isfinite(q0)) or not np.all(np.isfinite(v0)):
        return OpenLoopFeedVerdict(
            accepted=False,
            reason="q0/v0 must be finite",
            used_measured_state_reset=False,
        )
    if controls.ndim != 2 or controls.shape[0] != times.size - 1:
        return OpenLoopFeedVerdict(
            accepted=False,
            reason="controls_u must be (N-1, nu) aligned with timestamps",
            used_measured_state_reset=False,
        )
    if request.prescribed_poses_q is not None:
        return OpenLoopFeedVerdict(
            accepted=False,
            reason="per-frame pose prescription is forbidden for independent replay",
            used_measured_state_reset=True,
        )
    return OpenLoopFeedVerdict(
        accepted=True,
        reason="",
        used_measured_state_reset=False,
    )


@postcondition(lambda result: len(result) == 3)
def required_robustness_checks(club: ClubKind) -> tuple[RobustnessCheck, ...]:
    """Return the driver/iron robustness roster folded in from PF-07."""
    require(isinstance(club, ClubKind), "club must be ClubKind", club)
    return tuple(
        RobustnessCheck(kind=kind, club=club)
        for kind in (
            RobustnessCheckKind.TIMESTEP_REFINEMENT,
            RobustnessCheckKind.CONTACT_TRANSITION,
            RobustnessCheckKind.INITIAL_STATE_PERTURBATION,
        )
    )


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def export_independent_replay_package(
    package_dir: Path,
    *,
    schedule: ContinuationSchedule,
    armature: ArmaturePlantDeclaration,
    candidate_sha256: str,
    controls_sha256: str,
    q0_sha256: str,
    v0_sha256: str,
) -> Path:
    """Write a portable independent-replay package with content hashes."""
    require(
        schedule.claims_native_success is False,
        "export must not claim native success without receipt",
        schedule.claims_native_success,
    )
    root = Path(package_dir)
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "schedule": schedule.as_dict(),
        "armature": armature.as_dict(),
        "candidate_sha256": candidate_sha256,
        "controls_sha256": controls_sha256,
        "q0_sha256": q0_sha256,
        "v0_sha256": v0_sha256,
        "qualification": QualificationClaim.CONTRACT_READY.value,
        "claims_native_success": False,
    }
    package_sha = _sha256_bytes(_canonical_json(payload))
    payload["package_sha256"] = package_sha
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


def import_independent_replay_package(package_dir: Path) -> IndependentReplayPackage:
    """Reopen an independent-replay package; fail closed on schema drift."""
    root = Path(package_dir)
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file(), "manifest.json missing", str(manifest_path))
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        raw.get("schema_version") == SCHEMA_VERSION,
        "schema version mismatch or missing",
        raw.get("schema_version"),
    )
    schedule_raw = raw["schedule"]
    schedule = ContinuationSchedule(
        club=ClubKind(schedule_raw["club"]),
        target_horizon=Horizon(schedule_raw["target_horizon"]),
        g1_end_s=float(schedule_raw["g1_end_s"]),
        target_end_s=float(schedule_raw["target_end_s"]),
        stage_ends_s=tuple(float(t) for t in schedule_raw["stage_ends_s"]),
        declared_armature_kg_m2=float(schedule_raw["declared_armature_kg_m2"]),
        node_integrator=str(schedule_raw["node_integrator"]),
        replay_integrator=str(schedule_raw["replay_integrator"]),
        rk45_rtol=float(schedule_raw["rk45_rtol"]),
        document_id=str(schedule_raw["document_id"]),
        claims_native_success=bool(schedule_raw.get("claims_native_success", False)),
    )
    armature = ArmaturePlantDeclaration(**raw["armature"])
    stored_sha = str(raw.get("package_sha256", ""))
    check_payload = {k: v for k, v in raw.items() if k != "package_sha256"}
    expected = _sha256_bytes(_canonical_json(check_payload))
    require(
        stored_sha == expected,
        "package_sha256 mismatch on reopen",
        stored_sha,
    )
    return IndependentReplayPackage(
        root=root,
        schedule=schedule,
        armature=armature,
        candidate_sha256=str(raw["candidate_sha256"]),
        controls_sha256=str(raw["controls_sha256"]),
        q0_sha256=str(raw["q0_sha256"]),
        v0_sha256=str(raw["v0_sha256"]),
        package_sha256=stored_sha,
        qualification=QualificationClaim(raw.get("qualification", "contract_ready")),
        claims_native_success=False,
    )


@postcondition(lambda result: result["claims_native_success"] is False)
def ms111_contract_status() -> dict[str, Any]:
    """Honest MS-111 status: contracts ready, native evidence still required."""
    clubs = {
        club.value: {
            "g2": build_continuation_schedule(club, Horizon.G2).as_dict(),
            "g3": build_continuation_schedule(club, Horizon.G3).as_dict(),
            "robustness": [c.as_dict() for c in required_robustness_checks(club)],
        }
        for club in (ClubKind.DRIVER, ClubKind.IRON)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 10385,
        "milestone": "MS-111",
        "qualification": QualificationClaim.AWAITING_NATIVE_EVIDENCE.value,
        "claims_native_success": False,
        "depends_on": ["MS-107", "MS-100", "MS-102"],
        "clubs": clubs,
        "notes": (
            "Software contracts for G2/G3 continuation, armature-propagated "
            "independent replay, and fail-closed integrator parity. Native "
            "ControlTower desk success is not claimed."
        ),
    }


@dataclass(frozen=True, slots=True)
class CandidatePromotionRequest:
    """Request to promote a Pinocchio native fitting candidate to qualified status."""

    club: ClubKind
    target_horizon: Horizon
    solver_converged: bool
    solver_cost: float
    solve_integrator: IntegratorConfig
    replay_integrator: IntegratorConfig
    capture_declaration: CaptureAttachmentDeclaration
    physical_audit: Mapping[str, Any]
    metrics: Mapping[str, Any]
    g1_accepted: bool = False
    g1_receipt: Mapping[str, Any] | None = None
    rom_violations: Mapping[str, Any] | None = None
    max_penetration_m: float = 0.010
    max_closure_m: float = 0.005
    max_closure_rad: float = 0.05
    max_normal_force_bw: float = 3.0
    min_weight_fraction: float = 0.20
    max_rom_excess_deg: float = 0.5
    claims_native_success: bool = False


@dataclass(frozen=True, slots=True)
class CandidatePromotionVerdict:
    """Verdict of candidate promotion evaluation."""

    promoted: bool
    rejection_reasons: tuple[str, ...]
    club: ClubKind
    target_horizon: Horizon
    claims_native_success: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "rejection_reasons": list(self.rejection_reasons),
            "club": self.club.value,
            "target_horizon": self.target_horizon.value,
            "claims_native_success": self.claims_native_success,
        }


@dataclass(frozen=True, slots=True)
class TorqueTransferTolerances:
    """Declared numerical tolerances for cross-engine torque transfer."""

    max_acceleration_parity_m_s2: float = 0.05
    max_equilibrium_residual: float = 1e-3
    max_root_residual_n_m: float = 1e-3
    engine_tolerances: Mapping[str, float] | None = None
    engine_equilibrium_tolerances: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class EngineTransferVerdict:
    """Outcome of importing controls/torques into a specific target engine."""

    target_engine: str
    accepted: bool
    acceleration_parity_residual: float
    max_equilibrium_residual: float
    declared_tolerance: float
    diagnostics: dict[str, Any]
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_engine": self.target_engine,
            "accepted": self.accepted,
            "acceleration_parity_residual": self.acceleration_parity_residual,
            "max_equilibrium_residual": self.max_equilibrium_residual,
            "declared_tolerance": self.declared_tolerance,
            "diagnostics": self.diagnostics,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CrossEngineTransferMatrix:
    """Transfer matrix capturing per-engine verdicts and diagnostics."""

    source_engine: str
    club: ClubKind
    target_verdicts: dict[str, EngineTransferVerdict]
    all_accepted: bool
    declared_armature_kg_m2: float
    tracking_gains: dict[str, float]
    residual_root_assistance_included: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_engine": self.source_engine,
            "club": self.club.value,
            "target_verdicts": {
                k: v.as_dict() for k, v in self.target_verdicts.items()
            },
            "all_accepted": self.all_accepted,
            "declared_armature_kg_m2": self.declared_armature_kg_m2,
            "tracking_gains": dict(self.tracking_gains),
            "residual_root_assistance_included": self.residual_root_assistance_included,
        }


def _check_club_and_capture(request: CandidatePromotionRequest) -> list[str]:
    reasons: list[str] = []
    decl = request.capture_declaration
    if request.club == ClubKind.IRON:
        if decl.club != ClubKind.IRON:
            reasons.append(
                "cross-club attachment contamination: iron candidate cannot use driver capture declaration"
            )
        if "driver" in decl.document_id.lower():
            reasons.append(
                f"cross-club document contamination: iron candidate cannot use driver document ({decl.document_id})"
            )
        if "driver" in decl.attachment_calibration_hash.lower():
            reasons.append(
                "cross-club calibration contamination: iron candidate cannot use driver calibration"
            )
    elif request.club == ClubKind.DRIVER:
        if decl.club != ClubKind.DRIVER:
            reasons.append(
                "cross-club attachment contamination: driver candidate cannot use iron capture declaration"
            )
        if "iron" in decl.document_id.lower():
            reasons.append(
                f"cross-club document contamination: driver candidate cannot use iron document ({decl.document_id})"
            )
        if "iron" in decl.attachment_calibration_hash.lower():
            reasons.append(
                "cross-club calibration contamination: driver candidate cannot use iron calibration"
            )
    return reasons


def _check_integrator_and_horizon(request: CandidatePromotionRequest) -> list[str]:
    reasons: list[str] = []
    parity = evaluate_integrator_parity(
        request.solve_integrator, request.replay_integrator
    )
    if not parity.accepted:
        reasons.append(f"integrator parity failed: {parity.reason}")
    if request.target_horizon in (Horizon.G2, Horizon.G3):
        if not request.g1_accepted:
            reasons.append("G1 must pass before G2/G3 promotion")
    return reasons


def _check_physics_and_solver(request: CandidatePromotionRequest) -> list[str]:
    reasons: list[str] = []
    if not request.solver_converged:
        reasons.append("solver convergence is required for promotion")

    audit = request.physical_audit
    bw = audit.get("max_normal_force_body_weights")
    if bw is None and "max_normal_force_n" in audit:
        bw = float(audit["max_normal_force_n"]) / (80.0 * 9.81)
    if bw is not None and float(bw) > request.max_normal_force_bw:
        reasons.append(
            f"max normal force {float(bw):.2f} BW exceeds limit {request.max_normal_force_bw:.2f} BW"
        )

    pen = audit.get("max_penetration_m")
    if pen is not None and float(pen) > request.max_penetration_m:
        reasons.append(
            f"max ground penetration {float(pen) * 1e3:.2f} mm exceeds limit {request.max_penetration_m * 1e3:.2f} mm"
        )

    closure_m = audit.get("closure_translation_error_max_m") or audit.get(
        "max_closure_m"
    )
    if closure_m is not None and float(closure_m) > request.max_closure_m:
        reasons.append(
            f"max weld closure translation {float(closure_m) * 1e3:.2f} mm exceeds limit {request.max_closure_m * 1e3:.2f} mm"
        )

    closure_rad = audit.get("closure_rotation_error_max_rad") or audit.get(
        "max_closure_rad"
    )
    if closure_rad is not None and float(closure_rad) > request.max_closure_rad:
        reasons.append(
            f"max weld closure rotation {float(closure_rad):.4f} rad exceeds limit {request.max_closure_rad:.4f} rad"
        )

    wf_min = audit.get("weight_fraction_min")
    if wf_min is None and isinstance(audit.get("weight_fraction"), Mapping):
        wf_min = audit["weight_fraction"].get("min")
    if wf_min is not None and float(wf_min) < request.min_weight_fraction:
        reasons.append(
            f"min weight fraction {float(wf_min):.2f} below support floor {request.min_weight_fraction:.2f}"
        )

    if request.rom_violations:
        violating = []
        for coord, excess in request.rom_violations.items():
            val = getattr(excess, "max_excess_deg", excess)
            if float(val) > request.max_rom_excess_deg:
                violating.append(coord)
        if violating:
            reasons.append(
                f"range of motion (RoM) violated on coordinates: {sorted(violating)}"
            )
    return reasons


def evaluate_candidate_promotion(
    request: CandidatePromotionRequest,
) -> CandidatePromotionVerdict:
    """Evaluate candidate against Pinocchio native qualification contracts."""
    reasons = (
        _check_club_and_capture(request)
        + _check_integrator_and_horizon(request)
        + _check_physics_and_solver(request)
    )
    return CandidatePromotionVerdict(
        promoted=len(reasons) == 0,
        rejection_reasons=tuple(reasons),
        club=request.club,
        target_horizon=request.target_horizon,
        claims_native_success=False,
    )


def _eval_step_parity_and_residual(
    adapter: Any,
    allocator: Any,
    q_k: NDArray[np.float64],
    v_k: NDArray[np.float64],
    a_k: NDArray[np.float64],
    u_k: NDArray[np.float64],
    delta_root_k: NDArray[np.float64] | None,
) -> tuple[float, float]:
    tau_rnea = adapter.compute_inverse_dynamics(q_k, v_k, a_k)
    j_g = adapter.compute_contact_jacobian(q_k)
    j_w = adapter.compute_grip_jacobian(q_k)
    tau_bounds = (u_k, u_k)
    alloc = allocator.allocator.allocate(
        tau_rnea=tau_rnea,
        j_ground=j_g,
        j_grip=j_w,
        tau_bounds=tau_bounds,
    )
    eq_res = float(alloc.equilibrium_residual)

    tau_full = np.zeros(adapter.nv, dtype=np.float64)
    tau_full[adapter.actuated_indices] = u_k
    root_force = (
        delta_root_k[:6]
        if delta_root_k is not None and np.any(delta_root_k)
        else alloc.delta_tau_root
    )
    root_full = np.concatenate([root_force, np.zeros(adapter.nv - 6)])
    tau_eff = tau_full + j_g.T @ alloc.f_ground + j_w.T @ alloc.lambda_grip + root_full
    parity_err = float(
        adapter.verify_acceleration_parity(
            q=q_k, v=v_k, tau_effective=tau_eff, a_target=a_k
        )
    )
    return parity_err, eq_res


@dataclass(frozen=True)
class _EngineTransferContext:
    target_engine: str
    time_s: NDArray[np.float64]
    q: NDArray[np.float64]
    v: NDArray[np.float64]
    a: NDArray[np.float64]
    controls_u: NDArray[np.float64]
    coordinate_names: tuple[str, ...]
    declared_armature_kg_m2: float
    delta_tau_root: NDArray[np.float64] | None
    tolerances: TorqueTransferTolerances
    adapter: Any | None = None


def _compute_parity_residuals(
    ctx: _EngineTransferContext,
    adapter: Any,
    allocator: Any,
) -> tuple[float, float, float]:
    n_eval = len(ctx.time_s)
    sample_indices = (
        [0, n_eval // 2, n_eval - 1] if n_eval >= 3 else list(range(n_eval))
    )
    parity_residuals: list[float] = []
    max_eq_res = 0.0

    for idx in sample_indices:
        q_k, v_k, a_k = ctx.q[idx], ctx.v[idx], ctx.a[idx]
        u_k = ctx.controls_u[idx][: len(adapter.actuated_indices)]
        delta_k = ctx.delta_tau_root[idx] if ctx.delta_tau_root is not None else None
        p_err, eq_res = _eval_step_parity_and_residual(
            adapter=adapter,
            allocator=allocator,
            q_k=q_k,
            v_k=v_k,
            a_k=a_k,
            u_k=u_k,
            delta_root_k=delta_k,
        )
        parity_residuals.append(p_err)
        max_eq_res = max(max_eq_res, eq_res)

    max_parity = float(np.max(parity_residuals)) if parity_residuals else 0.0
    mean_parity = float(np.mean(parity_residuals)) if parity_residuals else 0.0
    return max_parity, max_eq_res, mean_parity


def _evaluate_single_engine_transfer(
    ctx: _EngineTransferContext,
) -> EngineTransferVerdict:
    tols = ctx.tolerances
    target_engine = ctx.target_engine
    tol = (
        tols.engine_tolerances.get(target_engine, tols.max_acceleration_parity_m_s2)
        if tols.engine_tolerances and target_engine in tols.engine_tolerances
        else tols.max_acceleration_parity_m_s2
    )
    eq_tol = (
        tols.engine_equilibrium_tolerances.get(
            target_engine, tols.max_equilibrium_residual
        )
        if tols.engine_equilibrium_tolerances
        and target_engine in tols.engine_equilibrium_tolerances
        else tols.max_equilibrium_residual
    )
    try:
        adapter = ctx.adapter
        if adapter is None:
            from src.shared.python.motion_matching.multi_engine_torque_allocator import (
                create_engine_force_adapter,
            )

            adapter = create_engine_force_adapter(
                target_engine, allow_synthetic=True, nv=len(ctx.coordinate_names)
            )

        if getattr(adapter, "nv", len(ctx.coordinate_names)) != len(
            ctx.coordinate_names
        ):
            return EngineTransferVerdict(
                target_engine=target_engine,
                accepted=False,
                acceleration_parity_residual=float("inf"),
                max_equilibrium_residual=float("inf"),
                declared_tolerance=tol,
                diagnostics={"error": "nv mismatch"},
                reason=f"coordinate count mismatch ({adapter.nv} != {len(ctx.coordinate_names)})",
            )

        from src.shared.python.motion_matching.multi_engine_torque_allocator import (
            MultiEngineTorqueAllocator,
        )

        allocator = MultiEngineTorqueAllocator(adapter)
        max_p, max_eq, mean_p = _compute_parity_residuals(ctx, adapter, allocator)

        reasons_failed: list[str] = []
        if max_p > tol:
            reasons_failed.append(
                f"acceleration parity error {max_p:.6f} exceeds declared tolerance {tol:.6f}"
            )
        if max_eq > eq_tol:
            reasons_failed.append(
                f"equilibrium residual {max_eq:.6f} exceeds declared tolerance {eq_tol:.6f}"
            )

        return EngineTransferVerdict(
            target_engine=target_engine,
            accepted=len(reasons_failed) == 0,
            acceleration_parity_residual=max_p,
            max_equilibrium_residual=max_eq,
            declared_tolerance=tol,
            diagnostics={
                "mean_parity_residual": mean_p,
                "max_parity_residual": max_p,
                "max_equilibrium_residual": max_eq,
                "armature_kg_m2": ctx.declared_armature_kg_m2,
            },
            reason="; ".join(reasons_failed),
        )
    except Exception as exc:
        return EngineTransferVerdict(
            target_engine=target_engine,
            accepted=False,
            acceleration_parity_residual=float("inf"),
            max_equilibrium_residual=float("inf"),
            declared_tolerance=tol,
            diagnostics={"error": str(exc)},
            reason=f"target engine evaluation failed: {exc}",
        )


def evaluate_cross_engine_torque_transfer(
    *args: Any,
    **kwargs: Any,
) -> CrossEngineTransferMatrix:
    """Evaluate controls transferred into MuJoCo, Simscape, and other target engines."""
    source_engine: str = kwargs["source_engine"]
    club: ClubKind = kwargs["club"]
    target_engines: tuple[str, ...] = tuple(kwargs["target_engines"])
    time_s: NDArray[np.float64] = kwargs["time_s"]
    q: NDArray[np.float64] = kwargs["q"]
    v: NDArray[np.float64] = kwargs["v"]
    a: NDArray[np.float64] = kwargs["a"]
    controls_u: NDArray[np.float64] = kwargs["controls_u"]
    coordinate_names: tuple[str, ...] = tuple(kwargs["coordinate_names"])
    declared_armature: float = kwargs.get(
        "declared_armature_kg_m2", DEFAULT_ARMATURE_KG_M2
    )
    delta_root: NDArray[np.float64] | None = kwargs.get("delta_tau_root")
    gains: dict[str, float] | None = kwargs.get("tracking_gains")
    tol: TorqueTransferTolerances = (
        kwargs.get("tolerances") or TorqueTransferTolerances()
    )
    custom_adapters: Mapping[str, Any] | None = kwargs.get("custom_adapters")

    verdicts: dict[str, EngineTransferVerdict] = {}
    for engine in target_engines:
        ctx = _EngineTransferContext(
            target_engine=engine,
            time_s=time_s,
            q=q,
            v=v,
            a=a,
            controls_u=controls_u,
            coordinate_names=coordinate_names,
            declared_armature_kg_m2=declared_armature,
            delta_tau_root=delta_root,
            tolerances=tol,
            adapter=custom_adapters.get(engine) if custom_adapters else None,
        )
        verdicts[engine] = _evaluate_single_engine_transfer(ctx)

    all_accepted = len(verdicts) > 0 and all(v.accepted for v in verdicts.values())
    return CrossEngineTransferMatrix(
        source_engine=source_engine,
        club=club,
        target_verdicts=verdicts,
        all_accepted=all_accepted,
        declared_armature_kg_m2=declared_armature,
        tracking_gains=gains or {},
        residual_root_assistance_included=delta_root is not None,
    )
