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
from typing import Any, Mapping

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
