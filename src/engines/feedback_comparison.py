"""F01 admission contracts for controlled-swing evidence (issue #11785).

This is a view of :mod:`model_inventory`, not another engine registry. A
smoke-ready package is never promoted to replay or physiology qualification.
Actual integration, scoring, and certification belong to their existing owners.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from src.engines.model_inventory import (
    EngineModelInventory,
    ModelPackage,
    PackageStatus,
    sha256_file,
)

REPLAY_COMPARISON_CONTRACT_VERSION = "feedback-comparison/1.1.0"


class DriveMode(str, enum.Enum):
    TORQUE = "torque"
    MUSCLE_EXCITATION = "muscle_excitation"


class InputKind(str, enum.Enum):
    ACTUATOR_COMMAND = "actuator_command"
    ACTUATOR_TORQUE = "actuator_torque"
    ACTUATOR_FORCE = "actuator_force"
    GENERALIZED_EFFORT = "generalized_effort"
    MUSCLE_EXCITATION = "muscle_excitation"
    MUSCLE_ACTIVATION = "muscle_activation"
    EXTERNAL_LOAD = "external_load"


class ReplayPolicy(str, enum.Enum):
    INDEPENDENT_TIME_ONLY = "independent_time_only"
    STATE_FEEDBACK = "state_feedback"


class EvidenceMode(str, enum.Enum):
    SHARED_RIGID_BODY_EMULATION = "shared_rigid_body_emulation"
    NATIVE_OWN_CONTACT = "native_own_contact"
    EXTERNALLY_FORCED = "externally_forced"


class ComparisonLevel(str, enum.Enum):
    IDENTITY = "identity"
    TRANSCRIPTION_FEASIBILITY = "transcription_feasibility"
    WITHIN_ENGINE_REPLAY = "within_engine_replay"
    SAME_INPUT = "same_input"
    OBSERVATION_ACCURACY = "observation_accuracy"
    BIOMECHANICAL_EQUIVALENCE = "biomechanical_equivalence"


@dataclass(frozen=True)
class ComparisonRow:
    """Stable package/variant/drive key and separate program/evidence axes."""

    package_id: str
    variant_id: str
    drive_mode: DriveMode
    engine: str
    source_model_sha256: str
    provider_id: str
    provider_sha256: str
    required: bool
    support: str
    availability: str
    qualification: str
    scope_note: str
    required_capabilities: tuple[str, ...]
    capability_support: Mapping[str, str]


@dataclass(frozen=True)
class ComparisonEvidence:
    """A receipt's identity and replay facts; no score or success claim."""

    package_id: str
    variant_id: str
    drive_mode: DriveMode
    source_model_sha256: str
    provider_id: str
    provider_sha256: str
    state_schema_sha256: str
    policy_sha256: str
    applied_input_sha256: str
    input_kind: InputKind
    input_interpolation: str
    timebase_id: str
    replay_policy: ReplayPolicy
    evidence_mode: EvidenceMode
    horizon_s: float
    observation_sha256: str
    channel_ids: tuple[str, ...]
    full_state: bool
    full_horizon: bool
    state_resets: int
    nq: int
    nv: int
    bundle_schema: str = "experiment-replay/1.0.0"
    physical_model_sha256: str = ""
    loaded_native_model_sha256: str = ""
    time_grid_sha256: str = ""
    physics_sha256: str = ""
    contact_sha256: str = ""
    integrator_sha256: str = ""
    input_channel_schema_sha256: str = ""
    contact_evidence_sha256: str = ""
    force_evidence_sha256: str = ""
    muscle_state_evidence_sha256: str = ""
    transcription_receipt_sha256: str = ""
    observation_score_receipt_sha256: str = ""
    observation_time_grid_sha256: str = ""
    initial_state_sha256: str = ""
    comparison_contract_version: str = ""


@dataclass(frozen=True)
class ComparisonRequest:
    level: ComparisonLevel
    left: ComparisonEvidence
    right: ComparisonEvidence


@dataclass(frozen=True)
class BaselineRun:
    name: str
    observation_sha256: str
    horizon_s: float
    compute_budget_s: float


class FeedbackComparisonRegistry:
    """Fail-closed admission view over the authoritative package inventory."""

    required_baselines = frozenset(
        {"independent_polynomial", "computed_torque", "mosaic_tvlqr", "selected_ocp"}
    )
    required_capabilities = (
        "torque_drive",
        "muscle_excitation",
        "contact",
        "grip",
        "flexible_club",
        "full_state",
        "native_own_contact",
        "independent_replay",
    )

    def __init__(self, inventory: EngineModelInventory) -> None:
        self.inventory = inventory
        self.rows = tuple(
            row
            for p in inventory.packages
            for row in self._rows_for(p, inventory.repo_root)
        )
        keys = [(r.package_id, r.variant_id, r.drive_mode) for r in self.rows]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate comparison row")

    @staticmethod
    def _provider_digest(provider_id: str, root: Path) -> str:
        source = provider_id.split(":", 1)[0]
        candidate = root / (
            source.replace(".", "/") + ".py"
            if not source.endswith((".py", ".m", ".xml", ".osim")) and "/" not in source
            else source
        )
        return sha256_file(candidate) if candidate.is_file() else ""

    @staticmethod
    def _rows_for(package: ModelPackage, root: Path) -> Iterable[ComparisonRow]:
        declared = package.matching_variants or (
            {
                "id": "default",
                "drive_mode": "muscle_excitation"
                if (package.actuation or {}).get("type") == "muscle_activation"
                else "torque",
                "support": "supported" if package.actuation else "unknown",
                "scope_note": "Package metadata only; replay and physiology unqualified.",
            },
        )
        for variant in declared:
            mode = DriveMode(str(variant["drive_mode"]))
            support_state = str(variant.get("support", "unknown"))
            native_contact_support = str(
                variant.get("native_own_contact_support", "unknown")
            )
            if support_state not in {"supported", "unsupported", "unknown"}:
                raise ValueError("invalid variant support state")
            if native_contact_support not in {"supported", "unsupported", "unknown"}:
                raise ValueError("invalid native-contact support state")
            support = dict.fromkeys(
                FeedbackComparisonRegistry.required_capabilities, "unknown"
            )
            support[
                "torque_drive" if mode == DriveMode.TORQUE else "muscle_excitation"
            ] = support_state
            support["contact"] = "supported" if package.contact_config else "unknown"
            support["grip"] = "supported" if package.grip_config else "unknown"
            support["native_own_contact"] = native_contact_support
            provider_id = str(
                variant.get("factory")
                or package.generator
                or package.generated_model
                or package.source_spec
                or ""
            )
            yield ComparisonRow(
                package_id=package.id,
                variant_id=str(variant["id"]),
                drive_mode=mode,
                engine=package.engine,
                source_model_sha256=package.identity_hash(),
                provider_id=provider_id,
                provider_sha256=FeedbackComparisonRegistry._provider_digest(
                    provider_id, root
                ),
                required=True,
                support=support_state,
                availability="unavailable"
                if package.status in {PackageStatus.REPAIR, PackageStatus.RETIRED}
                else "unknown",
                qualification="unqualified",
                scope_note=str(variant.get("scope_note", "")),
                required_capabilities=FeedbackComparisonRegistry.required_capabilities,
                capability_support=support,
            )

    def get(
        self, package_id: str, variant_id: str, drive_mode: DriveMode
    ) -> ComparisonRow:
        for row in self.rows:
            if (row.package_id, row.variant_id, row.drive_mode) == (
                package_id,
                variant_id,
                drive_mode,
            ):
                return row
        raise ValueError(
            f"unknown comparison row: {package_id}/{variant_id}/{drive_mode}"
        )

    @staticmethod
    def _digest(value: str, name: str) -> None:
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
            raise ValueError(f"{name} must be a SHA-256 hex digest")

    def _admit_replay_evidence(self, evidence: ComparisonEvidence) -> None:
        """Validate independent applied-input replay without scoring accuracy."""
        if evidence.comparison_contract_version != REPLAY_COMPARISON_CONTRACT_VERSION:
            raise ValueError("replay comparison contract version is unsupported")
        self._digest(evidence.initial_state_sha256, "initial_state_sha256")
        self._digest(evidence.applied_input_sha256, "applied_input_sha256")
        for name in (
            "physical_model_sha256",
            "time_grid_sha256",
            "physics_sha256",
            "contact_sha256",
            "integrator_sha256",
            "input_channel_schema_sha256",
        ):
            self._digest(getattr(evidence, name), name)
        if evidence.evidence_mode == EvidenceMode.NATIVE_OWN_CONTACT:
            self._digest(
                evidence.loaded_native_model_sha256, "loaded_native_model_sha256"
            )
        if evidence.replay_policy != ReplayPolicy.INDEPENDENT_TIME_ONLY:
            raise ValueError("replay must use an independent time-only input player")
        if evidence.state_resets or not evidence.full_state:
            raise ValueError("replay must preserve full physical state without resets")
        if not evidence.full_horizon:
            raise ValueError("replay must cover the full horizon")
        if not math.isfinite(evidence.horizon_s) or evidence.horizon_s <= 0:
            raise ValueError("replay horizon must be positive and finite")
        if evidence.nq <= 0 or evidence.nv <= 0 or not evidence.channel_ids:
            raise ValueError("replay requires dimensions and channel identities")
        if evidence.bundle_schema == "same-input-bundle/v1":
            if evidence.nq != evidence.nv:
                raise ValueError("Euclidean v1 bundle cannot represent nq != nv")
            if evidence.input_kind == InputKind.MUSCLE_EXCITATION:
                raise ValueError("v1 bundle cannot represent muscle excitation")
            raise ValueError("v1 lacks binding of execution policy and applied inputs")
        if evidence.bundle_schema != "experiment-replay/1.0.0":
            raise ValueError("unrecognized replay bundle schema")
        if evidence.timebase_id != "simulation_relative":
            raise ValueError("replay timebase must be simulation_relative")
        if (
            evidence.input_kind
            in {
                InputKind.ACTUATOR_COMMAND,
                InputKind.ACTUATOR_TORQUE,
                InputKind.ACTUATOR_FORCE,
                InputKind.GENERALIZED_EFFORT,
            }
            and evidence.input_interpolation != "zero_order_hold"
        ):
            raise ValueError("direct effort replay requires exact zero-order hold")
        if (
            evidence.input_kind == InputKind.MUSCLE_EXCITATION
            and evidence.input_interpolation
            not in {
                "zero_order_hold",
                "linear",
            }
        ):
            raise ValueError("muscle excitation interpolation is undeclared")
        if (
            evidence.drive_mode == DriveMode.MUSCLE_EXCITATION
            and evidence.input_kind != InputKind.MUSCLE_EXCITATION
        ):
            raise ValueError("muscle drive requires excitation, not effort")
        if evidence.drive_mode == DriveMode.TORQUE and evidence.input_kind not in {
            InputKind.ACTUATOR_COMMAND,
            InputKind.ACTUATOR_TORQUE,
            InputKind.ACTUATOR_FORCE,
            InputKind.GENERALIZED_EFFORT,
        }:
            raise ValueError("torque drive requires an actuator or effort input")

    def _admit_biomechanical_evidence(
        self,
        evidence: ComparisonEvidence,
        row: ComparisonRow,
        claim_native_muscle: bool,
    ) -> None:
        if evidence.evidence_mode == EvidenceMode.EXTERNALLY_FORCED:
            raise ValueError(
                "externally forced replay cannot establish biomechanical equivalence"
            )
        if not evidence.full_horizon or evidence.state_resets:
            raise ValueError(
                "biomechanical evidence requires uninterrupted full horizon"
            )
        self._digest(evidence.contact_evidence_sha256, "contact_evidence_sha256")
        self._digest(evidence.force_evidence_sha256, "force_evidence_sha256")
        if claim_native_muscle:
            if row.capability_support["native_own_contact"] != "supported":
                raise ValueError("native own-contact capability is not supported")
            if evidence.drive_mode != DriveMode.MUSCLE_EXCITATION:
                raise ValueError("native muscle claim requires excitation drive")
            if evidence.evidence_mode != EvidenceMode.NATIVE_OWN_CONTACT:
                raise ValueError(
                    "native muscle claim requires native own-contact physics"
                )
            self._digest(
                evidence.muscle_state_evidence_sha256, "muscle_state_evidence_sha256"
            )

    def admit(
        self,
        evidence: ComparisonEvidence,
        level: ComparisonLevel,
        *,
        claim_qualified: bool = False,
        claim_native_muscle: bool = False,
    ) -> ComparisonRow:
        """Check prerequisites for a claim, never certify a numerical gate."""
        row = self.get(evidence.package_id, evidence.variant_id, evidence.drive_mode)
        if (
            evidence.source_model_sha256 != row.source_model_sha256
            or not row.source_model_sha256
        ):
            raise ValueError("stale or missing model identity")
        if evidence.provider_id != row.provider_id or not row.provider_id:
            raise ValueError("stale or missing provider identity")
        if evidence.provider_sha256 != row.provider_sha256 or not row.provider_sha256:
            raise ValueError("stale or missing provider source hash")
        for name in (
            "provider_sha256",
            "state_schema_sha256",
            "policy_sha256",
            "observation_sha256",
        ):
            self._digest(getattr(evidence, name), name)
        if claim_qualified and row.qualification != "qualified":
            raise ValueError("unqualified row cannot be claimed qualified")
        if level == ComparisonLevel.IDENTITY:
            return row
        if level == ComparisonLevel.TRANSCRIPTION_FEASIBILITY:
            self._digest(
                evidence.transcription_receipt_sha256, "transcription_receipt_sha256"
            )
            return row
        self._admit_replay_evidence(evidence)
        if level == ComparisonLevel.OBSERVATION_ACCURACY:
            self._digest(
                evidence.observation_time_grid_sha256,
                "observation_time_grid_sha256",
            )
            self._digest(
                evidence.observation_score_receipt_sha256,
                "observation_score_receipt_sha256",
            )
        if level == ComparisonLevel.BIOMECHANICAL_EQUIVALENCE or claim_native_muscle:
            self._admit_biomechanical_evidence(evidence, row, claim_native_muscle)
        return row

    def compare(self, request: ComparisonRequest) -> None:
        """Check common-input comparability without computing a score."""
        if request.left.input_kind != request.right.input_kind:
            raise ValueError(
                "input kind differs; torque and excitation are incomparable"
            )
        self.admit(request.left, request.level)
        self.admit(request.right, request.level)
        if request.level == ComparisonLevel.SAME_INPUT:
            fields = (
                "physical_model_sha256",
                "time_grid_sha256",
                "physics_sha256",
                "contact_sha256",
                "integrator_sha256",
                "input_channel_schema_sha256",
                "applied_input_sha256",
                "policy_sha256",
                "state_schema_sha256",
                "initial_state_sha256",
                "observation_sha256",
                "channel_ids",
                "timebase_id",
                "input_interpolation",
                "horizon_s",
            )
            for name in fields:
                if getattr(request.left, name) != getattr(request.right, name):
                    raise ValueError(f"same-input comparison differs in {name}")
            if request.left.evidence_mode != request.right.evidence_mode:
                raise ValueError("same-input physics evidence mode differs")

    def validate_baselines(self, runs: tuple[BaselineRun, ...]) -> None:
        """Require a fair prospective benchmark; actual metrics live elsewhere."""
        names = {run.name for run in runs}
        if names != self.required_baselines or len(runs) != len(names):
            raise ValueError(
                f"missing baseline: {sorted(self.required_baselines - names)}"
            )
        if len({run.observation_sha256 for run in runs}) != 1:
            raise ValueError("baseline observation sets differ")
        if len({run.horizon_s for run in runs}) != 1:
            raise ValueError("baseline horizons differ")
        if len({run.compute_budget_s for run in runs}) != 1:
            raise ValueError("baseline budget differs")
        for run in runs:
            self._digest(run.observation_sha256, "observation_sha256")
            if not math.isfinite(run.horizon_s) or run.horizon_s <= 0:
                raise ValueError("baseline horizon invalid")
            if not math.isfinite(run.compute_budget_s) or run.compute_budget_s <= 0:
                raise ValueError("baseline budget invalid")
