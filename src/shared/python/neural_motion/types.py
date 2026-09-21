"""Typed contracts for neural artifact inventory and claim status (NM-00 #10615)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class ArtifactKind(str, Enum):
    """Kind of inventoried artifact."""

    DATASET = "dataset"
    CHECKPOINT = "checkpoint"
    TRAINING_CLAIM = "training_claim"
    FIXTURE = "fixture"
    REMOTE_POINTER = "remote_pointer"


class ArtifactRole(str, Enum):
    """Learning / supervision role for an artifact."""

    FORWARD_SURROGATE = "forward_surrogate"
    INVERSE_CVAE = "inverse_cvae"
    INVERSE_REGRESSOR = "inverse_regressor"
    INVERSE_TIMESTEP = "inverse_timestep"
    HYBRID_WARMSTART = "hybrid_warmstart"
    COMPACT_CORPUS = "compact_corpus"
    SWEEP_CORPUS = "sweep_corpus"
    SIMSCAPE_ML = "simscape_ml"
    HISTORICAL_ISSUE = "historical_issue"


class Disposition(str, Enum):
    """Retain / repair / migrate / reject / quarantine decision."""

    RETAIN = "retain"
    REPAIR = "repair"
    MIGRATE = "migrate"
    REJECT = "reject"
    QUARANTINE = "quarantine"


class ClaimStatus(str, Enum):
    """How far a training or speed claim is currently evidenced."""

    SOFTWARE_CONTRACT_ONLY = "software_contract_only"
    NOTE_ONLY = "note_only"
    REPRODUCED_LOCALLY = "reproduced_locally"
    NATIVE_QUALIFIED = "native_qualified"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ArtifactIdentity:
    """Immutable identity row for one dataset, checkpoint, fixture, or claim.

    Design by Contract:
    - ``artifact_id`` is non-empty.
    - Absent artifacts have ``exists is False`` and ``content_sha256 is None``.
    - Synthetic fixtures never carry ``ClaimStatus.NATIVE_QUALIFIED``.
    - ``NATIVE_QUALIFIED`` requires ``exists``, non-synthetic, and complete
      provenance fields (model/engine/source_revision).
    """

    artifact_id: str
    kind: ArtifactKind
    role: ArtifactRole
    path: Path | None
    exists: bool
    content_sha256: str | None
    schema_version: str | None
    model_ids: tuple[str, ...]
    engine: str | None
    source_revision: str | None
    units: str | None
    seeds: tuple[int, ...]
    trial_ancestry: str | None
    required_channels: tuple[str, ...]
    label_availability: Mapping[str, bool]
    is_synthetic_fixture: bool
    disposition: Disposition
    claim_status: ClaimStatus
    blockers: tuple[str, ...]
    retrieval_instructions: str
    notes: str = ""
    related_issues: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("artifact_id must be non-empty")
        if self.exists and self.path is None:
            raise ValueError(
                f"{self.artifact_id}: exists=True requires a concrete path"
            )
        if not self.exists and self.content_sha256 is not None:
            raise ValueError(
                f"{self.artifact_id}: absent artifacts cannot carry a content hash"
            )
        if (
            self.is_synthetic_fixture
            and self.claim_status == ClaimStatus.NATIVE_QUALIFIED
        ):
            raise ValueError(
                f"{self.artifact_id}: synthetic fixtures cannot certify native"
            )
        if self.claim_status == ClaimStatus.NATIVE_QUALIFIED:
            missing = []
            if not self.exists:
                missing.append("exists")
            if self.is_synthetic_fixture:
                missing.append("non-synthetic")
            if not self.engine:
                missing.append("engine")
            if not self.source_revision:
                missing.append("source_revision")
            if not self.model_ids:
                missing.append("model_ids")
            if missing:
                raise ValueError(
                    f"{self.artifact_id}: NATIVE_QUALIFIED missing {', '.join(missing)}"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "role": self.role.value,
            "path": str(self.path) if self.path is not None else None,
            "exists": self.exists,
            "content_sha256": self.content_sha256,
            "schema_version": self.schema_version,
            "model_ids": list(self.model_ids),
            "engine": self.engine,
            "source_revision": self.source_revision,
            "units": self.units,
            "seeds": list(self.seeds),
            "trial_ancestry": self.trial_ancestry,
            "required_channels": list(self.required_channels),
            "label_availability": dict(self.label_availability),
            "is_synthetic_fixture": self.is_synthetic_fixture,
            "disposition": self.disposition.value,
            "claim_status": self.claim_status.value,
            "blockers": list(self.blockers),
            "retrieval_instructions": self.retrieval_instructions,
            "notes": self.notes,
            "related_issues": list(self.related_issues),
        }
