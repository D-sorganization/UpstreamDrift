"""Per-model neural dataset/checkpoint coverage matrix (NM-00 #10615)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.core.contracts import postcondition, precondition
from src.shared.python.tour_baselines.models import EvidenceStatus
from src.shared.python.tour_baselines.registry import list_golf_models

from .audit import audit_neural_artifacts
from .types import ClaimStatus, Disposition


@dataclass(frozen=True)
class NeuralCoverageCell:
    """Coverage status for one golf model against neural corpora/checkpoints."""

    model_id: str
    dataset_id: str | None
    checkpoint_id: str | None
    disposition: Disposition
    claim_status: ClaimStatus
    evidence_status: EvidenceStatus
    governing_issue: str
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "checkpoint_id": self.checkpoint_id,
            "disposition": self.disposition.value,
            "claim_status": self.claim_status.value,
            "evidence_status": self.evidence_status.value,
            "governing_issue": self.governing_issue,
            "blockers": list(self.blockers),
        }


@precondition(
    lambda repo_root: Path(repo_root).is_dir(),
    "repo_root must be an existing directory",
)
@postcondition(
    lambda result: all(
        cell.claim_status != ClaimStatus.NATIVE_QUALIFIED for cell in result
    ),
    "coverage cells cannot claim native qualification from NM-00 alone",
)
def generate_neural_coverage_matrix(
    repo_root: Path | str,
) -> list[NeuralCoverageCell]:
    """Build one coverage cell per registered GolfModelIdentity.

    Links the TB-00 model roster to the NM-00 artifact inventory. Until a
    model-specific corpus and checkpoint exist with complete provenance,
    every cell is quarantined / unavailable — never silently supported.
    """
    receipt = audit_neural_artifacts(repo_root)
    synthetic = None
    ten_k = None
    ckpt = None
    try:
        synthetic = receipt.artifact("fixture.sweep_synthetic")
    except KeyError:
        pass
    try:
        ten_k = receipt.artifact("corpus.ten_thousand_files")
    except KeyError:
        pass
    try:
        ckpt = receipt.artifact("checkpoint.surrogate_production_default")
    except KeyError:
        try:
            ckpt = receipt.artifact("checkpoint.surrogate_best_default")
        except KeyError:
            pass

    cells: list[NeuralCoverageCell] = []
    for model in list_golf_models():
        blockers: list[str] = [
            "no model-specific native neural corpus qualified under NM-00",
        ]
        dataset_id: str | None = None
        if ten_k is not None and not ten_k.exists:
            blockers.append(f"documented compact corpus absent ({ten_k.path})")
        if synthetic is not None and synthetic.exists:
            dataset_id = synthetic.artifact_id
            blockers.append(
                "only synthetic sweep fixture available; software contracts only"
            )

        checkpoint_id = ckpt.artifact_id if ckpt is not None else None
        if ckpt is None or not ckpt.exists:
            blockers.append("no surrogate checkpoint present at default paths")
            checkpoint_id = ckpt.artifact_id if ckpt is not None else None

        cells.append(
            NeuralCoverageCell(
                model_id=model.model_id,
                dataset_id=dataset_id,
                checkpoint_id=checkpoint_id,
                disposition=Disposition.QUARANTINE,
                claim_status=ClaimStatus.UNSUPPORTED,
                evidence_status=EvidenceStatus.UNAVAILABLE,
                governing_issue="#10615",
                blockers=tuple(blockers),
            )
        )
    return cells
