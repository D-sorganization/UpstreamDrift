"""Checkpoint IO for masked proposal models (NM-06)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import MaskedProposalModel
from .types import PROPOSAL_SCHEMA

__all__ = [
    "ProposalCheckpointCard",
    "load_proposal_checkpoint",
    "save_proposal_checkpoint",
]


@dataclass(frozen=True, slots=True)
class ProposalCheckpointCard:
    schema: str
    model_id: str
    u_dim: int
    control_basis: str
    weight_digest: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_id": self.model_id,
            "u_dim": self.u_dim,
            "control_basis": self.control_basis,
            "weight_digest": self.weight_digest,
        }


def _digest_payload(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def assert_checkpoint_compatible(
    *,
    expected_model_id: str,
    expected_u_dim: int,
    expected_control_basis: str,
    card: ProposalCheckpointCard,
) -> None:
    if card.schema != PROPOSAL_SCHEMA:
        raise ValueError(
            "incompatible proposal checkpoint schema: "
            f"expected {PROPOSAL_SCHEMA!r}, got {card.schema!r}"
        )
    if card.model_id != expected_model_id:
        raise ValueError(
            "incompatible proposal checkpoint model_id: "
            f"expected {expected_model_id!r}, got {card.model_id!r}"
        )
    if card.u_dim != int(expected_u_dim):
        raise ValueError(
            "incompatible proposal checkpoint u_dim: "
            f"expected {expected_u_dim}, got {card.u_dim}"
        )
    if card.control_basis != expected_control_basis:
        raise ValueError(
            "incompatible proposal checkpoint control_basis: "
            f"expected {expected_control_basis!r}, got {card.control_basis!r}"
        )


def save_proposal_checkpoint(path: str | Path, model: MaskedProposalModel) -> Path:
    if not isinstance(model, MaskedProposalModel):
        raise TypeError("model must be a MaskedProposalModel")
    payload = model.state_payload()
    digest = _digest_payload(payload)
    payload["checkpoint_card"] = {
        "schema": PROPOSAL_SCHEMA,
        "model_id": model.config.model_id,
        "u_dim": model.config.u_dim,
        "control_basis": model.config.control_basis,
        "weight_digest": digest,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return target


def load_proposal_checkpoint(
    path: str | Path,
    *,
    model_id: str,
    u_dim: int,
    control_basis: str,
) -> tuple[MaskedProposalModel, ProposalCheckpointCard]:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    card_raw = payload.get("checkpoint_card")
    if not isinstance(card_raw, dict):
        raise ValueError("checkpoint_card missing from payload")
    card = ProposalCheckpointCard(
        schema=str(card_raw.get("schema", "")),
        model_id=str(card_raw.get("model_id", "")),
        u_dim=int(card_raw.get("u_dim", 0)),
        control_basis=str(card_raw.get("control_basis", "")),
        weight_digest=str(card_raw.get("weight_digest", "")),
    )
    assert_checkpoint_compatible(
        expected_model_id=model_id,
        expected_u_dim=int(u_dim),
        expected_control_basis=control_basis,
        card=card,
    )
    model = MaskedProposalModel.from_state_payload(payload)
    return model, card
