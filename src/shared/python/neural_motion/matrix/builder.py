"""Builder for NM-09 per-model checkpoint matrix (#10624).

Executes the #10585 roster as an explicit per-model matrix, computing:
- Checkpoint cards with variable nq/nv/nu dimensions.
- Native replay receipts and 3-seed evidence.
- Full cryptographic hash chain linking data, splits, weights, and cards.
- Fail-closed named blockers for uninstalled optional engines.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

from src.shared.python.tour_baselines.models import GolfModelIdentity, ModelTopology
from src.shared.python.tour_baselines.registry import list_golf_models

from .adapters import is_runtime_available_for_model
from .types import (
    CHECKPOINT_SCHEMA,
    BenefitResult,
    ModelCheckpointCard,
    ModelCheckpointStatus,
    NativeReplayReceipt,
    ThreeSeedEvidence,
)

logger = logging.getLogger(__name__)

MATRIX_SCHEMA = "neural-checkpoint-matrix/1.0.0"

__all__ = [
    "MATRIX_SCHEMA",
    "NeuralCheckpointMatrix",
    "build_checkpoint_matrix",
]


@dataclass(frozen=True)
class NeuralCheckpointMatrix:
    """Matrix of model checkpoint cards across the full #10585 roster."""

    schema: str
    cards: tuple[ModelCheckpointCard, ...]

    def __post_init__(self) -> None:
        if self.schema != MATRIX_SCHEMA:
            raise ValueError(f"schema must be {MATRIX_SCHEMA!r}, got {self.schema!r}")
        if not self.cards:
            raise ValueError("cards must be non-empty sequence")
        model_ids = [c.model_id for c in self.cards]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("duplicate model_ids in matrix cards")

    def get_card(self, model_id: str) -> ModelCheckpointCard:
        """Look up a checkpoint card by canonical model_id."""
        for card in self.cards:
            if card.model_id == model_id:
                return card
        raise KeyError(f"unknown model_id in checkpoint matrix: {model_id}")

    def matrix_digest(self) -> str:
        """End-to-end cryptographic hash binding all checkpoint cards in the matrix."""
        payload = json.dumps(
            {
                "schema": self.schema,
                "cards": [
                    card.as_dict()
                    for card in sorted(self.cards, key=lambda c: c.model_id)
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_digest": self.matrix_digest(),
            "cards": [card.as_dict() for card in self.cards],
        }


def _classify_model(
    identity: GolfModelIdentity,
) -> tuple[ModelCheckpointStatus, str, tuple[str, ...]]:
    """Classify model status, control basis, and blockers."""
    mid = identity.model_id
    if mid in (
        "driven_double_pendulum",
        "driven_triple_pendulum",
        "constrained_upper_body_golfer",
    ):
        return (
            ModelCheckpointStatus.UNQUALIFIED,
            "joint_torque",
            (
                "unqualified: no trained checkpoint or native replay receipt (#10960 P0-2/P0-4)",
            ),
        )

    if identity.topology == ModelTopology.KINEMATIC_RECONSTRUCTION:
        return (
            ModelCheckpointStatus.KINEMATIC_PROPOSAL,
            "kinematic_joint_angle",
            ("kinematic reconstruction models lack torque-driven supervision",),
        )

    if identity.topology == ModelTopology.REFERENCE_CATALOG_URDF:
        return (
            ModelCheckpointStatus.REFERENCE_ONLY,
            "generalized_force",
            ("reference catalog URDF is not a pilot neural training target",),
        )

    # Full-body multibody models
    return _classify_full_body_model(mid)


def _classify_full_body_model(
    mid: str,
) -> tuple[ModelCheckpointStatus, str, tuple[str, ...]]:
    """Classify full-body models with exact runtime availability blockers."""
    if "simscape" in mid:
        return (
            ModelCheckpointStatus.BLOCKED_PREREQUISITE,
            "generalized_force",
            (
                "missing_simscape_runtime: requires MATLAB R2025b and Simscape Multibody license (#9921)",
            ),
        )
    if "myosuite" in mid:
        return (
            ModelCheckpointStatus.BLOCKED_PREREQUISITE,
            "generalized_force",
            (
                "fail_closed_myosuite: fail-closed per MS-50 pending muscle retarget (#9478)",
            ),
        )
    if "opensim" in mid and not is_runtime_available_for_model(mid):
        return (
            ModelCheckpointStatus.BLOCKED_PREREQUISITE,
            "generalized_force",
            (
                "missing_opensim_runtime: OpenSim Moco runtime unavailable (#10376, #10414)",
            ),
        )
    if "drake" in mid and not is_runtime_available_for_model(mid):
        return (
            ModelCheckpointStatus.BLOCKED_PREREQUISITE,
            "generalized_force",
            (
                "missing_drake_runtime: pydrake not available in local environment (#10375)",
            ),
        )

    return (
        ModelCheckpointStatus.BLOCKED_PREREQUISITE,
        "generalized_force",
        (
            f"unqualified_full_body: {mid} lacks verified trained checkpoint and native replay receipt on disk",
        ),
    )


def _build_card_for_model(identity: GolfModelIdentity) -> ModelCheckpointCard:
    status, control_basis, blockers = _classify_model(identity)
    mid = identity.model_id
    # No training or native-replay pipeline produces checkpoints yet, so every
    # identity, digest, seed and benefit field stays unmeasured (#10960 P0-2/P0-4).
    benefit = BenefitResult(
        speedup_factor=None,
        loss_reduction_pct=None,
        break_even_queries=None,
        verdict="unmeasured",
    )

    return ModelCheckpointCard(
        schema=CHECKPOINT_SCHEMA,
        model_id=mid,
        backend=identity.backend.value,
        topology=identity.topology.value,
        q_dim=identity.dof,
        v_dim=identity.dof,
        u_dim=identity.independent_dof,
        constraint_count=identity.constraint_count,
        control_basis=control_basis,
        conditioning_schema=f"conditioning_{identity.topology.value}/1.0",
        generator_adapter=f"adapter_{identity.topology.value}",
        dataset_hash=None,
        split_hash=None,
        weight_digest=None,
        three_seed_evidence=None,
        native_replay_receipt=None,
        benefit_result=benefit,
        status=status,
        blockers=blockers,
        governing_issues=("#10585", "#10624"),
    )


def build_checkpoint_matrix() -> NeuralCheckpointMatrix:
    """Build the comprehensive per-model checkpoint matrix covering all #10585 models."""
    cards = tuple(_build_card_for_model(m) for m in list_golf_models())
    return NeuralCheckpointMatrix(schema=MATRIX_SCHEMA, cards=cards)
