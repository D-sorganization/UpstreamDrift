"""Types, contracts, and model cards for NM-09 checkpoint matrix (#10624).

Implements Design by Contract (DbC) guards, model-card provenance, learning
curve metrics, native replay summaries, and compatibility assertions to prevent
accidental cross-model checkpoint loading.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

CHECKPOINT_SCHEMA = "neural-model-checkpoint/1.0.0"

__all__ = [
    "CHECKPOINT_SCHEMA",
    "BenefitResult",
    "ModelCheckpointCard",
    "ModelCheckpointStatus",
    "NativeReplayReceipt",
    "ThreeSeedEvidence",
    "assert_model_checkpoint_compatible",
]


class ModelCheckpointStatus(str, Enum):
    """Lifecycle qualification status for a physical model's checkpoint."""

    QUALIFIED_NATIVE = "qualified_native"
    TRAINED_SURROGATE = "trained_surrogate"
    KINEMATIC_PROPOSAL = "kinematic_proposal"
    BLOCKED_PREREQUISITE = "blocked_prerequisite"
    REFERENCE_ONLY = "reference_only"
    UNQUALIFIED = "unqualified"


@dataclass(frozen=True)
class ThreeSeedEvidence:
    """Multi-seed learning curve evidence ensuring training stability."""

    seed_losses: tuple[tuple[int, float], ...]
    mean_loss: float
    std_loss: float
    converged: bool

    def __post_init__(self) -> None:
        if not self.seed_losses:
            raise ValueError("seed_losses must be non-empty sequence")
        if not math.isfinite(self.mean_loss) or self.mean_loss < 0.0:
            raise ValueError(
                f"mean_loss must be finite non-negative float, got {self.mean_loss}"
            )
        if not math.isfinite(self.std_loss) or self.std_loss < 0.0:
            raise ValueError(
                f"std_loss must be finite non-negative float, got {self.std_loss}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed_losses": [list(item) for item in self.seed_losses],
            "mean_loss": self.mean_loss,
            "std_loss": self.std_loss,
            "converged": self.converged,
        }


@dataclass(frozen=True)
class NativeReplayReceipt:
    """Receipt from independent forward dynamic replay of the checkpoint."""

    is_valid: bool
    replay_rmse: float
    max_constraint_violation: float
    horizon_s: float
    time_step_s: float
    backend: str
    native_engine_version: str
    receipt_digest: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.replay_rmse) or self.replay_rmse < 0.0:
            raise ValueError(
                f"replay_rmse must be non-negative finite float, got {self.replay_rmse}"
            )
        if (
            not math.isfinite(self.max_constraint_violation)
            or self.max_constraint_violation < 0.0
        ):
            raise ValueError(
                f"max_constraint_violation must be non-negative finite float, got {self.max_constraint_violation}"
            )
        if self.horizon_s <= 0.0:
            raise ValueError(f"horizon_s must be positive float, got {self.horizon_s}")
        if self.time_step_s <= 0.0:
            raise ValueError(
                f"time_step_s must be positive float, got {self.time_step_s}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "replay_rmse": self.replay_rmse,
            "max_constraint_violation": self.max_constraint_violation,
            "horizon_s": self.horizon_s,
            "time_step_s": self.time_step_s,
            "backend": self.backend,
            "native_engine_version": self.native_engine_version,
            "receipt_digest": self.receipt_digest,
        }


@dataclass(frozen=True)
class BenefitResult:
    """Measured acceleration and benefit over classical matching."""

    speedup_factor: float | None = None
    loss_reduction_pct: float | None = None
    break_even_queries: int | None = None
    verdict: str = "unmeasured"

    def __post_init__(self) -> None:
        if self.speedup_factor is not None:
            if not math.isfinite(self.speedup_factor) or self.speedup_factor < 0.0:
                raise ValueError(
                    f"speedup_factor must be non-negative, got {self.speedup_factor}"
                )
        if self.loss_reduction_pct is not None:
            if not math.isfinite(self.loss_reduction_pct):
                raise ValueError(
                    f"loss_reduction_pct must be finite, got {self.loss_reduction_pct}"
                )
        if self.break_even_queries is not None:
            if self.break_even_queries < 0:
                raise ValueError(
                    f"break_even_queries must be >= 0, got {self.break_even_queries}"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "speedup_factor": self.speedup_factor,
            "loss_reduction_pct": self.loss_reduction_pct,
            "break_even_queries": self.break_even_queries,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class ModelCheckpointCard:
    """Versioned model card describing a qualified checkpoint or open blocker."""

    schema: str
    model_id: str
    backend: str
    topology: str
    q_dim: int
    v_dim: int
    u_dim: int
    constraint_count: int
    control_basis: str
    conditioning_schema: str
    generator_adapter: str
    dataset_hash: str | None = None
    split_hash: str | None = None
    weight_digest: str | None = None
    three_seed_evidence: ThreeSeedEvidence | None = None
    native_replay_receipt: NativeReplayReceipt | None = None
    benefit_result: BenefitResult = field(default_factory=BenefitResult)
    status: ModelCheckpointStatus = ModelCheckpointStatus.UNQUALIFIED
    blockers: tuple[str, ...] = ()
    governing_issues: tuple[str, ...] = ("#10585", "#10624")

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty string")
        if self.q_dim < 0 or self.v_dim < 0:
            raise ValueError("q_dim and v_dim must be non-negative")
        if self.u_dim < 0 or self.constraint_count < 0:
            raise ValueError("u_dim and constraint_count must be non-negative")
        if self.status == ModelCheckpointStatus.QUALIFIED_NATIVE:
            if not self.dataset_hash or not self.weight_digest:
                raise ValueError(
                    "dataset_hash and weight_digest must be non-empty for QUALIFIED_NATIVE"
                )

    def checkpoint_hash(self) -> str:
        """Cryptographic SHA-256 fingerprint binding all metadata and weights."""
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_id": self.model_id,
            "backend": self.backend,
            "topology": self.topology,
            "q_dim": self.q_dim,
            "v_dim": self.v_dim,
            "u_dim": self.u_dim,
            "constraint_count": self.constraint_count,
            "control_basis": self.control_basis,
            "conditioning_schema": self.conditioning_schema,
            "generator_adapter": self.generator_adapter,
            "dataset_hash": self.dataset_hash,
            "split_hash": self.split_hash,
            "weight_digest": self.weight_digest,
            "three_seed_evidence": (
                self.three_seed_evidence.as_dict()
                if self.three_seed_evidence is not None
                else None
            ),
            "native_replay_receipt": (
                self.native_replay_receipt.as_dict()
                if self.native_replay_receipt is not None
                else None
            ),
            "benefit_result": self.benefit_result.as_dict(),
            "status": self.status.value,
            "blockers": list(self.blockers),
            "governing_issues": list(self.governing_issues),
        }


def assert_model_checkpoint_compatible(
    *,
    card: ModelCheckpointCard,
    expected_model_id: str,
    expected_u_dim: int,
    expected_control_basis: str,
) -> None:
    """Precondition check preventing cross-model checkpoint confusion."""
    if card.schema != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"incompatible checkpoint schema: expected {CHECKPOINT_SCHEMA!r}, got {card.schema!r}"
        )
    if card.model_id != expected_model_id:
        raise ValueError(
            f"incompatible checkpoint model_id: expected {expected_model_id!r}, got {card.model_id!r}"
        )
    if card.u_dim != int(expected_u_dim):
        raise ValueError(
            f"incompatible checkpoint u_dim: expected {expected_u_dim}, got {card.u_dim}"
        )
    if card.control_basis != expected_control_basis:
        raise ValueError(
            f"incompatible checkpoint control_basis: expected {expected_control_basis!r}, got {card.control_basis!r}"
        )
