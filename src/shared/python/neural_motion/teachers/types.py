"""Typed contracts for NM-04 teacher generation and acquisition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.shared.python.neural_motion.episodes.record import EpisodeRecord
from src.shared.python.neural_motion.json_io import SortedJsonWritableMixin

__all__ = [
    "ACQUISITION_SCHEMA",
    "TEACHER_SCHEMA",
    "AcquisitionEntry",
    "AcquisitionLog",
    "AcquisitionStrategy",
    "PerturbationKind",
    "RejectionEntry",
    "RejectionLedger",
    "TeacherOutcome",
    "TeacherSpec",
]

TEACHER_SCHEMA = "neural-teacher-episodes/1.0.0"
ACQUISITION_SCHEMA = "neural-acquisition-log/1.0.0"

_ALLOWED_REQUESTED = frozenset(
    {"q", "v", "u", "a_native", "q_next", "muscle_excitation", "muscle_activation"}
)


class PerturbationKind(str, Enum):
    """How a teacher candidate is drawn relative to a baseline."""

    NEAR_BASELINE = "near_baseline"
    STRATIFIED = "stratified"
    LOW_DISCREPANCY = "low_discrepancy"
    RANDOM_TORQUE = "random_torque"


class AcquisitionStrategy(str, Enum):
    """Active-learning scoring strategy (random is the control arm)."""

    UNCERTAINTY = "uncertainty"
    COVERAGE = "coverage"
    RANDOM_CONTROL = "random_control"


@dataclass(frozen=True)
class TeacherSpec:
    """One teacher candidate request.

    Design by Contract:
    - ``duration_s`` finite and > 0.
    - ``requested_channels`` non-empty and drawn from the allow-list.
    - Identity strings non-empty; ancestry is a tuple of strings.
    """

    seed: int
    family_id: str
    model_id: str
    perturbation: PerturbationKind
    geometry_stratum: str
    contact_stratum: str
    club_stratum: str
    duration_s: float
    ancestry: tuple[str, ...]
    requested_channels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        for name, value in (
            ("family_id", self.family_id),
            ("model_id", self.model_id),
            ("geometry_stratum", self.geometry_stratum),
            ("contact_stratum", self.contact_stratum),
            ("club_stratum", self.club_stratum),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.perturbation, PerturbationKind):
            raise TypeError("perturbation must be a PerturbationKind")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be a positive finite value")
        if not self.requested_channels:
            raise ValueError("requested_channels must be non-empty")
        unknown = set(self.requested_channels) - _ALLOWED_REQUESTED
        if unknown:
            raise ValueError(f"unknown requested channels: {sorted(unknown)}")
        if not isinstance(self.ancestry, tuple):
            raise TypeError("ancestry must be a tuple of strings")


@dataclass(frozen=True)
class TeacherOutcome:
    """Result of one teacher generation attempt (feasible or rejected)."""

    feasible: bool
    reason: str
    seed: int
    rejection_cost: float
    episode: EpisodeRecord | None = None
    teacher_objective: float | None = None
    converged: bool = False
    replay_digest: str | None = None
    corpus_role: str = "primary"

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if not math.isfinite(self.rejection_cost) or self.rejection_cost < 0.0:
            raise ValueError("rejection_cost must be a finite non-negative value")
        if self.feasible and self.episode is None:
            raise ValueError("feasible outcomes must carry an episode")
        if (
            not self.feasible
            and self.episode is not None
            and self.corpus_role != "quarantine"
        ):
            raise ValueError(
                "infeasible outcomes must not carry an accepted episode "
                "(quarantine role may retain the raw rollout)"
            )
        if self.corpus_role not in {"primary", "comparison", "quarantine"}:
            raise ValueError("corpus_role must be primary, comparison, or quarantine")
        if self.corpus_role == "quarantine" and self.episode is None:
            raise ValueError("quarantine outcomes must retain the raw episode")


@dataclass(frozen=True)
class RejectionEntry:
    """One counted rejected simulation attempt."""

    seed: int
    reason: str
    cost: float

    def as_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "reason": self.reason, "cost": self.cost}


@dataclass(frozen=True)
class RejectionLedger(SortedJsonWritableMixin):
    """Accumulated rejection reasons and simulation cost."""

    entries: tuple[RejectionEntry, ...]
    total_rejected_cost: float

    @classmethod
    def empty(cls) -> RejectionLedger:
        return cls(entries=(), total_rejected_cost=0.0)

    def append(self, *, seed: int, reason: str, cost: float) -> RejectionLedger:
        if not reason.strip():
            raise ValueError("reason must be non-empty")
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError("cost must be a finite non-negative value")
        entry = RejectionEntry(seed=seed, reason=reason, cost=cost)
        return RejectionLedger(
            entries=(*self.entries, entry),
            total_rejected_cost=self.total_rejected_cost + cost,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "entries": [entry.as_dict() for entry in self.entries],
            "total_rejected_cost": self.total_rejected_cost,
        }


@dataclass(frozen=True)
class AcquisitionEntry:
    """One selected active-learning candidate."""

    trial_id: str
    strategy: AcquisitionStrategy
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "strategy": self.strategy.value,
            "score": self.score,
        }


@dataclass(frozen=True)
class AcquisitionLog(SortedJsonWritableMixin):
    """Audited acquisition batch that never consumes test labels."""

    schema: str
    selected: tuple[AcquisitionEntry, ...]
    test_labels_consumed: int
    content_digest: str

    def __post_init__(self) -> None:
        if self.schema != ACQUISITION_SCHEMA:
            raise ValueError(f"schema must be {ACQUISITION_SCHEMA!r}")
        if self.test_labels_consumed != 0:
            raise ValueError("test_labels_consumed must remain 0")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "selected": [entry.as_dict() for entry in self.selected],
            "test_labels_consumed": self.test_labels_consumed,
            "content_digest": self.content_digest,
        }
