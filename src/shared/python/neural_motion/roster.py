"""Checkpoint and control dimensions keyed to the TB-00 model roster (NM-01 #10616).

Dimensions are derived from :class:`GolfModelIdentity` entries registered under
TB-00 (#10585). Callers must not hard-code a 27×7 (or any other) shape; each
model declares its own ``n_q`` / ``n_v`` / ``n_u``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.shared.python.tour_baselines.models import BackendType, GolfModelIdentity
from src.shared.python.tour_baselines.registry import get_golf_model, list_golf_models

__all__ = [
    "CheckpointDimensions",
    "NeuralModelRosterEntry",
    "dimensions_for_model",
    "freeze_neural_model_roster",
]


@dataclass(frozen=True, slots=True)
class CheckpointDimensions:
    """Tensor / state dimensions for one model-specific neural checkpoint.

    Design by Contract:
    - ``model_id`` must resolve in the TB-00 golf-model registry.
    - ``n_q`` must equal the registered model's ``dof``.
    - ``n_v`` must equal the registered model's ``independent_dof``.
    - ``n_u`` and ``n_contact`` are positive / non-negative respectively and
      are supplied by the caller (actuation layout is model-specific and is
      not invented here).
    """

    model_id: str
    n_q: int
    n_v: int
    n_u: int
    n_contact: int = 0

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        identity = get_golf_model(self.model_id)
        if self.n_q != identity.dof:
            raise ValueError(
                f"n_q ({self.n_q}) must equal roster dof ({identity.dof}) "
                f"for model_id={self.model_id!r}"
            )
        if self.n_v != identity.independent_dof:
            raise ValueError(
                f"n_v ({self.n_v}) must equal roster independent_dof "
                f"({identity.independent_dof}) for model_id={self.model_id!r}"
            )
        if isinstance(self.n_u, bool) or not isinstance(self.n_u, int) or self.n_u < 1:
            raise ValueError(f"n_u must be a positive int, got {self.n_u!r}")
        if (
            isinstance(self.n_contact, bool)
            or not isinstance(self.n_contact, int)
            or self.n_contact < 0
        ):
            raise ValueError(
                f"n_contact must be a non-negative int, got {self.n_contact!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "n_q": self.n_q,
            "n_v": self.n_v,
            "n_u": self.n_u,
            "n_contact": self.n_contact,
        }


@dataclass(frozen=True, slots=True)
class NeuralModelRosterEntry:
    """Frozen neural-facing view of one TB-00 :class:`GolfModelIdentity`."""

    model_id: str
    backend: BackendType
    dof: int
    independent_dof: int
    constraint_count: int
    has_simulated_club: bool
    governing_issue: str = "#10585"

    @classmethod
    def from_identity(cls, identity: GolfModelIdentity) -> NeuralModelRosterEntry:
        return cls(
            model_id=identity.model_id,
            backend=identity.backend,
            dof=identity.dof,
            independent_dof=identity.independent_dof,
            constraint_count=identity.constraint_count,
            has_simulated_club=identity.has_simulated_club,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backend": self.backend.value,
            "dof": self.dof,
            "independent_dof": self.independent_dof,
            "constraint_count": self.constraint_count,
            "has_simulated_club": self.has_simulated_club,
            "governing_issue": self.governing_issue,
        }


def dimensions_for_model(
    model_id: str,
    n_u: int,
    *,
    n_contact: int = 0,
) -> CheckpointDimensions:
    """Build :class:`CheckpointDimensions` from the TB-00 roster entry.

    ``n_q`` / ``n_v`` are taken from the registered identity; ``n_u`` must be
    supplied because control layouts differ across backends and are not a
    fixed multiple of DoF.
    """
    identity = get_golf_model(model_id)
    return CheckpointDimensions(
        model_id=identity.model_id,
        n_q=identity.dof,
        n_v=identity.independent_dof,
        n_u=n_u,
        n_contact=n_contact,
    )


def freeze_neural_model_roster() -> tuple[NeuralModelRosterEntry, ...]:
    """Return the frozen neural roster mirroring every TB-00 golf model."""
    return tuple(
        NeuralModelRosterEntry.from_identity(model) for model in list_golf_models()
    )
