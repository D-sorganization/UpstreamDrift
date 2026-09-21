"""Neural campaign model roster freeze (NM-01 #10616).

One roster entry per TB-00 ``GolfModelIdentity``. Checkpoint dimensions follow
registered DOF / independent DOF / constraint counts — never a hard-coded
27x7 layout. Full-body cells stay deferred until the benefit review.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.shared.python.tour_baselines.models import (
    BackendType,
    GolfModelIdentity,
    ModelTopology,
)
from src.shared.python.tour_baselines.registry import get_golf_model, list_golf_models

__all__ = [
    "NeuralModelRoster",
    "NeuralRosterEntry",
    "RosterStage",
    "build_neural_model_roster",
    "resolve_roster_entry",
]

ROSTER_SCHEMA = "neural-model-roster/1.0.0"


class RosterStage(str, Enum):
    """Whether a model is in the pilot learning campaign."""

    PILOT_ELIGIBLE = "pilot_eligible"
    DEFERRED_PENDING_BENEFIT = "deferred_pending_benefit"
    REFERENCE_ONLY = "reference_only"


@dataclass(frozen=True)
class NeuralRosterEntry:
    """Frozen roster row for one registered golf model."""

    model_id: str
    backend: BackendType
    topology: ModelTopology
    q_dim: int
    independent_dof: int
    constraint_count: int
    has_simulated_club: bool
    checkpoint_layout: str
    pilot_stage: RosterStage
    blockers: tuple[str, ...]
    governing_issues: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        identity = get_golf_model(self.model_id)
        if self.backend is not identity.backend:
            raise ValueError(
                f"backend mismatch for {self.model_id}: "
                f"{self.backend} != {identity.backend}"
            )
        if self.q_dim != identity.dof:
            raise ValueError(
                f"q_dim {self.q_dim} != registered dof {identity.dof} "
                f"for {self.model_id}"
            )
        if self.checkpoint_layout == "hardcoded_27x7":
            raise ValueError(
                "checkpoint_layout must not use hardcoded_27x7; "
                "derive dimensions from GolfModelIdentity"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backend": self.backend.value,
            "topology": self.topology.value,
            "q_dim": self.q_dim,
            "independent_dof": self.independent_dof,
            "constraint_count": self.constraint_count,
            "has_simulated_club": self.has_simulated_club,
            "checkpoint_layout": self.checkpoint_layout,
            "pilot_stage": self.pilot_stage.value,
            "blockers": list(self.blockers),
            "governing_issues": list(self.governing_issues),
        }


@dataclass(frozen=True)
class NeuralModelRoster:
    """Immutable campaign roster keyed by model_id."""

    schema: str
    entries: tuple[NeuralRosterEntry, ...]

    def __post_init__(self) -> None:
        if self.schema != ROSTER_SCHEMA:
            raise ValueError(f"schema must be {ROSTER_SCHEMA}")
        if not self.entries:
            raise ValueError("roster entries must be non-empty")
        ids = [entry.model_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("roster model_ids must be unique")

    def model_ids(self) -> tuple[str, ...]:
        return tuple(entry.model_id for entry in self.entries)

    def content_digest(self) -> str:
        payload = json.dumps(
            {
                "schema": self.schema,
                "entries": [entry.as_dict() for entry in self.entries],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "content_digest": self.content_digest(),
            "entries": [entry.as_dict() for entry in self.entries],
        }


def _stage_for(identity: GolfModelIdentity) -> tuple[RosterStage, tuple[str, ...]]:
    if identity.topology is ModelTopology.FULL_BODY_MULTIBODY:
        return (
            RosterStage.DEFERRED_PENDING_BENEFIT,
            (
                "full-body training deferred pending NM-01 benefit review "
                "and generation/cost break-even",
            ),
        )
    if identity.topology is ModelTopology.REFERENCE_CATALOG_URDF:
        return (
            RosterStage.REFERENCE_ONLY,
            ("reference catalog URDF is not a pilot neural training target",),
        )
    if identity.topology is ModelTopology.KINEMATIC_RECONSTRUCTION:
        return (
            RosterStage.REFERENCE_ONLY,
            ("kinematic reconstruction models lack torque-driven supervision",),
        )
    return (RosterStage.PILOT_ELIGIBLE, ())


def _entry_from_identity(identity: GolfModelIdentity) -> NeuralRosterEntry:
    stage, blockers = _stage_for(identity)
    layout = (
        f"from_identity:q={identity.dof},"
        f"nu={identity.independent_dof},"
        f"nc={identity.constraint_count}"
    )
    return NeuralRosterEntry(
        model_id=identity.model_id,
        backend=identity.backend,
        topology=identity.topology,
        q_dim=identity.dof,
        independent_dof=identity.independent_dof,
        constraint_count=identity.constraint_count,
        has_simulated_club=identity.has_simulated_club,
        checkpoint_layout=layout,
        pilot_stage=stage,
        blockers=blockers,
        governing_issues=("#10585", "#10616"),
    )


def build_neural_model_roster() -> NeuralModelRoster:
    """Freeze one roster entry per registered golf model."""
    entries = tuple(
        _entry_from_identity(identity)
        for identity in sorted(list_golf_models(), key=lambda m: m.model_id)
    )
    return NeuralModelRoster(schema=ROSTER_SCHEMA, entries=entries)


def resolve_roster_entry(roster: NeuralModelRoster, model_id: str) -> NeuralRosterEntry:
    """Look up a roster row by canonical model_id."""
    for entry in roster.entries:
        if entry.model_id == model_id:
            return entry
    raise KeyError(f"unknown model_id in neural roster: {model_id}")
