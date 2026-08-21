"""Fail-closed runtime boundary for governed articulated atlas authorities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
)
from scripts.research.proximal_distal_energy.spatial_full_body import SpatialModel


@dataclass(frozen=True, slots=True)
class AtlasStateSelection:
    """Registered state denominator, executable subset, and retained failures."""

    planned_states: tuple[tuple[int, int], ...]
    feasible_states: tuple[tuple[int, int], ...]
    retained_failures: tuple[dict[str, int | str], ...]


@dataclass(frozen=True, slots=True)
class AtlasCaseModel:
    """One corner-consistent model and its validated provenance."""

    model: SpatialModel
    metadata: dict[str, Any]
    model_sha256: str


def _require_authority(authority: object) -> ArticulatedAtlasAuthority:
    if not isinstance(authority, ArticulatedAtlasAuthority):
        raise TypeError("authority must be an ArticulatedAtlasAuthority")
    return authority


def _validate_indices(
    name: str,
    values: tuple[int, ...],
    *,
    upper: int,
) -> None:
    if (
        not values
        or len(set(values)) != len(values)
        or any(not isinstance(value, int) or not 0 <= value < upper for value in values)
    ):
        raise ValueError(f"{name} must contain unique in-range integers")


def resolve_atlas_states(
    authority: ArticulatedAtlasAuthority,
    case_indices: tuple[int, ...],
    sample_indices: tuple[int, ...],
) -> AtlasStateSelection:
    """Resolve the exact planned and feasible design without deleting failures."""

    governed = _require_authority(authority)
    _validate_indices("case_indices", case_indices, upper=18)
    _validate_indices("sample_indices", sample_indices, upper=13)
    selected_cases = set(governed.selected_case_indices.tolist())
    unregistered = [case for case in case_indices if case not in selected_cases]
    if unregistered:
        raise ValueError(
            f"case_indices contain an unregistered selected case: {unregistered}"
        )
    planned = tuple(
        (case_index, sample_index)
        for case_index in case_indices
        for sample_index in sample_indices
    )
    feasible = governed.feasible_states(case_indices, sample_indices)
    feasible_set = set(feasible)
    retained = tuple(
        {
            "case_index": case_index,
            "phase_index": sample_index,
            "failure_class": str(governed.failure_class[case_index, sample_index]),
        }
        for case_index, sample_index in planned
        if (case_index, sample_index) not in feasible_set
    )
    if len(planned) != len(feasible) + len(retained):
        raise RuntimeError("planned authority states were not fully accounted for")
    return AtlasStateSelection(
        planned_states=planned,
        feasible_states=feasible,
        retained_failures=retained,
    )


def build_atlas_case_model(
    authority: ArticulatedAtlasAuthority,
    case_index: int,
) -> AtlasCaseModel:
    """Build and revalidate the exact scaled model for one registered case."""

    governed = _require_authority(authority)
    model, metadata = governed.build_case_model(case_index)
    model_sha256 = governed.validate_case_model(case_index, model, metadata)
    return AtlasCaseModel(
        model=model,
        metadata=deepcopy(metadata),
        model_sha256=model_sha256,
    )


def runtime_authority_provenance(
    authority: ArticulatedAtlasAuthority,
) -> dict[str, Any]:
    """Return detached digest-bound provenance for an atlas result record."""

    governed = _require_authority(authority)
    return deepcopy(governed.provenance_record())


__all__ = [
    "AtlasCaseModel",
    "AtlasStateSelection",
    "build_atlas_case_model",
    "resolve_atlas_states",
    "runtime_authority_provenance",
]
