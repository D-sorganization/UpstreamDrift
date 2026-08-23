"""Assemble release-qualified evidence for one structural corner and pathway."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from scripts.research.proximal_distal_energy.articulated_structural_cell_evidence import (
    Array,
    build_structural_cell_evidence_from_atlas,
)
from scripts.research.proximal_distal_energy.articulated_structural_common_support import (
    CommonSupportComparison,
    Pathway,
    build_corner_support_summary,
    compare_common_support,
    corner_support_summary_record,
    extract_headline_cells,
)


@dataclass(frozen=True, slots=True)
class StructuralCornerPathwayEvidence:
    """One complete corner/pathway record with aligned cell evidence."""

    pathway: Pathway
    comparison: CommonSupportComparison
    corner_record: dict[str, Any]
    cell_evidence: dict[str, Array]

    def __post_init__(self) -> None:
        if self.comparison.pathway != self.pathway:
            raise ValueError("corner comparison pathway does not agree")
        if str(self.cell_evidence["pathway"].item()) != self.pathway:
            raise ValueError("corner cell-evidence pathway does not agree")


@dataclass(frozen=True, slots=True)
class StructuralCornerEvidenceRequest:
    """Governed denominators, authority, and output identity for one corner."""

    corner_id: str
    cell_evidence_artifact: str
    requested_state_count: int
    feasible_state_count: int
    retained_failures: tuple[Mapping[str, Any], ...]
    planned_headline_cell_count: int
    all_registered_gates_passed: bool
    authority: Mapping[str, Any]


def assemble_structural_corner_pathway_evidence(
    pathway: Pathway,
    nominal_atlas_arrays: Mapping[str, Any],
    corner_atlas_arrays: Mapping[str, Any],
    *,
    request: StructuralCornerEvidenceRequest,
    absolute_resolution_floor_m_s: float = 0.001,
) -> StructuralCornerPathwayEvidence:
    """Join common support, denominators, gates, and authority fail closed."""

    nominal = extract_headline_cells(pathway, nominal_atlas_arrays)
    corner = extract_headline_cells(pathway, corner_atlas_arrays)
    comparison = compare_common_support(
        nominal,
        corner,
        absolute_resolution_floor_m_s=absolute_resolution_floor_m_s,
    )
    cell_evidence = build_structural_cell_evidence_from_atlas(
        pathway,
        corner_atlas_arrays,
        comparison=comparison,
    )
    if not bool(cell_evidence["gate_status"].all()):
        raise RuntimeError("corner contains failed per-cell comparison gates")
    summary = build_corner_support_summary(
        request.corner_id,
        corner,
        requested_state_count=request.requested_state_count,
        feasible_state_count=request.feasible_state_count,
        retained_failures=request.retained_failures,
        planned_headline_cell_count=request.planned_headline_cell_count,
        all_registered_gates_passed=request.all_registered_gates_passed,
        authority=request.authority,
    )
    corner_record = corner_support_summary_record(summary)
    corner_record["cell_evidence_artifact"] = request.cell_evidence_artifact
    corner_record["cell_evidence_sha256"] = str(cell_evidence["evidence_sha256"].item())
    return StructuralCornerPathwayEvidence(
        pathway=pathway,
        comparison=comparison,
        corner_record=corner_record,
        cell_evidence=cell_evidence,
    )


__all__ = [
    "StructuralCornerEvidenceRequest",
    "StructuralCornerPathwayEvidence",
    "assemble_structural_corner_pathway_evidence",
]
