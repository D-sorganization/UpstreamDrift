"""Assemble one resolution-aware structural axis/pathway evidence record."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scripts.research.proximal_distal_energy.articulated_structural_common_support import (
    OneSidedEngineeringSecants,
    SecantClassification,
    build_axis_summary_record,
    build_one_sided_engineering_secants,
    classify_one_sided_engineering_secants,
)
from scripts.research.proximal_distal_energy.articulated_structural_corner_evidence import (
    StructuralCornerPathwayEvidence,
)

REGISTERED_AXES = ("height_scale", "body_mass_scale", "joint_limit_scale")


@dataclass(frozen=True, slots=True)
class StructuralAxisPathwayEvidence:
    """One axis/pathway summary with its unpooled cell-level secants."""

    axis_record: dict[str, Any]
    secants: OneSidedEngineeringSecants
    classification: SecantClassification


def assemble_structural_axis_pathway_evidence(
    axis_name: str,
    low: StructuralCornerPathwayEvidence,
    high: StructuralCornerPathwayEvidence,
    *,
    low_scale: float,
    nominal_scale: float,
    high_scale: float,
) -> StructuralAxisPathwayEvidence:
    """Build one-sided secants only from correctly paired corner evidence."""

    if axis_name not in REGISTERED_AXES:
        raise ValueError("axis_name is not registered")
    if low.pathway != high.pathway:
        raise ValueError("axis corners must use the same pathway")
    expected = (f"{axis_name}-low", f"{axis_name}-high")
    observed = (low.corner_record["corner_id"], high.corner_record["corner_id"])
    if observed != expected:
        raise ValueError("axis corners do not match the registered low/high pair")
    if low.corner_record["pathway"] != low.pathway or (
        high.corner_record["pathway"] != high.pathway
    ):
        raise ValueError("axis corner records do not agree with their pathways")

    secants = build_one_sided_engineering_secants(
        axis_name,
        low.comparison,
        high.comparison,
        low_scale=low_scale,
        nominal_scale=nominal_scale,
        high_scale=high_scale,
    )
    classification = classify_one_sided_engineering_secants(secants)
    return StructuralAxisPathwayEvidence(
        axis_record=build_axis_summary_record(secants, classification),
        secants=secants,
        classification=classification,
    )


__all__ = [
    "REGISTERED_AXES",
    "StructuralAxisPathwayEvidence",
    "assemble_structural_axis_pathway_evidence",
]
