"""Contracts for structural axis/pathway evidence assembly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_axis_evidence import (
    assemble_structural_axis_pathway_evidence,
)
from scripts.research.proximal_distal_energy.articulated_structural_corner_evidence import (
    StructuralCornerPathwayEvidence,
    assemble_structural_corner_pathway_evidence,
)

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _atlas(pathway: str) -> dict[str, np.ndarray]:
    filename = f"articulated_{pathway}_atlas.npz"
    with np.load(DATA / filename, allow_pickle=False) as source:
        return {name: np.asarray(source[name]).copy() for name in source.files}


def _authority() -> dict:
    plan = json.loads(
        (DATA / "articulated_structural_propagation_plan.json").read_text()
    )
    nominal = next(row for row in plan["corners"] if row["corner_id"] == "nominal")
    return nominal["authority"]


def _corner(
    pathway: str, corner_id: str, nominal: dict[str, np.ndarray]
) -> StructuralCornerPathwayEvidence:
    return assemble_structural_corner_pathway_evidence(
        pathway,
        nominal,
        nominal,
        corner_id=corner_id,
        requested_state_count=12,
        feasible_state_count=12,
        retained_failures=(),
        planned_headline_cell_count=384,
        all_registered_gates_passed=True,
        authority=_authority(),
    )


@pytest.mark.parametrize(
    ("pathway", "support", "classification"),
    [
        ("shaft", 126, "resolution_limited_on_shared_support"),
        ("ground", 0, "insufficient_shared_persistent_support"),
    ],
)
def test_axis_assembler_preserves_pathway_support_and_null_boundary(
    pathway, support, classification
) -> None:
    nominal = _atlas(pathway)
    low = _corner(pathway, "height_scale-low", nominal)
    high = _corner(pathway, "height_scale-high", nominal)

    evidence = assemble_structural_axis_pathway_evidence(
        "height_scale",
        low,
        high,
        low_scale=0.9,
        nominal_scale=1.0,
        high_scale=1.1,
    )

    assert evidence.axis_record["pathway"] == pathway
    assert evidence.axis_record["shared_persistent_cell_count"] == support
    assert evidence.axis_record["nonmonotonic_classification"] == classification
    if pathway == "ground":
        assert evidence.axis_record["low_to_nominal_secant_m_s_per_unit_scale"] is None
        assert evidence.axis_record["nominal_to_high_secant_m_s_per_unit_scale"] is None
    else:
        assert evidence.axis_record["low_to_nominal_secant_m_s_per_unit_scale"] == 0.0
        assert evidence.axis_record["nominal_to_high_secant_m_s_per_unit_scale"] == 0.0


def test_axis_assembler_rejects_crossed_axis_or_pathway_evidence() -> None:
    shaft = _atlas("shaft")
    ground = _atlas("ground")
    low = _corner("shaft", "height_scale-low", shaft)
    wrong_axis = _corner("shaft", "body_mass_scale-high", shaft)
    wrong_pathway = _corner("ground", "height_scale-high", ground)

    with pytest.raises(ValueError, match="low/high pair"):
        assemble_structural_axis_pathway_evidence(
            "height_scale",
            low,
            wrong_axis,
            low_scale=0.9,
            nominal_scale=1.0,
            high_scale=1.1,
        )
    with pytest.raises(ValueError, match="same pathway"):
        assemble_structural_axis_pathway_evidence(
            "height_scale",
            low,
            wrong_pathway,
            low_scale=0.9,
            nominal_scale=1.0,
            high_scale=1.1,
        )
