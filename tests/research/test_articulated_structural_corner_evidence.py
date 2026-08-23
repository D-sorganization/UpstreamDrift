"""Contracts for release-qualified structural corner/pathway evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_corner_evidence import (
    StructuralCornerEvidenceRequest,
    assemble_structural_corner_pathway_evidence,
)

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _arrays(filename: str) -> dict[str, np.ndarray]:
    with np.load(DATA / filename, allow_pickle=False) as source:
        return {name: np.asarray(source[name]).copy() for name in source.files}


def _authority() -> dict:
    plan = json.loads(
        (DATA / "articulated_structural_propagation_plan.json").read_text()
    )
    nominal = next(row for row in plan["corners"] if row["corner_id"] == "nominal")
    return nominal["authority"]


def _request(corner_id: str, pathway: str, *, gates_passed: bool = True):
    return StructuralCornerEvidenceRequest(
        corner_id=corner_id,
        cell_evidence_artifact=f"cells/{corner_id}-{pathway}.npz",
        requested_state_count=12,
        feasible_state_count=12,
        retained_failures=(),
        planned_headline_cell_count=384,
        all_registered_gates_passed=gates_passed,
        authority=_authority(),
    )


@pytest.mark.parametrize(
    ("pathway", "filename", "matched"),
    [
        ("shaft", "articulated_shaft_atlas.npz", 126),
        ("ground", "articulated_ground_atlas.npz", 0),
    ],
)
def test_nominal_corner_assembles_complete_aligned_evidence(
    pathway, filename, matched
) -> None:
    arrays = _arrays(filename)

    evidence = assemble_structural_corner_pathway_evidence(
        pathway,
        arrays,
        arrays,
        request=_request("nominal", pathway),
    )

    assert evidence.corner_record["executed_headline_cell_count"] == 384
    assert evidence.corner_record["pathway"] == pathway
    assert evidence.corner_record["cell_evidence_sha256"] == str(
        evidence.cell_evidence["evidence_sha256"].item()
    )
    assert evidence.corner_record["matched_cell_count"] == matched
    assert evidence.comparison.common_executed_cell_count == 384
    assert len(evidence.comparison.persistent_identities) == matched
    assert evidence.cell_evidence["cell_identity"].shape == (384,)
    expected_status = {"persistent_unresolved", "common_unmatched"}
    assert set(evidence.cell_evidence["comparison_status"].tolist()) <= expected_status


def test_corner_assembler_rejects_partial_execution_and_global_gate_failure() -> None:
    arrays = _arrays("articulated_shaft_atlas.npz")
    partial = {
        name: value[:-1] if value.ndim and value.shape[0] == 12 else value
        for name, value in arrays.items()
    }
    with pytest.raises(RuntimeError, match="does not qualify"):
        assemble_structural_corner_pathway_evidence(
            "shaft",
            arrays,
            partial,
            request=_request("partial", "shaft"),
        )
    with pytest.raises(RuntimeError, match="does not qualify"):
        assemble_structural_corner_pathway_evidence(
            "shaft",
            arrays,
            arrays,
            request=_request("failed-global-gate", "shaft", gates_passed=False),
        )


def test_corner_assembler_rejects_per_cell_gate_failure() -> None:
    arrays = _arrays("articulated_shaft_atlas.npz")
    corner = {name: value.copy() for name, value in arrays.items()}
    corner["numerical_gates_passed"][0, 3, 0, 0, 0, 0] = False

    with pytest.raises(RuntimeError, match="per-cell comparison gates"):
        assemble_structural_corner_pathway_evidence(
            "shaft",
            arrays,
            corner,
            request=_request("failed-cell-gate", "shaft"),
        )
