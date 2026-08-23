"""Contracts for deterministic articulated structural figure data."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_axis_evidence import (
    assemble_structural_axis_pathway_evidence,
)
from scripts.research.proximal_distal_energy.articulated_structural_corner_evidence import (
    StructuralCornerEvidenceRequest,
    assemble_structural_corner_pathway_evidence,
)
from scripts.research.proximal_distal_energy.articulated_structural_figure_data import (
    build_structural_figure_data,
    validate_structural_figure_data,
    write_structural_figure_data,
)
from scripts.research.proximal_distal_energy.articulated_structural_result import (
    CORNER_IDS,
    assemble_structural_propagation_result,
)

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _atlas(pathway: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"articulated_{pathway}_atlas.npz") as source:
        return {name: np.asarray(source[name]).copy() for name in source.files}


def _authority() -> dict:
    plan = json.loads(
        (DATA / "articulated_structural_propagation_plan.json").read_text()
    )
    return next(row for row in plan["corners"] if row["corner_id"] == "nominal")[
        "authority"
    ]


def _result_and_packs():
    atlases = {pathway: _atlas(pathway) for pathway in ("shaft", "ground")}
    corners = {}
    records = []
    for corner_id in CORNER_IDS:
        for pathway in ("shaft", "ground"):
            evidence = assemble_structural_corner_pathway_evidence(
                pathway,
                atlases[pathway],
                atlases[pathway],
                request=StructuralCornerEvidenceRequest(
                    corner_id=corner_id,
                    cell_evidence_artifact=f"cells/{corner_id}-{pathway}.npz",
                    requested_state_count=12,
                    feasible_state_count=12,
                    retained_failures=(),
                    planned_headline_cell_count=384,
                    all_registered_gates_passed=True,
                    authority=_authority(),
                ),
            )
            corners[(corner_id, pathway)] = evidence
            records.append(evidence.corner_record)
    axes = []
    for axis_name, low, high in (
        ("height_scale", 0.9, 1.1),
        ("body_mass_scale", 0.85, 1.15),
        ("joint_limit_scale", 0.85, 1.15),
    ):
        for pathway in ("shaft", "ground"):
            axes.append(
                assemble_structural_axis_pathway_evidence(
                    axis_name,
                    corners[(f"{axis_name}-low", pathway)],
                    corners[(f"{axis_name}-high", pathway)],
                    low_scale=low,
                    nominal_scale=1.0,
                    high_scale=high,
                ).axis_record
            )
    result = assemble_structural_propagation_result(
        plan_contract_sha256="c" * 64,
        corner_records=tuple(records),
        axis_records=tuple(axes),
    )
    packs = {key: value.cell_evidence for key, value in corners.items()}
    return result, packs


def _redigest(record: dict) -> dict:
    payload = {
        key: value for key, value in record.items() if key != "figure_data_sha256"
    }
    record["figure_data_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return record


def test_figure_data_exposes_all_panels_and_ground_boundary() -> None:
    result, packs = _result_and_packs()

    figure = build_structural_figure_data(result, packs)

    assert len(figure["support"]) == 14
    assert len(figure["axis_secants"]) == 6
    assert len(figure["persistent_outcomes"]) == 6 * 126
    assert figure["retained_failures"] == []
    assert figure["nominal_ground_matched_cell_count"] == 0
    assert len(figure["figure_data_sha256"]) == 64
    shaft = next(
        row
        for row in figure["support"]
        if row["corner_id"] == "height_scale-low" and row["pathway"] == "shaft"
    )
    assert shaft["persistent_cell_count"] == 126
    assert shaft["resolved_persistent_cell_count"] == 0
    assert shaft["planned_cell_count"] == 384


def test_figure_data_rejects_missing_pack_and_writes_atomically(tmp_path) -> None:
    result, packs = _result_and_packs()
    missing = dict(packs)
    missing.pop(("height_scale-low", "shaft"))
    with pytest.raises(ValueError, match="exactly 14"):
        build_structural_figure_data(result, missing)

    figure = build_structural_figure_data(result, packs)
    output = tmp_path / "figure-data.json"
    write_structural_figure_data(figure, output)
    assert not output.with_suffix(".json.tmp").exists()
    assert json.loads(output.read_text()) == figure
    assert validate_structural_figure_data(output, result["result_sha256"]) == figure


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (
            lambda value: value["support"][2].__setitem__("persistent_cell_count", 125),
            "support transitions",
        ),
        (
            lambda value: value["persistent_outcomes"][0].__setitem__("resolved", True),
            "resolution status",
        ),
        (
            lambda value: value["support"].__setitem__(
                slice(0, 2), list(reversed(value["support"][:2]))
            ),
            "registered order",
        ),
        (
            lambda value: value.__setitem__("nominal_ground_matched_cell_count", 1),
            "nominal ground",
        ),
        (
            lambda value: value["axis_secants"][0].__setitem__(
                "low_to_nominal_secant_m_s_per_unit_scale", float("nan")
            ),
            "finite and ordered",
        ),
        (
            lambda value: value["axis_secants"][0].__setitem__(
                "nonmonotonic_classification", "parameter_importance"
            ),
            "classification does not reproduce",
        ),
        (
            lambda value: value["axis_secants"][0].__setitem__(
                "cell_classification_counts", {"resolution_limited": "one"}
            ),
            "classifications are not registered",
        ),
    ],
)
def test_figure_data_rejects_redigested_semantic_tampering(
    tmp_path, tamper, message
) -> None:
    result, packs = _result_and_packs()
    figure = build_structural_figure_data(result, packs)
    tamper(figure)
    _redigest(figure)

    with pytest.raises(ValueError, match=message):
        write_structural_figure_data(figure, tmp_path / "tampered.json")


def test_figure_data_rejects_wrong_result_binding_and_noncanonical_bytes(
    tmp_path,
) -> None:
    result, packs = _result_and_packs()
    figure = build_structural_figure_data(result, packs)
    output = tmp_path / "figure-data.json"
    write_structural_figure_data(figure, output)

    with pytest.raises(RuntimeError, match="result binding"):
        validate_structural_figure_data(output, "a" * 64)

    output.write_text(json.dumps(figure, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact bytes"):
        validate_structural_figure_data(output, result["result_sha256"])
