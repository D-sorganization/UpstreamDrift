"""Contracts for complete, atomic structural-propagation result evidence."""

from __future__ import annotations

import json

import pytest

from scripts.research.proximal_distal_energy.articulated_structural_result import (
    AXIS_PATHWAYS,
    CORNER_PATHWAYS,
    assemble_structural_propagation_result,
    validate_structural_propagation_result,
    write_structural_propagation_result,
)

pytestmark = pytest.mark.scientific


def _corner(corner_id: str, pathway: str) -> dict[str, object]:
    requested = 12
    feasible = 11 if corner_id == "height_scale-low" else 12
    executed = feasible * 32
    failures = (
        [
            {
                "case_index": 0,
                "phase_index": 12,
                "failure_class": "ik_nonconvergence",
            }
        ]
        if feasible == 11
        else []
    )
    matched = 1 if pathway == "shaft" else 0
    return {
        "corner_id": corner_id,
        "pathway": pathway,
        "cell_evidence_sha256": ("d" if pathway == "shaft" else "e") * 64,
        "requested_state_count": requested,
        "feasible_state_count": feasible,
        "retained_failures": failures,
        "planned_headline_cell_count": requested * 32,
        "feasible_headline_cell_count": executed,
        "executed_headline_cell_count": executed,
        "matched_cell_count": matched,
        "matched_fraction_of_feasible": matched / executed,
        "all_registered_gates_passed": True,
        "authority": {
            "authority_sha256": "a" * 64,
            "scales": {"height": 1.0, "body_mass": 1.0, "joint_limit": 1.0},
            "model_sha256": {"0": "b" * 64},
        },
    }


def _axis(axis_name: str, pathway: str) -> dict[str, object]:
    return {
        "axis_name": axis_name,
        "pathway": pathway,
        "low_scale": 0.9,
        "nominal_scale": 1.0,
        "high_scale": 1.1,
        "shared_persistent_cell_count": 0,
        "summary_statistic": (
            "unweighted median on identities persistent in both one-sided comparisons"
        ),
        "low_to_nominal_secant_m_s_per_unit_scale": None,
        "nominal_to_high_secant_m_s_per_unit_scale": None,
        "low_to_nominal_secant_range_m_s_per_unit_scale": None,
        "nominal_to_high_secant_range_m_s_per_unit_scale": None,
        "cell_classification_counts": {},
        "nonmonotonic_classification": "insufficient_shared_persistent_support",
    }


def _result():
    return assemble_structural_propagation_result(
        plan_contract_sha256="c" * 64,
        corner_records=tuple(_corner(*value) for value in reversed(CORNER_PATHWAYS)),
        axis_records=tuple(_axis(*value) for value in reversed(AXIS_PATHWAYS)),
    )


def test_complete_result_is_deterministic_and_plan_bound() -> None:
    first = _result()
    second = _result()

    assert first == second
    assert first["schema_version"] == "articulated-structural-propagation/v2"
    assert first["status"] == "complete"
    assert first["plan_contract_sha256"] == "c" * 64
    assert [
        (value["corner_id"], value["pathway"]) for value in first["corners"]
    ] == list(CORNER_PATHWAYS)
    assert len(first["corners"]) == 14
    assert [(value["axis_name"], value["pathway"]) for value in first["axes"]] == list(
        AXIS_PATHWAYS
    )
    assert len(first["result_sha256"]) == 64
    assert first["limitations"]["human_inference"] == "none"


def test_result_rejects_missing_duplicate_or_partial_evidence() -> None:
    corners = tuple(_corner(*value) for value in CORNER_PATHWAYS)
    axes = tuple(_axis(*value) for value in AXIS_PATHWAYS)
    with pytest.raises(ValueError, match="registered corner-pathway set"):
        assemble_structural_propagation_result(
            plan_contract_sha256="c" * 64,
            corner_records=corners[:-1],
            axis_records=axes,
        )
    with pytest.raises(ValueError, match="exactly the registered axis pathways"):
        assemble_structural_propagation_result(
            plan_contract_sha256="c" * 64,
            corner_records=corners,
            axis_records=axes[:-1] + (axes[0],),
        )
    partial = _corner(*CORNER_PATHWAYS[0])
    partial["executed_headline_cell_count"] = 352
    with pytest.raises(ValueError, match="complete feasible execution"):
        assemble_structural_propagation_result(
            plan_contract_sha256="c" * 64,
            corner_records=(partial, *corners[1:]),
            axis_records=axes,
        )
    bad_digest = _corner(*CORNER_PATHWAYS[0])
    bad_digest["cell_evidence_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="cell-evidence digest"):
        assemble_structural_propagation_result(
            plan_contract_sha256="c" * 64,
            corner_records=(bad_digest, *corners[1:]),
            axis_records=axes,
        )


def test_result_write_is_atomic_exact_and_tamper_evident(tmp_path) -> None:
    output = tmp_path / "result.json"
    record = _result()

    write_structural_propagation_result(record, output)

    assert not output.with_suffix(".json.tmp").exists()
    assert validate_structural_propagation_result(output, "c" * 64) == record
    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["corners"][0]["matched_cell_count"] = 1
    output.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not reproduce"):
        validate_structural_propagation_result(output, "c" * 64)
