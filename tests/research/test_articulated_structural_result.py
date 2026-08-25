"""Contracts for complete, atomic structural-propagation result evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_cell_evidence import (
    build_structural_cell_evidence_from_atlas,
    write_structural_cell_evidence,
)
from scripts.research.proximal_distal_energy.articulated_structural_result import (
    AXIS_PATHWAYS,
    CORNER_PATHWAYS,
    assemble_structural_propagation_result,
    reconcile_structural_result_to_plan,
    validate_structural_propagation_bundle,
    validate_structural_propagation_result,
    write_structural_propagation_result,
)

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
AXIS_BOUNDS = {
    "height_scale": ("height", 0.9, 1.1),
    "body_mass_scale": ("body_mass", 0.85, 1.15),
    "joint_limit_scale": ("joint_limit", 0.85, 1.15),
}


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
    scales = {"height": 1.0, "body_mass": 1.0, "joint_limit": 1.0}
    for axis_name, (scale_key, low, high) in AXIS_BOUNDS.items():
        if corner_id == f"{axis_name}-low":
            scales[scale_key] = low
        elif corner_id == f"{axis_name}-high":
            scales[scale_key] = high
    return {
        "corner_id": corner_id,
        "pathway": pathway,
        "cell_evidence_artifact": f"cells/{corner_id}-{pathway}.npz",
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
            "scales": scales,
            "model_sha256": {"0": "b" * 64},
        },
    }


def _axis(axis_name: str, pathway: str) -> dict[str, object]:
    _, low, high = AXIS_BOUNDS[axis_name]
    return {
        "axis_name": axis_name,
        "pathway": pathway,
        "low_scale": low,
        "nominal_scale": 1.0,
        "high_scale": high,
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


def _plan() -> dict[str, object]:
    corners = []
    for corner_id, _ in CORNER_PATHWAYS[::2]:
        source = _corner(corner_id, "shaft")
        corners.append(
            {
                "corner_id": corner_id,
                "requested_state_count": source["requested_state_count"],
                "feasible_state_count": source["feasible_state_count"],
                "retained_failures": source["retained_failures"],
                "expected_shaft_headline_cell_count": source[
                    "feasible_headline_cell_count"
                ],
                "expected_ground_headline_cell_count": source[
                    "feasible_headline_cell_count"
                ],
                "authority": source["authority"],
            }
        )
    return {"status": "ready", "contract_sha256": "c" * 64, "corners": corners}


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
    tampered["corners"][0]["matched_cell_count"] += 1
    output.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not reproduce"):
        validate_structural_propagation_result(output, "c" * 64)


def test_result_bundle_reopens_and_reconciles_every_cell_pack(tmp_path) -> None:
    packs = {}
    for pathway in ("shaft", "ground"):
        with np.load(DATA / f"articulated_{pathway}_atlas.npz") as source:
            arrays = {name: np.asarray(source[name]) for name in source.files}
        packs[pathway] = build_structural_cell_evidence_from_atlas(pathway, arrays)
    corners = []
    for corner_id, pathway in CORNER_PATHWAYS:
        corner = _corner(corner_id, pathway)
        pack = packs[pathway]
        corner["feasible_state_count"] = 12
        corner["retained_failures"] = []
        corner["feasible_headline_cell_count"] = 384
        corner["executed_headline_cell_count"] = 384
        corner["cell_evidence_sha256"] = str(pack["evidence_sha256"].item())
        corner["matched_cell_count"] = int(np.count_nonzero(pack["matched_load_work"]))
        corner["matched_fraction_of_feasible"] = (
            corner["matched_cell_count"] / corner["executed_headline_cell_count"]
        )
        write_structural_cell_evidence(
            pack, tmp_path / corner["cell_evidence_artifact"]
        )
        corners.append(corner)
    record = assemble_structural_propagation_result(
        plan_contract_sha256="c" * 64,
        corner_records=tuple(corners),
        axis_records=tuple(_axis(*value) for value in AXIS_PATHWAYS),
    )
    output = tmp_path / "result.json"
    write_structural_propagation_result(record, output)

    assert validate_structural_propagation_bundle(output, "c" * 64) == record
    tampered_path = tmp_path / corners[0]["cell_evidence_artifact"]
    with np.load(tampered_path, allow_pickle=False) as source:
        tampered = {name: np.asarray(source[name]).copy() for name in source.files}
    tampered["matched_load_work"][0] = ~tampered["matched_load_work"][0]
    with tampered_path.open("wb") as stream:
        np.savez_compressed(stream, **tampered)
    with pytest.raises(RuntimeError, match="artifact is invalid"):
        validate_structural_propagation_bundle(output, "c" * 64)


def test_result_rejects_unsafe_or_duplicate_cell_artifacts() -> None:
    corners = [_corner(*value) for value in CORNER_PATHWAYS]
    axes = tuple(_axis(*value) for value in AXIS_PATHWAYS)
    corners[0]["cell_evidence_artifact"] = "../escape.npz"
    with pytest.raises(ValueError, match="safe relative NPZ"):
        assemble_structural_propagation_result(
            plan_contract_sha256="c" * 64,
            corner_records=tuple(corners),
            axis_records=axes,
        )
    corners = [_corner(*value) for value in CORNER_PATHWAYS]
    corners[1]["cell_evidence_artifact"] = corners[0]["cell_evidence_artifact"]
    with pytest.raises(ValueError, match="artifact paths must be unique"):
        assemble_structural_propagation_result(
            plan_contract_sha256="c" * 64,
            corner_records=tuple(corners),
            axis_records=axes,
        )


def test_result_reconciliation_binds_corner_authority_and_axis_scales() -> None:
    record = _result()
    plan = _plan()

    assert reconcile_structural_result_to_plan(record, plan) == record
    altered = json.loads(json.dumps(plan))
    altered["corners"][0]["authority"]["authority_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="corner evidence"):
        reconcile_structural_result_to_plan(record, altered)
    altered = json.loads(json.dumps(plan))
    altered["corners"][1]["authority"]["scales"]["height"] = 0.8
    with pytest.raises(RuntimeError, match="corner evidence"):
        reconcile_structural_result_to_plan(record, altered)


def test_result_reconciliation_rejects_plan_identity_or_denominator_drift() -> None:
    record = _result()
    plan = _plan()
    plan["contract_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="registered plan"):
        reconcile_structural_result_to_plan(record, plan)
    plan = _plan()
    plan["corners"][1]["expected_shaft_headline_cell_count"] = 320
    with pytest.raises(RuntimeError, match="corner evidence"):
        reconcile_structural_result_to_plan(record, plan)
