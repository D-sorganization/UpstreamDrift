"""End-to-end contracts for articulated structural publication assets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_structural_cell_evidence import (
    build_structural_cell_evidence,
    write_structural_cell_evidence,
)
from scripts.research.proximal_distal_energy.articulated_structural_common_support import (
    HeadlineCells,
)
from scripts.research.proximal_distal_energy.articulated_structural_publication import (
    main,
    publish_structural_figure_bundle,
)
from scripts.research.proximal_distal_energy.articulated_structural_result import (
    AXIS_PATHWAYS,
    AXIS_SCALE_KEYS,
    assemble_structural_propagation_result,
    write_structural_propagation_result,
)

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
PLAN = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/articulated_structural_propagation_plan.json"
)


def _cells(pathway: str, size: int) -> HeadlineCells:
    identities = tuple(
        (index // 32, index % 13, float(index % 4), 0.0, "primary", 0.0)
        for index in range(size)
    )
    zeros = np.zeros(size, dtype=float)
    return HeadlineCells(
        pathway=pathway,
        identities=identities,
        matched=np.zeros(size, dtype=bool),
        final_speed_difference_m_s=zeros.copy(),
        load_match_relative_error=zeros.copy(),
        work_match_relative_error=zeros.copy(),
        two_engine_speed_difference_discrepancy_m_s=zeros.copy(),
        time_step_speed_difference_discrepancy_m_s=zeros.copy(),
    )


def _axis(plan: dict, axis_name: str, pathway: str) -> dict:
    corners = {row["corner_id"]: row for row in plan["corners"]}
    scale_key = AXIS_SCALE_KEYS[axis_name]
    return {
        "axis_name": axis_name,
        "pathway": pathway,
        "low_scale": corners[f"{axis_name}-low"]["authority"]["scales"][scale_key],
        "nominal_scale": corners["nominal"]["authority"]["scales"][scale_key],
        "high_scale": corners[f"{axis_name}-high"]["authority"]["scales"][scale_key],
        "shared_persistent_cell_count": 0,
        "summary_statistic": "unweighted median on identities persistent in both one-sided comparisons",
        "low_to_nominal_secant_m_s_per_unit_scale": None,
        "nominal_to_high_secant_m_s_per_unit_scale": None,
        "low_to_nominal_secant_range_m_s_per_unit_scale": None,
        "nominal_to_high_secant_range_m_s_per_unit_scale": None,
        "cell_classification_counts": {},
        "nonmonotonic_classification": "insufficient_shared_persistent_support",
    }


def _write_bundle(tmp_path: Path) -> Path:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    corners = []
    for planned in plan["corners"]:
        corner_id = planned["corner_id"]
        for pathway in ("shaft", "ground"):
            size = planned[f"expected_{pathway}_headline_cell_count"]
            pack = build_structural_cell_evidence(
                _cells(pathway, size),
                gate_status=np.ones(size, dtype=bool),
                failure_class=np.full(size, "none"),
            )
            artifact = f"cells/{corner_id}-{pathway}.npz"
            write_structural_cell_evidence(pack, tmp_path / artifact)
            corners.append(
                {
                    "corner_id": corner_id,
                    "pathway": pathway,
                    "cell_evidence_artifact": artifact,
                    "cell_evidence_sha256": str(pack["evidence_sha256"].item()),
                    "requested_state_count": planned["requested_state_count"],
                    "feasible_state_count": planned["feasible_state_count"],
                    "retained_failures": planned["retained_failures"],
                    "planned_headline_cell_count": planned["requested_state_count"]
                    * 32,
                    "feasible_headline_cell_count": size,
                    "executed_headline_cell_count": size,
                    "matched_cell_count": 0,
                    "matched_fraction_of_feasible": 0.0,
                    "all_registered_gates_passed": True,
                    "authority": planned["authority"],
                }
            )
    axes = tuple(_axis(plan, *key) for key in AXIS_PATHWAYS)
    result = assemble_structural_propagation_result(
        plan_contract_sha256=plan["contract_sha256"],
        corner_records=tuple(corners),
        axis_records=axes,
    )
    output = tmp_path / "result.json"
    write_structural_propagation_result(result, output)
    return output


def test_publication_revalidates_bundle_and_emits_searchable_assets(tmp_path) -> None:
    result_path = _write_bundle(tmp_path)
    data_output = tmp_path / "figure-data.json"
    figure_output = tmp_path / "figure.svg"

    main(
        [
            "--result",
            str(result_path),
            "--plan",
            str(PLAN),
            "--figure-data",
            str(data_output),
            "--figure",
            str(figure_output),
        ]
    )

    record = json.loads(data_output.read_text(encoding="utf-8"))
    assert record["schema_version"] == "articulated-structural-figure-data/v1"
    figure = figure_output.read_text(encoding="utf-8")
    assert "Nominal Ground Matching: 0/384" in figure
    assert "No Human Or Coaching Inference" in figure
    assert "No Persistent Paired Outcomes" in figure
    assert record["figure_data_sha256"] in figure
    assert record["result_sha256"] in figure


def test_publication_rejects_output_contract_before_writing(tmp_path) -> None:
    result_path = _write_bundle(tmp_path)
    data_output = tmp_path / "figure-data.txt"
    figure_output = tmp_path / "figure.svg"

    with pytest.raises(ValueError, match="must be JSON"):
        publish_structural_figure_bundle(
            result_path=result_path,
            plan_path=PLAN,
            figure_data_output=data_output,
            figure_output=figure_output,
        )

    assert not data_output.exists()
    assert not figure_output.exists()


def test_publication_rejects_tampered_cell_pack_before_writing(tmp_path) -> None:
    result_path = _write_bundle(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    pack_path = tmp_path / result["corners"][0]["cell_evidence_artifact"]
    with np.load(pack_path, allow_pickle=False) as source:
        tampered = {name: np.asarray(source[name]).copy() for name in source.files}
    tampered["matched_load_work"][0] = True
    with pack_path.open("wb") as stream:
        np.savez_compressed(stream, **tampered)
    data_output = tmp_path / "figure-data.json"
    figure_output = tmp_path / "figure.svg"

    with pytest.raises(RuntimeError, match="artifact is invalid"):
        publish_structural_figure_bundle(
            result_path=result_path,
            plan_path=PLAN,
            figure_data_output=data_output,
            figure_output=figure_output,
        )

    assert not data_output.exists()
    assert not figure_output.exists()
