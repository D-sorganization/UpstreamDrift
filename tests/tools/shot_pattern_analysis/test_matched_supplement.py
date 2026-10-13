"""Predeclared supplements select existing matrix cells without copying physics."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

from src.tools.shot_pattern_analysis import matched_supplement

pytestmark = pytest.mark.unit


def test_three_protocols_have_exact_reviewed_distinct_cell_inventories():
    primary = matched_supplement.supplement_cell_names("pw_primary_face")
    pw_control = matched_supplement.supplement_cell_names("pw_control_face")
    driver_control = matched_supplement.supplement_cell_names("driver_control_speed")
    assert len(primary) == 18
    assert len(pw_control) == len(driver_control) == 4
    assert len({*primary, *pw_control, *driver_control}) == 26
    assert all(name.startswith("pitching_wedge_lean0_curve2_") for name in primary)
    assert all(name.startswith("pitching_wedge_lean5_curve2_") for name in pw_control)
    assert all(name.startswith("driver_lean5_curve2_") for name in driver_control)
    assert {name.rsplit("_", 1)[-1] for name in primary} >= {"only", "combined2"}
    aliases = matched_supplement.pw_control_alias_candidates()
    assert len(aliases) == 4
    assert set(aliases) <= set(primary)
    assert all(source == destination for source, destination in aliases.items())


def test_supplement_runner_reuses_matrix_pipeline_and_separates_speed_protocol(
    monkeypatch, tmp_path
):
    captured = []
    monkeypatch.setattr(
        matched_supplement,
        "run_matched_matrix",
        lambda *args, **kwargs: captured.append((args, kwargs)) or [],
    )
    matched_supplement.run_supplement_matrix(
        "pw_primary_face", tmp_path / "primary", n_shots=32
    )
    matched_supplement.run_supplement_matrix(
        "pw_control_face", tmp_path / "control", n_shots=32
    )
    matched_supplement.run_supplement_matrix(
        "driver_control_speed", tmp_path / "speed", n_shots=32
    )
    assert [len(item[1]["selection"].selected_cell_names) for item in captured] == [
        18,
        4,
        4,
    ]
    assert all(
        item[1]["selection"].match_face_bounds_deg == (-6.0, 6.0) for item in captured
    )
    assert [item[1]["selection"].match_speed_factors for item in captured] == [
        (0.75, 1.25),
        (0.75, 1.25),
        (0.75, 1.30),
    ]
    assert captured[0][1]["selection"].common_targets is None
    assert captured[1][1]["selection"].common_targets == {
        "pitching_wedge": pytest.approx(77.40247724866299)
    }
    assert captured[2][1]["selection"].common_targets == {
        "driver": pytest.approx(178.94875782337786)
    }
    assert len({item[1]["selection"].supplement_protocol for item in captured}) == 3
    with pytest.raises(ValueError, match="protocol"):
        matched_supplement.supplement_cell_names("unknown")


def _review_receipt(tmp_path):  # noqa: ANN001
    execution = {"source_sha256": {"reviewed": "same"}}
    protocols = {}
    for protocol, cells in matched_supplement.expected_supplement_final_cells().items():
        filename = protocol.replace("/", "_") + "_pilot.json"
        pilot = {
            "stage": "pilot",
            "execution": execution,
            "supplement_protocol": protocol,
            "selected_cell_names": list(cells),
            "cells": {
                name: {**config, "n_shots": 32} for name, config in cells.items()
            },
        }
        path = tmp_path / filename
        path.write_text(json.dumps(pilot))
        protocols[protocol] = {
            "final_cells": cells,
            "pilot_plan": filename,
            "pilot_plan_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    receipt = {
        "execution": execution,
        "supplement_stage": "final",
        "supplement_protocols": protocols,
        "evidence_files": {
            group["pilot_plan"]: group["pilot_plan_sha256"]
            for group in protocols.values()
        },
    }
    receipt_path = tmp_path / "accepted.json"
    receipt_path.write_text(json.dumps(receipt))
    return receipt_path, receipt


def test_final_supplement_gate_binds_all_26_cells_to_matching_pilots(tmp_path):  # noqa: ANN001
    receipt_path, receipt = _review_receipt(tmp_path)
    matched_supplement.validate_supplement_acceptance(
        receipt_path,
        "pw-primary-face-bounds/1",
        receipt["execution"],
        receipt["supplement_protocols"]["pw-primary-face-bounds/1"]["final_cells"],
    )
    assert (
        sum(
            len(group["final_cells"])
            for group in receipt["supplement_protocols"].values()
        )
        == 26
    )

    for mutation in (
        "missing_protocol",
        "wrong_bounds",
        "wrong_pilot_stage",
        "wrong_pilot_source",
        "wrong_acceptance_stage",
    ):
        changed = deepcopy(receipt)
        if mutation == "missing_protocol":
            changed["supplement_protocols"].pop("driver-control-speed-bounds/1")
        elif mutation == "wrong_bounds":
            group = changed["supplement_protocols"]["pw-primary-face-bounds/1"]
            first = next(iter(group["final_cells"].values()))
            first["match_face_bounds_deg"] = [-5.0, 6.0]
        elif mutation == "wrong_acceptance_stage":
            changed["supplement_stage"] = "pilot"
        else:
            group = changed["supplement_protocols"]["pw-primary-face-bounds/1"]
            pilot_path = tmp_path / group["pilot_plan"]
            original_pilot = pilot_path.read_text()
            pilot = json.loads(original_pilot)
            if mutation == "wrong_pilot_stage":
                pilot["stage"] = "final"
            else:
                pilot["execution"] = {"source_sha256": {"reviewed": "different"}}
            pilot_path.write_text(json.dumps(pilot))
            group["pilot_plan_sha256"] = hashlib.sha256(
                pilot_path.read_bytes()
            ).hexdigest()
            changed["evidence_files"][group["pilot_plan"]] = group["pilot_plan_sha256"]
        receipt_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="supplement|pilot"):
            matched_supplement.validate_supplement_acceptance(
                receipt_path,
                "pw-primary-face-bounds/1",
                receipt["execution"],
                receipt["supplement_protocols"]["pw-primary-face-bounds/1"][
                    "final_cells"
                ],
            )
        receipt_path.write_text(json.dumps(receipt))
        if mutation in ("wrong_pilot_stage", "wrong_pilot_source"):
            pilot_path.write_text(original_pilot)

    with pytest.raises(ValueError, match="protocol"):
        matched_supplement.validate_supplement_acceptance(
            receipt_path, "unreviewed-protocol/1", receipt["execution"], {}
        )
    with pytest.raises(ValueError, match="requested cells"):
        matched_supplement.validate_supplement_acceptance(
            receipt_path, "pw-primary-face-bounds/1", receipt["execution"], {}
        )


def test_matrix_final_path_enforces_supplement_review_before_any_cell(
    tmp_path, monkeypatch
):  # noqa: ANN001
    import src.tools.shot_pattern_analysis.matched_matrix as matrix

    receipt_path, receipt = _review_receipt(tmp_path)
    monkeypatch.setattr(matrix, "matched_source_snapshot", lambda: receipt["execution"])
    monkeypatch.setattr(matrix, "validate_final_acceptance", lambda *args: None)
    monkeypatch.setattr(
        matrix,
        "execute_and_export",
        lambda *args, **kwargs: pytest.fail("empty shard cannot simulate"),
    )
    options = {
        "n_shots": 10000,
        "stage": "final",
        "acceptance": receipt_path,
        "figures": "none",
        "shard_index": 18,
        "shard_count": 19,
        "selection": matrix.MatrixSelection(
            selected_cell_names=matched_supplement.supplement_cell_names(
                "pw_primary_face"
            ),
            match_face_bounds_deg=(-6.0, 6.0),
            match_speed_factors=(0.75, 1.25),
            supplement_protocol="pw-primary-face-bounds/1",
        ),
    }
    assert matrix.run_matched_matrix(tmp_path / "accepted", **options) == []
    with pytest.raises(ValueError, match="requested cells"):
        matrix.run_matched_matrix(
            tmp_path / "rejected",
            **{
                **options,
                "selection": replace(
                    options["selection"], match_face_bounds_deg=(-5.0, 6.0)
                ),
            },
        )
