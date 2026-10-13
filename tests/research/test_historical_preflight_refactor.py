"""Synthetic saved-preflight review parity; historical default outputs stay untouched."""

import json
import math
from pathlib import Path

import pytest

from docs.research.shot_pattern_analysis.phase2.validation import (
    review_supplement_preflight as current,
)
from tests.research.test_historical_geometry_refactor import _archived_module

pytestmark = pytest.mark.unit


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _fixture(root, mode):
    phase = root / "docs/research/shot_pattern_analysis/phase2"
    validation = phase / "validation"
    execution = {"source": "synthetic"}
    for name in ("source_transition_supplement.json",):
        _write(phase / name, {})
    for name in (
        "independent_final_audit.py",
        "supplement_audit_protocol.py",
        "supplemental_refinement.py",
        "independent_preflight.py",
        "supplement_pilot_aliases.json",
        "SUPPLEMENT_COMPOSITION_TESTS.txt",
    ):
        _write(validation / name, {"synthetic": name})
    inputs = {"execution": execution, "pilots": {}}
    reports = {}
    for protocol in ("a", "b"):
        pilot = validation / protocol
        cells = {f"{protocol}_{i}": {} for i in range(13)}
        _write(pilot / "matrix_plan.json", {"cells": cells})
        audit = {
            "pass": True,
            "complete": True,
            "failures": [],
            "execution": execution,
            "plan_sha256": current.sha(pilot / "matrix_plan.json"),
            "script_sha256": current.sha(validation / "independent_final_audit.py"),
            "protocol_auditor_sha256": current.sha(
                validation / "supplement_audit_protocol.py"
            ),
            "audit_dependency_sha256": {},
            "audited_cells": 13,
            "planned_cells": 13,
            "cells": {
                cell: {"status": "completed", "rows": 96, "manifest_sha256": cell}
                for cell in cells
            },
        }
        audit_path = validation / f"{protocol}_audit.json"
        _write(audit_path, audit)
        inputs["pilots"][protocol] = {
            "root": protocol,
            "audit": {"path": audit_path.name, "sha256": current.sha(audit_path)},
        }
        reports[pilot.resolve()] = {
            "cells": [{"cell": cell, "status": "completed"} for cell in cells],
            "cell_manifest_sha256": {cell: cell for cell in cells},
        }
    if mode == "audit_hash":
        inputs["pilots"]["a"]["audit"]["sha256"] = "bad"
    inputs_path = validation / "supplement_refinement_inputs_v1.json"
    _write(inputs_path, inputs)
    nominals = [{"cell": f"nominal_{i}", "target_x_m": 100.0 + i} for i in range(9)]
    cases = [{"name": "nominal"}] + [{"name": f"tail_{i}"} for i in range(36)]
    rows = []
    for nominal in nominals:
        for case in cases:
            runs = [
                {
                    "status": "valid",
                    "endpoint_m": [nominal["target_x_m"] + i * 0.001, 0.0],
                }
                for i in range(6)
            ]
            comparisons = {
                name: {
                    "status": "evaluated",
                    "change_m": math.dist(runs[a]["endpoint_m"], runs[b]["endpoint_m"]),
                    "pass": True,
                }
                for name, (a, b) in current.COMPARISONS.items()
            }
            row = {**nominal, "case": case, "runs": runs, "comparisons": comparisons}
            if case["name"] == "nominal":
                row["nominal_fine_residual_m"] = math.dist(
                    runs[4]["endpoint_m"], [nominal["target_x_m"], 0.0]
                )
            if case["name"] == "tail_0":
                row["runs"] = [
                    {"status": "unsupported", "reason": "synthetic domain"}
                    for _ in range(6)
                ]
                row["comparisons"] = dict.fromkeys(
                    current.COMPARISONS,
                    {"status": "same_unsupported_domain", "pass": None},
                )
            rows.append(row)
    if mode == "comparison_change":
        rows[-1]["comparisons"]["contact_coarse"]["change_m"] += 0.01
    refinement = {
        "execution": execution,
        "inputs_sha256": current.sha(inputs_path),
        "script_sha256": current.sha(validation / "supplemental_refinement.py"),
        "comparison_helper_sha256": current.sha(
            validation / "independent_preflight.py"
        ),
        "settings_steps_dt": [list(x) for x in current.REFINEMENT_SETTINGS],
        "gate_m": 0.05,
        "pass": True,
        "failures": [],
        "cases": 333,
        "rows": rows,
    }
    _write(validation / "supplement_refinement_v1.json", refinement)
    _write(
        validation / "supplement_composition.json",
        {
            "tests": {
                "passed": 97,
                "log": "SUPPLEMENT_COMPOSITION_TESTS.txt",
                "sha256": current.sha(validation / "SUPPLEMENT_COMPOSITION_TESTS.txt"),
            }
        },
    )
    return phase, validation, execution, inputs, nominals, cases, reports


def _run(module, setup, monkeypatch, mode):
    _, validation, execution, inputs, nominals, cases, reports = setup
    events = []
    snapshots = iter(
        (execution, {"source": "changed"} if mode == "source_change" else execution)
    )

    def snapshot():
        events.append("snapshot")
        return next(snapshots)

    monkeypatch.setattr(module, "matched_source_snapshot", snapshot)
    monkeypatch.setattr(
        module,
        "load_qualified_nominals",
        lambda path: (events.append("nominals"), (inputs, nominals))[1],
    )
    monkeypatch.setattr(
        module,
        "validate_supplement_plan",
        lambda plan, stage, phase: (
            events.append("plan"),
            {"planned": len(plan["cells"]), "stage": stage},
        )[1],
    )
    monkeypatch.setattr(
        module,
        "load_matrix_report",
        lambda root: (events.append("matrix"), reports[root])[1],
    )
    monkeypatch.setattr(
        module,
        "validate_control_aliases",
        lambda *args: (
            events.append("aliases"),
            {"verified_aliases": 4, "new_executions": 0},
        )[1],
    )
    monkeypatch.setattr(module, "tail_cases", lambda: cases)
    output = validation / "supplement_independent_preflight_review.json"
    try:
        module.main()
        error = None
    except AssertionError as exc:
        error = repr(exc)
    raw = output.read_text() if output.exists() else None
    if output.exists():
        output.unlink()
    return {"bytes": raw, "events": events, "error": error}


@pytest.mark.parametrize(
    "mode", ["success", "source_change", "audit_hash", "comparison_change"]
)
def test_saved_preflight_exact_receipt_and_failure_order(tmp_path, monkeypatch, mode):
    setup = _fixture(tmp_path, mode)
    monkeypatch.chdir(tmp_path)
    with monkeypatch.context() as patch:
        before = _run(_archived_module(current), setup, patch, mode)
    with monkeypatch.context() as patch:
        after = _run(current, setup, patch, mode)
    assert after == before


def test_pilot_review_retains_all_rows_and_protocol_order(tmp_path, monkeypatch):
    setup = _fixture(tmp_path, "success")
    phase, validation, execution, inputs, _, _, reports = setup
    monkeypatch.setattr(
        current,
        "validate_supplement_plan",
        lambda plan, stage, phase: {"planned": len(plan["cells"]), "stage": stage},
    )
    monkeypatch.setattr(current, "load_matrix_report", lambda root: reports[root])
    files = []
    rows, protocols = current._review_pilots(
        inputs, execution, validation, phase, files
    )
    assert rows == 2496 and list(protocols) == ["a", "b"]
    assert [p.name for p in files] == [
        "matrix_plan.json",
        "a_audit.json",
        "matrix_plan.json",
        "b_audit.json",
    ]
