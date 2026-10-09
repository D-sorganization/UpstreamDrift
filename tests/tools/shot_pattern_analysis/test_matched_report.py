"""Report assembly cannot hide missing or modified matrix evidence."""

import hashlib
import json
from dataclasses import asdict
import pytest
from src.tools.shot_pattern_analysis.matched_matrix import matched_matrix_cells
from src.tools.shot_pattern_analysis.matched_report import (
    load_matrix_report,
    load_validated_cell,
)

pytestmark = pytest.mark.unit


def fixture_matrix(root):
    name, config = matched_matrix_cells(n_shots=32)[0]
    execution = {"source_sha256": {"example.py": "abc"}}
    encoded = json.loads(json.dumps(asdict(config)))
    (root / "matrix_plan.json").write_text(
        json.dumps({"stage": "pilot", "execution": execution, "cells": {name: encoded}})
    )
    return name, encoded, execution


def test_incomplete_matrix_is_explicit_and_final_reporting_rejects(tmp_path):
    fixture_matrix(tmp_path)
    result = load_matrix_report(tmp_path, require_complete=False)
    assert result["counts"]["not_run"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        load_matrix_report(tmp_path)


def test_unavailable_cell_is_preserved_and_integrity_checked(tmp_path):
    name, config, execution = fixture_matrix(tmp_path)
    cell = tmp_path / name
    cell.mkdir()
    for filename, value in {
        "run_start.json": {"config": config, "execution": execution},
        "summary.json": {
            "status": "unavailable",
            "reason": "nonairborne_launch",
            "config": config,
        },
    }.items():
        (cell / filename).write_text(json.dumps(value))
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in cell.iterdir()
    }
    (cell / "manifest.json").write_text(json.dumps({"files_sha256": hashes}))
    result = load_matrix_report(tmp_path)
    assert result["counts"]["unavailable"] == 1
    assert result["cells"][0]["reason"] == "nonairborne_launch"
    (cell / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        load_matrix_report(tmp_path)


def test_reusable_cell_loader_requires_plan_inputs_and_source(tmp_path):
    name, config, execution = fixture_matrix(tmp_path)
    cell = tmp_path / name
    cell.mkdir()
    for filename, value in {
        "run_start.json": {"config": config, "execution": execution},
        "summary.json": {"status": "unavailable", "config": config},
    }.items():
        (cell / filename).write_text(json.dumps(value))
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in cell.iterdir()
    }
    (cell / "manifest.json").write_text(json.dumps({"files_sha256": hashes}))
    assert load_validated_cell(cell, config, execution)["status"] == "unavailable"
    with pytest.raises(ValueError, match="input mismatch"):
        load_validated_cell(cell, {**config, "n_shots": 999}, execution)
    with pytest.raises(RuntimeError, match="source changed"):
        load_validated_cell(cell, config, {"source_sha256": {"a": "new"}})
