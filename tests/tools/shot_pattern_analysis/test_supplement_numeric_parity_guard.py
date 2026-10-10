"""Exact saved-report parity must fail before graphics and remain source bound."""

from __future__ import annotations
import json
from pathlib import Path
import pytest
from src.tools.shot_pattern_analysis import matched_supplement_pipeline as module

pytestmark = pytest.mark.unit
NAMES = (
    "REPORT_SUPPLEMENT.json",
    "REPORT_SUPPLEMENT.csv",
    "control_aliases.json",
    "REPORT_SUPPLEMENT_COMPARISONS.json",
    "REPORT_SUPPLEMENT_COMPARISONS.csv",
    "REPORT_SUPPLEMENT_COMPARISONS.md",
    "REPORT_ORIGINAL_SOURCE_INVENTORY.csv",
)


def reference(tmp_path):
    output = tmp_path / "data"
    output.mkdir()
    for name in NAMES:
        (output / name).write_text(name)
    path = tmp_path / "expected.json"
    path.write_text(json.dumps({n: module._sha256(output / n) for n in NAMES}))
    return path, output


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "malformed", "not_object", "duplicate"]
)
def test_reference_requires_exact_seven_valid_hashes(tmp_path, mutation):
    path, output = reference(tmp_path)
    value = json.loads(path.read_text())
    if mutation == "missing":
        value.pop(NAMES[0])
    if mutation == "extra":
        value["other"] = "0" * 64
    if mutation == "malformed":
        value[NAMES[0]] = "xyz"
    if mutation == "not_object":
        value = []
    path.write_text(json.dumps(value))
    if mutation == "duplicate":
        path.write_text(
            path.read_text()[:-1] + ',"' + NAMES[0] + '":"' + "0" * 64 + '"}'
        )
    with pytest.raises(ValueError):
        module._load_numeric_parity_reference(path)


def test_reference_checks_output_and_source_mutations(tmp_path):
    path, output = reference(tmp_path)
    bound = module._load_numeric_parity_reference(path)
    module._check_numeric_parity_reference(bound, output)
    (output / NAMES[0]).write_text("changed")
    with pytest.raises(ValueError, match="numeric"):
        module._check_numeric_parity_reference(bound, output)
    (output / NAMES[0]).write_text(NAMES[0])
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="reference"):
        module._check_numeric_parity_reference(bound, output)


def test_duplicate_figure_paths_are_rejected_even_with_same_hash(tmp_path):
    directory = tmp_path / "figures"
    directory.mkdir()
    p = directory / "one.png"
    p.write_bytes(b"image")
    row = {"relative_path": "one.png", "sha256": module._sha256(p)}
    with pytest.raises(ValueError, match="duplicate"):
        module._verify_figure_assets(tmp_path, {"figures": [row, dict(row)]})


@pytest.fixture
def small_pipeline(tmp_path, monkeypatch):
    report = {
        "cells": [],
        "plan_sha256": "plan",
        "planned_cells": 0,
        "counts": {},
        "common_targets_path": str(tmp_path / "targets"),
        "common_targets_sha256": "target",
        "common_target_provenance": {},
    }
    ref = {
        "cell": "pw",
        "source_protocol": "pw-primary-face-bounds/1",
        "source_root": str(tmp_path),
        "status": "completed",
        "config": {},
        "summary": {},
    }
    supplement = {
        "execution": {},
        "unique_cells": [ref],
        "aliases": [],
        "protocols": {},
        "inventory_sha256": "inventory",
        "unique_executed_cells": 1,
        "control_aliases": {"references": []},
    }
    monkeypatch.setattr(module, "_load_original_sources", lambda *a: (report, []))
    monkeypatch.setattr(module, "load_supplement_sources", lambda *a, **k: supplement)
    monkeypatch.setattr(module, "select_explicit_pairs", lambda *a: [])
    monkeypatch.setattr(module, "_recheck_source_inputs", lambda *a: None)
    monkeypatch.setattr(module, "_recheck_transition_evidence", lambda *a: None)

    def write_files(s, output):
        for name in ("REPORT_SUPPLEMENT.csv", "control_aliases.json"):
            (output / name).write_text("{}")
        return {}

    def write_comparisons(s, output):
        for name in (
            "REPORT_SUPPLEMENT_COMPARISONS.csv",
            "REPORT_SUPPLEMENT_COMPARISONS.json",
        ):
            (output / name).write_text("{}")
        return {}

    monkeypatch.setattr(module, "write_supplement_artifacts", write_files)
    monkeypatch.setattr(module, "write_supplement_comparisons", write_comparisons)
    calls = []
    monkeypatch.setattr(
        module, "_render_cell_figures", lambda *a: calls.append("render") or []
    )
    transition = tmp_path / "transition"
    transition.write_text("{}")
    args = {
        "original_primary_root": tmp_path / "primary",
        "original_control_root": tmp_path / "control",
        "protocol_roots": {},
        "inventory_path": tmp_path / "inventory",
        "transition_receipt_path": transition,
        "transition_receipt_sha256": module._sha256(transition),
    }
    return args, calls


def test_bad_parity_fails_before_any_graphic(tmp_path, small_pipeline):
    args, calls = small_pipeline
    path = tmp_path / "expected.json"
    path.write_text(json.dumps(dict.fromkeys(NAMES, "0" * 64)))
    with pytest.raises(ValueError, match="numeric"):
        module.build_final_supplement_report(
            **args, output_dir=tmp_path / "out", expected_numeric_hashes_path=path
        )
    assert calls == []
    assert not (tmp_path / "out/REPORT_SUPPLEMENT_PIPELINE.json").exists()


def test_valid_parity_is_bound_and_late_reference_mutation_blocks_receipt(
    tmp_path, small_pipeline, monkeypatch
):
    args, calls = small_pipeline
    module.build_final_supplement_report(**args, output_dir=tmp_path / "first")
    assert "numeric_parity_reference" not in json.loads(
        (tmp_path / "first/REPORT_SUPPLEMENT_PIPELINE.json").read_text()
    )
    path = tmp_path / "expected.json"
    path.write_text(
        json.dumps({n: module._sha256(tmp_path / "first" / n) for n in NAMES})
    )
    module.build_final_supplement_report(
        **args, output_dir=tmp_path / "second", expected_numeric_hashes_path=path
    )
    receipt = json.loads(
        (tmp_path / "second/REPORT_SUPPLEMENT_PIPELINE.json").read_text()
    )
    assert receipt["numeric_parity_reference"]["sha256"] == module._sha256(path)

    def mutate(*a):
        path.write_text(path.read_text() + " ")
        return []

    monkeypatch.setattr(module, "_render_cell_figures", mutate)
    with pytest.raises(ValueError, match="reference"):
        module.build_final_supplement_report(
            **args, output_dir=tmp_path / "third", expected_numeric_hashes_path=path
        )
    assert not (tmp_path / "third/REPORT_SUPPLEMENT_PIPELINE.json").exists()


def test_public_loader_rechecks_reference_after_asset_validation(tmp_path, monkeypatch):
    output = tmp_path / "out"
    output.mkdir()
    for name in NAMES:
        (output / name).write_text("{}")
    ledger_name = "REPORT_SUPPLEMENT_FIGURES.json"
    (output / ledger_name).write_text(
        json.dumps({"schema": "matched-supplement-figure-ledger/1", "figures": []})
    )
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps({name: module._sha256(output / name) for name in NAMES})
    )
    keys = (
        "report",
        "inventory",
        "control_aliases",
        "comparisons_json",
        "comparisons_csv",
        "comparison_markdown",
        "original_source_inventory",
        "figure_ledger",
    )
    paths = dict(zip(keys, (*NAMES, ledger_name), strict=True))
    receipt = {
        "schema": "matched-supplement-pipeline-receipt/1",
        "stage": "final",
        "artifact_sha256": {
            key: module._sha256(output / name) for key, name in paths.items()
        },
        "postprocessing_source_sha256": {},
        "numeric_parity_reference": module._load_numeric_parity_reference(
            reference_path
        ),
    }
    (output / "REPORT_SUPPLEMENT_PIPELINE.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(module, "_report_source_hashes", dict)
    monkeypatch.setattr(module, "_recheck_receipt_sources", lambda receipt: None)
    assert (
        module.load_verified_supplement_artifacts(output)["receipt"][
            "numeric_parity_reference"
        ]
        == receipt["numeric_parity_reference"]
    )
    original = module._verify_figure_assets

    def mutate(directory, ledger):
        original(directory, ledger)
        reference_path.write_text(reference_path.read_text() + " ")

    monkeypatch.setattr(module, "_verify_figure_assets", mutate)
    with pytest.raises(ValueError, match="reference"):
        module.load_verified_supplement_artifacts(output)


def test_duplicate_resolved_paths_are_rejected(tmp_path):
    directory = tmp_path / "figures"
    directory.mkdir()
    path = directory / "one.png"
    path.write_bytes(b"image")
    rows = [
        {"relative_path": name, "sha256": module._sha256(path)}
        for name in ("one.png", "./one.png")
    ]
    with pytest.raises(ValueError, match="duplicate"):
        module._verify_figure_assets(tmp_path, {"figures": rows})


def test_cli_forwards_optional_numeric_reference(tmp_path, monkeypatch):
    captured = {}

    def build(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(module, "build_final_supplement_report", build)
    flags = (
        "original-primary",
        "original-control",
        "pw-primary-face-bounds",
        "pw-control-face-bounds",
        "driver-control-speed-bounds",
        "inventory",
        "transition-receipt",
        "output-dir",
    )
    argv = [
        argument for flag in flags for argument in ("--" + flag, str(tmp_path / flag))
    ]
    argv += [
        "--transition-sha256",
        "0" * 64,
        "--expected-numeric-hashes",
        str(tmp_path / "reference.json"),
    ]
    assert module.main(argv) == 0
    assert captured["expected_numeric_hashes_path"] == tmp_path / "reference.json"


@pytest.mark.parametrize("collision", ["pair", "cell"])
def test_planned_output_collision_precedes_statistics_and_graphics(
    tmp_path, small_pipeline, monkeypatch, collision
):
    args, calls = small_pipeline
    pairs = [{"comparison_kind": "curve_strength"}]
    monkeypatch.setattr(module, "select_explicit_pairs", lambda *a: pairs)

    def reject_plan(actual_pairs, cell_refs):
        assert actual_pairs is pairs
        assert len(cell_refs) == 1
        assert cell_refs[0]["source_protocol"] == "pw-primary-face-bounds/1"
        raise ValueError(f"duplicate planned {collision} output")

    monkeypatch.setattr(
        module, "validate_planned_figure_identities", reject_plan, raising=False
    )
    monkeypatch.setattr(
        module, "_pair_result", lambda *a: calls.append("statistics") or {}
    )
    monkeypatch.setattr(
        module, "_render_pair_figures", lambda *a: calls.append("pair render") or []
    )
    with pytest.raises(ValueError, match=f"duplicate planned {collision}"):
        module.build_final_supplement_report(**args, output_dir=tmp_path / "out")
    assert calls == []
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("defect", ["duplicate", "wrong_hash"])
def test_builder_rejects_invalid_completed_figure_ledger(
    tmp_path, small_pipeline, monkeypatch, defect
):
    args, _ = small_pipeline

    def render(ref, figure_dir):
        figure_dir.mkdir(parents=True, exist_ok=True)
        image = figure_dir / "one.png"
        image.write_bytes(b"image")
        row = {"relative_path": "one.png", "sha256": module._sha256(image)}
        if defect == "duplicate":
            return [row, {**row, "relative_path": "./one.png"}]
        return [{**row, "sha256": "0" * 64}]

    monkeypatch.setattr(module, "_render_cell_figures", render)
    with pytest.raises(ValueError, match="duplicate|changed"):
        module.build_final_supplement_report(**args, output_dir=tmp_path / "out")
    assert not (tmp_path / "out/REPORT_SUPPLEMENT_PIPELINE.json").exists()
