"""Synthetic pipeline bytes and checkpoint order protect additive publication."""

import hashlib
import json
from pathlib import Path

import pytest

from src.tools.shot_pattern_analysis import matched_supplement_pipeline as module
from tests.tools.shot_pattern_analysis.test_supplement_numeric_parity_guard import (
    small_pipeline,
)

pytestmark = pytest.mark.unit
REFERENCE = (
    Path(__file__).with_name("fixtures") / "pipeline_orchestration_reference.json"
)
CHECKPOINTS = (
    "_load_numeric_parity_reference",
    "_load_original_sources",
    "load_supplement_sources",
    "_report_source_hashes",
    "select_explicit_pairs",
    "validate_planned_figure_identities",
    "_recheck_transition_evidence",
    "_pair_result",
    "write_supplement_artifacts",
    "render_original_source_inventory",
    "write_supplement_comparisons",
    "render_comparison_markdown",
    "_check_numeric_parity_reference",
    "_render_cell_figures",
    "_render_pair_figures",
    "_recheck_source_inputs",
    "_source_cell_hashes",
    "_verify_figure_assets",
)


def _pipeline_fixture(root, monkeypatch):
    args, _ = small_pipeline.__wrapped__(root, monkeypatch)
    monkeypatch.setattr(module, "_report_source_hashes", lambda: {"synthetic": "fixed"})
    monkeypatch.setattr(
        module,
        "select_explicit_pairs",
        lambda cells: [{"first": cells[0], "second": cells[0]}],
    )
    monkeypatch.setattr(
        module, "validate_planned_figure_identities", lambda *args: None
    )
    monkeypatch.setattr(
        module,
        "_pair_result",
        lambda *args: {
            "status": "conditional_only",
            "comparison_kind": "curve_strength",
            "patterns": {},
        },
    )
    monkeypatch.setattr(module, "_render_pair_figures", lambda *args: [])
    events = []
    for name in CHECKPOINTS:
        function = getattr(module, name)

        def logged(*args, _function=function, _name=name, **kwargs):
            events.append(_name)
            return _function(*args, **kwargs)

        monkeypatch.setattr(module, name, logged)
    output = root / "out"
    paths = module.build_final_supplement_report(**args, output_dir=output)
    receipt = json.loads(paths["pipeline_receipt"].read_text())
    for key, digest in receipt["artifact_sha256"].items():
        assert module.sha256_file(output / receipt["artifact_paths"][key]) == digest
    # File locations are fixture-specific; bind hashes to normalized bytes solely
    # for comparison. Production receipt hashes were checked against real bytes above.
    files = {
        str(path.relative_to(output)): path.read_text()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    replacements = {}
    for name, text in files.items():
        if name != "REPORT_SUPPLEMENT_PIPELINE.json":
            replacements[hashlib.sha256(text.encode()).hexdigest()] = hashlib.sha256(
                text.replace(str(root), "<synthetic-root>").encode()
            ).hexdigest()
    normalized = {}
    for name, text in files.items():
        text = text.replace(str(root), "<synthetic-root>")
        for digest, normalized_digest in replacements.items():
            text = text.replace(digest, normalized_digest)
        normalized[name] = text
    return {
        "events": events,
        "paths": {key: str(path.relative_to(output)) for key, path in paths.items()},
        "files": normalized,
    }


def test_pipeline_checkpoint_and_artifact_bytes_remain_exact(tmp_path, monkeypatch):
    assert (
        _pipeline_fixture(tmp_path, monkeypatch)
        == json.loads(REFERENCE.read_text())["receipt"]
    )


def test_verified_aliases_retain_source_summary_and_requested_config():
    ref = {"cell": "primary", "summary": {"n": 4}, "config": {"lean": 0}}
    alias = {
        "proof_status": "verified",
        "alias_of_primary": "primary",
        "alias_cell": "alias",
        "requested_config": {"lean": 5},
    }
    report = {
        "unique_cells": [ref],
        "aliases": [alias],
        "execution": {"snapshot": "fixed"},
    }
    rows = module._supplement_figure_sources(report)
    assert rows[0] is ref
    assert rows[1]["summary"] is ref["summary"]
    assert rows[1]["config"] is alias["requested_config"]
    assert [row["execution_kind"] for row in rows[1:]] == ["provenance_alias"]
    assert all(row["_execution"] is report["execution"] for row in rows)


def test_future_pipeline_snapshot_binds_decomposed_source():
    path = Path(module.__file__)
    assert (
        module._report_source_hashes()[path.name]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )
