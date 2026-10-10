"""Saved pair identities survive a graphics-only revision without statistics."""

from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import pytest

pytestmark = pytest.mark.unit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def saved_pair(tmp_path):
    sources = []
    bound = {}
    aliases = []
    for cell, protocol in [
        ("a", "original_control/1"),
        ("b", "pw-primary-face-bounds/1"),
    ]:
        root = tmp_path / cell
        directory = root / cell
        directory.mkdir(parents=True)
        config = {"n_shots": 10, "target_x_m": None}
        summary = {"config": config, "target_x_m": 100.0, "status": "completed"}
        for name, value in [
            ("summary.json", json.dumps(summary)),
            ("manifest.json", "{}"),
            ("shots.csv.gz", "saved csv"),
        ]:
            (directory / name).write_text(value)
        (root / "matrix_plan.json").write_text("{}")
        proof = {
            "protocol_id": protocol,
            "cell": cell,
            "execution_kind": "unique",
            "alias_cell": None,
            "source_plan_sha256": sha(root / "matrix_plan.json"),
            "source_cell_manifest_sha256": sha(directory / "manifest.json"),
            "source_csv_sha256": sha(directory / "shots.csv.gz"),
        }
        sources.append(proof)
        bound[f"{protocol}:{cell}"] = {
            "source_root": str(root),
            "status": "completed",
            "config": config,
            "manifest_sha256": proof["source_cell_manifest_sha256"],
            "summary_sha256": sha(directory / "summary.json"),
            "csv_sha256": proof["source_csv_sha256"],
        }
    comparison = {
        "comparison_kind": "across_lean",
        "sources": sources,
        "source_protocols": [s["protocol_id"] for s in sources],
        "target_x_m": 100.0,
        "common_supported_n": 9,
        "total_n_per_population": 10,
    }
    record = {
        "first_cell": "a",
        "second_cell": "b",
        "first_protocol": sources[0]["protocol_id"],
        "second_protocol": sources[1]["protocol_id"],
        "comparison_kind": "across_lean",
        "target_x_m": 100.0,
        "common_supported_n": 9,
        "total_n_per_population": 10,
        "figure_kind": "density_png",
        "relative_path": "comparisons/test_density.png",
        "sha256": "old",
        "presentation": "academic",
        "source_kind": "six_population_common_index_comparison",
    }
    loaded = {
        "receipt": {"source_cells_sha256": bound},
        "report": {"aliases": aliases},
        "comparisons": {
            "schema": "matched-supplement-comparisons/1",
            "comparisons": [comparison],
        },
    }
    return loaded, record


def test_reconstructs_exact_order_and_verified_alias(saved_pair):
    from scripts.research.shot_pattern_analysis.supplement_revision_adapter import (
        reconstruct_pair,
    )

    loaded, row = saved_pair
    proof = loaded["comparisons"]["comparisons"][0]["sources"][1]
    proof.update(execution_kind="provenance_alias", alias_cell="b")
    loaded["report"]["aliases"] = [
        {
            "alias_cell": "b",
            "alias_of_primary": "b",
            "source_protocol": proof["protocol_id"],
            "proof_status": "verified",
            "execution_kind": "provenance_alias",
            "requested_config": {"n_shots": 10, "target_x_m": 100.0},
            **{k: v for k, v in proof.items() if k.startswith("source_")},
        }
    ]
    pair = reconstruct_pair(loaded, row)
    assert pair["first"]["cell"] == "a" and pair["second"]["cell"] == "b"
    assert pair["second"]["execution_kind"] == "provenance_alias"
    assert pair["second"]["config"]["target_x_m"] == 100.0
    assert (
        loaded["receipt"]["source_cells_sha256"][proof["protocol_id"] + ":b"]["config"][
            "target_x_m"
        ]
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    ["order", "protocol", "target", "population", "csv", "duplicate", "unproved_alias"],
)
def test_rejects_wrong_identity_or_changed_source(saved_pair, mutation):
    from scripts.research.shot_pattern_analysis.supplement_revision_adapter import (
        reconstruct_pair,
    )

    loaded, row = saved_pair
    c = loaded["comparisons"]["comparisons"][0]
    if mutation == "order":
        c["sources"].reverse()
    elif mutation == "protocol":
        row["first_protocol"] = "invented"
    elif mutation == "target":
        row["target_x_m"] = 101
    elif mutation == "population":
        row["common_supported_n"] = 8
    elif mutation == "csv":
        ref = next(iter(loaded["receipt"]["source_cells_sha256"].values()))
        (Path(ref["source_root"]) / "a/shots.csv.gz").write_text("changed")
    elif mutation == "duplicate":
        loaded["comparisons"]["comparisons"].append(copy.deepcopy(c))
    else:
        c["sources"][1].update(execution_kind="provenance_alias", alias_cell="b")
    with pytest.raises(ValueError):
        reconstruct_pair(loaded, row)


def test_adapter_calls_only_selected_renderer(saved_pair, tmp_path):
    from scripts.research.shot_pattern_analysis.supplement_revision_adapter import (
        render_selected_records,
    )

    loaded, row = saved_pair
    calls = []

    def render(pair, output_dir, *, figure_kinds):
        calls.append((pair, output_dir, figure_kinds))
        return [row]

    result = render_selected_records(loaded, [row], tmp_path, renderer=render)
    assert result == [row]
    assert calls[0][1] == tmp_path / "figures"
    assert calls[0][2] == frozenset({"density_png"})


def test_renderer_source_mutation_cannot_escape_postcheck(saved_pair, tmp_path):
    from scripts.research.shot_pattern_analysis.supplement_revision_adapter import (
        render_selected_records,
    )

    loaded, row = saved_pair

    def render(pair, output_dir, *, figure_kinds):
        (Path(pair["first"]["source_root"]) / "a/shots.csv.gz").write_text(
            "changed during rendering"
        )
        return [row]

    with pytest.raises(ValueError, match="saved figure source changed"):
        render_selected_records(loaded, [row], tmp_path, renderer=render)


def test_renderer_cannot_change_requested_kind_or_ordered_identity(
    saved_pair, tmp_path
):
    from scripts.research.shot_pattern_analysis.supplement_revision_adapter import (
        render_selected_records,
    )

    loaded, row = saved_pair
    with pytest.raises(ValueError, match="renderer changed figure identities"):
        render_selected_records(
            loaded,
            [row],
            tmp_path,
            renderer=lambda *a, **k: [{**row, "first_protocol": "wrong"}],
        )
