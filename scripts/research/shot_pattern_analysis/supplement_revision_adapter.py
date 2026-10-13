"""Reconstruct only verified saved figure inputs; never recompute pair statistics."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .supplement_publication import safe_file, sha256


def pair_identity(record: dict) -> tuple[str, str, str, str, str]:
    """Keep source order and protocol in the identity, including reused aliases."""
    return (
        record["comparison_kind"],
        record["first_protocol"],
        record["first_cell"],
        record["second_protocol"],
        record["second_cell"],
    )


def _comparison_identity(record: dict) -> tuple:
    sources = record.get("sources", [])
    if len(sources) != 2:
        raise ValueError("comparison must identify two ordered sources")
    return (
        record["comparison_kind"],
        sources[0]["protocol_id"],
        sources[0].get("alias_cell") or sources[0]["cell"],
        sources[1]["protocol_id"],
        sources[1].get("alias_cell") or sources[1]["cell"],
    )


def _source_ref(loaded: dict, proof: dict, target: float, total: int) -> dict:
    protocol = proof["protocol_id"]
    cell = proof["cell"]
    snapshot = loaded["receipt"]["source_cells_sha256"].get(f"{protocol}:{cell}")
    if not snapshot or snapshot["status"] != "completed":
        raise ValueError("figure source is not a bound completed cell")
    root = Path(snapshot["source_root"])
    directory = root / cell
    files = {
        "manifest.json": snapshot["manifest_sha256"],
        "summary.json": snapshot["summary_sha256"],
        "shots.csv.gz": snapshot["csv_sha256"],
    }
    if (
        proof["source_cell_manifest_sha256"] != files["manifest.json"]
        or proof["source_csv_sha256"] != files["shots.csv.gz"]
    ):
        raise ValueError("comparison source proof differs from receipt")
    for name, digest in files.items():
        if sha256(safe_file(directory, name)) != digest:
            raise ValueError("saved figure source changed")
    if sha256(safe_file(root, "matrix_plan.json")) != proof["source_plan_sha256"]:
        raise ValueError("saved source plan changed")
    summary = json.loads(safe_file(directory, "summary.json").read_text())
    if sha256(safe_file(directory, "summary.json")) != files["summary.json"]:
        raise ValueError("summary changed while reading")
    config = snapshot["config"]
    if (
        summary["config"] != config
        or config["n_shots"] != total
        or round(float(summary["target_x_m"]), 12) != round(float(target), 12)
    ):
        raise ValueError("saved figure config, count or target differs")
    ref = {
        "cell": cell,
        "source_protocol": protocol,
        "source_root": str(root),
        "status": "completed",
        "config": config,
        "summary": summary,
        "execution_kind": "unique",
        **{k: v for k, v in proof.items() if k.startswith("source_")},
    }
    if proof.get("execution_kind") == "provenance_alias":
        matches = [
            a
            for a in loaded["report"]["aliases"]
            if a.get("alias_cell") == proof.get("alias_cell")
            and a.get("alias_of_primary") == cell
            and a.get("source_protocol") == protocol
        ]
        if len(matches) != 1 or matches[0].get("proof_status") != "verified":
            raise ValueError("source alias is not uniquely verified")
        alias = matches[0]
        if any(
            alias.get(key) != proof[key]
            for key in (
                "source_plan_sha256",
                "source_cell_manifest_sha256",
                "source_csv_sha256",
            )
        ):
            raise ValueError("alias proof hashes differ")
        requested = alias["requested_config"]
        if {k: v for k, v in requested.items() if k != "target_x_m"} != {
            k: v for k, v in config.items() if k != "target_x_m"
        } or round(float(requested["target_x_m"]), 12) != round(float(target), 12):
            raise ValueError("alias requested configuration differs")
        ref.update(
            alias,
            cell=alias["alias_cell"],
            config=requested,
            execution_kind="provenance_alias",
        )
    elif proof.get("execution_kind") != "unique" or proof.get("alias_cell") is not None:
        raise ValueError("unknown source execution identity")
    return ref


def reconstruct_pair(loaded: dict, record: dict) -> dict:
    """Resolve a ledger pair using its already-verified numerical comparison."""
    comparisons = loaded["comparisons"]
    if comparisons.get("schema") != "matched-supplement-comparisons/1":
        raise ValueError("comparison inventory schema differs")
    matches = [
        row
        for row in comparisons["comparisons"]
        if _comparison_identity(row) == pair_identity(record)
    ]
    if len(matches) != 1:
        raise ValueError("pair is absent or ambiguous in verified comparisons")
    comparison = matches[0]
    proofs = comparison["sources"]
    if (
        comparison["source_protocols"] != [p["protocol_id"] for p in proofs]
        or comparison["target_x_m"] != record["target_x_m"]
    ):
        raise ValueError("ordered protocol or target differs")
    if record.get("source_kind") == "six_population_common_index_comparison" and (
        record.get("common_supported_n") != comparison["common_supported_n"]
        or record.get("total_n_per_population") != comparison["total_n_per_population"]
    ):
        raise ValueError("common-six population differs")
    refs = [
        _source_ref(
            loaded, p, comparison["target_x_m"], comparison["total_n_per_population"]
        )
        for p in proofs
    ]
    return {
        "first": refs[0],
        "second": refs[1],
        "comparison_kind": comparison["comparison_kind"],
        "target_x_m": comparison["target_x_m"],
    }


def render_selected_records(
    loaded: dict,
    records: list[dict],
    output_dir: Path,
    *,
    renderer: Callable[..., list[dict]] | None = None,
) -> list[dict]:
    """Call only the public selective figure API once per exact ordered pair."""
    if renderer is None:
        from src.tools.shot_pattern_analysis import matched_supplement_figures

        renderer = getattr(
            matched_supplement_figures, "render_selected_pair_figures", None
        )
        if renderer is None:
            raise RuntimeError("qualified selective renderer API is unavailable")
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[pair_identity(row)].append(row)
    results: list[dict[str, Any]] = []
    for rows in groups.values():
        pairs = [reconstruct_pair(loaded, row) for row in rows]
        if any(pair != pairs[0] for pair in pairs[1:]):
            raise ValueError("figure group source identities differ")
        kinds = frozenset(row["figure_kind"] for row in rows)
        if len(kinds) != len(rows):
            raise ValueError("duplicate selected figure kind")
        produced = renderer(pairs[0], Path(output_dir) / "figures", figure_kinds=kinds)
        expected = {
            r["relative_path"]: {k: v for k, v in r.items() if k != "sha256"}
            for r in rows
        }
        actual = {
            r["relative_path"]: {k: v for k, v in r.items() if k != "sha256"}
            for r in produced
        }
        if len(actual) != len(produced) or actual != expected:
            raise ValueError("renderer changed figure identities")
        # Re-read the bound inputs after rendering; final parent verification is broader.
        for row in rows:
            reconstruct_pair(loaded, row)
        results.extend(produced)
    return results
