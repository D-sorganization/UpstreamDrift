"""Publish only the measured density-layout repairs from saved shot samples.

This script deliberately never invokes the simulation or full-report exporter.
It requires an already copied child publication and replaces only the density
PNG/PDF/SVG triples enumerated by the frozen baseline geometry diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parents[6]
PHASE2 = ANALYSIS_ROOT / "docs/research/shot_pattern_analysis/phase2"
EVIDENCE = Path(__file__).resolve().parent
BASELINE = PHASE2 / "supplement_report_v1"
CHILD = PHASE2 / "supplement_report_v2_layout_repair"
STAGING = EVIDENCE / "layout_repair_render_staging"
REPAIR_SOURCE = Path(
    "/home/dieterolson/.codex/worktrees/shot-pattern-layout-gutter/UpstreamDrift"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_repair_scope() -> tuple[Path, dict, Path, dict, list, dict, dict]:
    """Check measured scope and unchanged baseline bytes before rendering."""
    scope_path = EVIDENCE / "density_layout_repair_scope.json"
    scope = json.loads(scope_path.read_text())
    targets = {Path(item["svg_path"]).name: item for item in scope["affected"]}
    if len(targets) != 49:
        raise ValueError("measured scope must contain exactly 49 unique triples")
    pipeline_path = BASELINE / "REPORT_SUPPLEMENT_PIPELINE.json"
    pipeline = json.loads(pipeline_path.read_text())
    for key in (
        "report",
        "inventory",
        "comparisons_json",
        "comparisons_csv",
        "comparison_markdown",
        "original_source_inventory",
        "control_aliases",
        "figure_ledger",
    ):
        relative = pipeline["artifact_paths"][key]
        if sha(BASELINE / relative) != pipeline["artifact_sha256"][key]:
            raise ValueError(f"baseline artifact changed: {relative}")
        if sha(CHILD / relative) != sha(BASELINE / relative):
            raise ValueError(f"child copy changed before repair: {relative}")
    comparisons = json.loads(
        (BASELINE / "REPORT_SUPPLEMENT_COMPARISONS.json").read_text()
    )["comparisons"]
    roots = {
        "original_primary/1": Path(
            pipeline["original_sources"]["original_primary"]["root"]
        ),
        "original_control/1": Path(
            pipeline["original_sources"]["original_control"]["root"]
        ),
    }
    plans = {
        protocol: json.loads((root / "matrix_plan.json").read_text())["cells"]
        for protocol, root in roots.items()
    }

    return scope_path, targets, pipeline_path, pipeline, comparisons, roots, plans


def _publish_repair_receipt(
    targets: dict,
    repairs: list,
    ledger_path: Path,
    ledger: dict,
    pipeline_path: Path,
    scope_path: Path,
) -> int:
    """Bind the selected replacements and source lineage to child receipts."""
    if len(repairs) != 147:
        raise ValueError("rendered scope differs from measured 49 triples")
    write_json(ledger_path, ledger)
    child_pipeline = json.loads(pipeline_path.read_text())
    ledger_sha = sha(ledger_path)
    child_pipeline["artifact_sha256"]["figure_ledger"] = ledger_sha
    child_pipeline["figure_ledger_sha256"] = ledger_sha
    child_pipeline["graphics_layout_repair"] = {
        "scope_sha256": sha(scope_path),
        "source_commit": "5a1fdbfe51",
        "replaced_density_triples": 49,
        "baseline_pipeline_sha256": sha(pipeline_path),
    }
    write_json(CHILD / "REPORT_SUPPLEMENT_PIPELINE.json", child_pipeline)
    receipt = {
        "schema": "supplement-density-layout-child-publication/1",
        "baseline": str(BASELINE),
        "child": str(CHILD),
        "scope_path": str(scope_path),
        "scope_sha256": sha(scope_path),
        "repair_source_commit": "5a1fdbfe51",
        "baseline_pipeline_sha256": sha(pipeline_path),
        "child_pipeline_sha256": sha(CHILD / "REPORT_SUPPLEMENT_PIPELINE.json"),
        "baseline_figure_ledger_sha256": sha(
            BASELINE / "REPORT_SUPPLEMENT_FIGURES.json"
        ),
        "child_figure_ledger_sha256": ledger_sha,
        "replacements": repairs,
    }
    receipt_path = EVIDENCE / "layout_repair_child_publication_receipt.json"
    write_json(receipt_path, receipt)
    sys.stdout.write(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "sha256": sha(receipt_path),
                "replacements": len(repairs),
            }
        )
        + "\n"
    )
    return 0


def main() -> int:
    if not BASELINE.is_dir() or not CHILD.is_dir():
        raise FileNotFoundError("baseline and copied child publication must exist")
    if not (
        REPAIR_SOURCE / "src/tools/shot_pattern_analysis/matched_supplement_figures.py"
    ).is_file():
        raise FileNotFoundError("isolated repair source is missing")
    sys.path.insert(0, str(REPAIR_SOURCE))
    from src.tools.shot_pattern_analysis.matched_figure_identity import (
        pair_comparison_slug,
    )
    from src.tools.shot_pattern_analysis.matched_supplement_figures import (
        render_selected_pair_figures,
    )

    scope_path, targets, pipeline_path, pipeline, comparisons, roots, plans = (
        _load_repair_scope()
    )

    def source_ref(src: dict) -> dict:
        protocol, cell = src["protocol_id"], src["cell"]
        root = roots[protocol]
        summary_path = root / cell / "summary.json"
        csv_path = root / cell / "shots.csv.gz"
        if sha(csv_path) != src["source_csv_sha256"]:
            raise ValueError(f"source shots changed: {protocol}:{cell}")
        return {
            "cell": cell,
            "source_protocol": protocol,
            "source_root": str(root),
            "status": "completed",
            "config": plans[protocol][cell],
            "summary": json.loads(summary_path.read_text()),
            "source_csv_sha256": src["source_csv_sha256"],
        }

    ledger_path = CHILD / "REPORT_SUPPLEMENT_FIGURES.json"
    ledger = json.loads(ledger_path.read_text())
    by_relative = {entry["relative_path"]: entry for entry in ledger["figures"]}
    if len(by_relative) != len(ledger["figures"]):
        raise ValueError("figure ledger has duplicate relative paths")
    STAGING.mkdir(parents=True, exist_ok=True)
    repairs = []
    matched = set()
    for comparison in comparisons:
        name = pair_comparison_slug(comparison) + "_density.svg"
        if name not in targets:
            continue
        matched.add(name)
        source_a, source_b = comparison["sources"]
        pair = {
            "comparison_kind": comparison["comparison_kind"],
            "target_x_m": comparison["target_x_m"],
            "first": source_ref(source_a),
            "second": source_ref(source_b),
        }
        rendered = render_selected_pair_figures(
            pair,
            STAGING,
            figure_kinds=frozenset({"density_png", "density_pdf", "density_svg"}),
        )
        if {entry["figure_kind"] for entry in rendered} != {
            "density_png",
            "density_pdf",
            "density_svg",
        }:
            raise ValueError(f"incomplete density triple: {name}")
        target = targets[name]
        if sha(Path(target["svg_path"])) != target["baseline_svg_sha256"]:
            raise ValueError(f"baseline SVG changed: {name}")
        for entry in rendered:
            relative = Path(entry["relative_path"])
            source = STAGING / relative
            destination = CHILD / "figures" / relative
            baseline = BASELINE / "figures" / relative
            if relative.as_posix() not in by_relative:
                raise ValueError(f"new figure is not in baseline ledger: {relative}")
            if sha(baseline) != by_relative[relative.as_posix()]["sha256"]:
                raise ValueError(f"baseline asset changed: {relative}")
            if sha(destination) != sha(baseline):
                raise ValueError(f"child asset changed before repair: {relative}")
            if sha(source) != entry["sha256"]:
                raise ValueError(f"render ledger mismatch: {relative}")
            os.replace(source, destination)
            by_relative[relative.as_posix()]["sha256"] = entry["sha256"]
            repairs.append(
                {
                    "relative_path": relative.as_posix(),
                    "baseline_sha256": sha(baseline),
                    "child_sha256": entry["sha256"],
                }
            )
    if matched != set(targets):
        raise ValueError("rendered scope differs from measured 49 triples")
    return _publish_repair_receipt(
        targets, repairs, ledger_path, ledger, pipeline_path, scope_path
    )


if __name__ == "__main__":
    raise SystemExit(main())
