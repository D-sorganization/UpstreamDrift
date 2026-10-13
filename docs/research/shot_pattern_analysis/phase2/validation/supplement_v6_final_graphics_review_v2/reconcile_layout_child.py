"""Prove selective figure replacement and unchanged numeric publication bytes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
BASELINE = PHASE / "supplement_report_v1"
CHILD = PHASE / "supplement_report_v2_layout_repair"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_figure_changes(
    scope_path: Path, publication: dict
) -> tuple[set[str], dict[str, dict], list[dict]]:
    """Prove the selected figure inventory and replacement bytes."""
    scope = json.loads(scope_path.read_text())["affected"]
    expected = {
        "comparisons/" + Path(item["svg_path"]).with_suffix(suffix).name
        for item in scope
        for suffix in (".png", ".pdf", ".svg")
    }
    if len(expected) != 147:
        raise ValueError("scope has duplicate figure paths")
    base_ledger = json.loads((BASELINE / "REPORT_SUPPLEMENT_FIGURES.json").read_text())
    child_ledger = json.loads((CHILD / "REPORT_SUPPLEMENT_FIGURES.json").read_text())
    base_by_path = {x["relative_path"]: x for x in base_ledger["figures"]}
    child_by_path = {x["relative_path"]: x for x in child_ledger["figures"]}
    if len(base_by_path) != 6456 or set(base_by_path) != set(child_by_path):
        raise ValueError("figure inventory changed")
    ledger_diffs = {k for k in base_by_path if base_by_path[k] != child_by_path[k]}
    if ledger_diffs != expected:
        raise ValueError("changed ledger entries differ from selected 147")
    changed = []
    for relative in sorted(expected):
        base = BASELINE / "figures" / relative
        child = CHILD / "figures" / relative
        base_hash = sha(base)
        child_hash = sha(child)
        if (
            base_hash != base_by_path[relative]["sha256"]
            or child_hash != child_by_path[relative]["sha256"]
        ):
            raise ValueError(f"figure ledger mismatch: {relative}")
        if base_hash == child_hash:
            raise ValueError(f"selected figure did not change: {relative}")
        changed.append(
            {
                "relative_path": relative,
                "baseline_sha256": base_hash,
                "child_sha256": child_hash,
            }
        )
    reported = {r["relative_path"]: r for r in publication["replacements"]}
    if len(reported) != 147 or any(reported[r["relative_path"]] != r for r in changed):
        raise ValueError("publication receipt differs from selected replacements")
    return expected, base_by_path, changed


def main() -> int:
    publication = json.loads(
        (HERE / "layout_repair_child_publication_receipt.json").read_text()
    )
    scope_path = HERE / "density_layout_repair_scope.json"
    expected, base_by_path, changed = _validated_figure_changes(scope_path, publication)
    base_pipeline = json.loads(
        (BASELINE / "REPORT_SUPPLEMENT_PIPELINE.json").read_text()
    )
    child_pipeline = json.loads((CHILD / "REPORT_SUPPLEMENT_PIPELINE.json").read_text())
    numeric = {}
    for key, relative in base_pipeline["artifact_paths"].items():
        if key == "figure_ledger":
            continue
        base_hash = sha(BASELINE / relative)
        child_hash = sha(CHILD / relative)
        if (
            base_hash != child_hash
            or base_hash != base_pipeline["artifact_sha256"][key]
        ):
            raise ValueError(f"non-figure report artifact changed: {relative}")
        numeric[key] = {"relative_path": relative, "sha256": base_hash}
    if child_pipeline["artifact_sha256"]["figure_ledger"] != sha(
        CHILD / "REPORT_SUPPLEMENT_FIGURES.json"
    ):
        raise ValueError("child pipeline does not bind its ledger")
    child_without_repair = dict(child_pipeline)
    repair = child_without_repair.pop("graphics_layout_repair")
    child_without_repair["artifact_sha256"] = dict(
        child_without_repair["artifact_sha256"]
    )
    child_without_repair["artifact_sha256"]["figure_ledger"] = base_pipeline[
        "artifact_sha256"
    ]["figure_ledger"]
    child_without_repair["figure_ledger_sha256"] = base_pipeline["figure_ledger_sha256"]
    if child_without_repair != base_pipeline:
        raise ValueError(
            "child pipeline altered fields beyond figure ledger and repair lineage"
        )
    if repair["scope_sha256"] != sha(scope_path):
        raise ValueError("repair scope hash mismatch")
    receipt = {
        "schema": "shot-pattern-layout-child-reconciliation/1",
        "baseline_pipeline_sha256": sha(BASELINE / "REPORT_SUPPLEMENT_PIPELINE.json"),
        "child_pipeline_sha256": sha(CHILD / "REPORT_SUPPLEMENT_PIPELINE.json"),
        "baseline_ledger_sha256": sha(BASELINE / "REPORT_SUPPLEMENT_FIGURES.json"),
        "child_ledger_sha256": sha(CHILD / "REPORT_SUPPLEMENT_FIGURES.json"),
        "scope_sha256": sha(scope_path),
        "figure_count": len(base_by_path),
        "changed_figure_count": len(expected),
        "unchanged_figure_count": len(base_by_path) - len(expected),
        "unchanged_report_artifacts": numeric,
        "changed_figures": changed,
        "verdict": "pass",
    }
    target = HERE / "layout_child_reconciliation_receipt.json"
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    sys.stdout.write(
        json.dumps({"receipt": str(target), "sha256": sha(target), "verdict": "pass"})
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
