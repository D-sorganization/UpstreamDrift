"""Independently restore control-final archives and bind the completed audit.

Run from the repository root via ``python -m`` with a new receipt path.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from docs.research.shot_pattern_analysis.phase2.validation import (
    restore_final_primary_archives as primary_helpers,
)
from docs.research.shot_pattern_analysis.phase2.validation.restore_archive_common import (
    RestorationSnapshot,
    assert_restoration_unchanged,
    restore_archive_and_compare,
    validate_complete_audit,
    validate_report_binding,
    verify_archive_bundle,
)
from src.tools.shot_pattern_analysis.matched_report import load_matrix_report

inventory = primary_helpers.inventory
require = primary_helpers.require
sha = primary_helpers.sha


def run(output: Path) -> dict[str, Any]:
    require(not output.exists(), "receipt must be new")
    root = Path.cwd()
    phase = root / "docs/research/shot_pattern_analysis/phase2"
    original = phase / "control_final_v1"
    bundle = phase / "data_archives/final_control_v1"
    audit_path = phase / "validation/independent_control_final_audit.json"
    sources = [
        Path(__file__).resolve(),
        Path(primary_helpers.__file__).resolve(),
        root
        / "docs/research/shot_pattern_analysis/phase2/validation/restore_archive_common.py",
        root / "scripts/research/shot_pattern_analysis/archive_data.py",
        root / "src/tools/shot_pattern_analysis/matched_report.py",
        root / "src/tools/shot_pattern_analysis/provenance.py",
    ]
    source_hashes = {str(p.relative_to(root)): sha(p) for p in sources}
    audit_sha = sha(audit_path)
    audit = json.loads(audit_path.read_text())
    rows = validate_complete_audit(audit, 72, 1560000)
    bundle_before = inventory(bundle)
    verified_bundle = verify_archive_bundle(bundle)
    count = verified_bundle.file_count
    manifest = verified_bundle.manifest
    require(manifest["source_name"] == "control_final_v1", "wrong dataset")
    expected = {
        name: {k: item[k] for k in ("sha256", "bytes")}
        for name, item in manifest["files"].items()
    }
    require(inventory(original) == expected, "original differs from archive inventory")
    original_report = load_matrix_report(original, require_complete=True)
    validate_report_binding(
        original_report,
        audit,
        {"completed": 52, "unavailable": 20, "not_run": 0},
        72,
    )
    restore_archive_and_compare(
        verified_bundle,
        "control_final_v1",
        expected,
        "p2-control-final-restore-",
        lambda path: load_matrix_report(path, require_complete=True),
        original_report,
        "restored bytes",
    )
    assert_restoration_unchanged(
        RestorationSnapshot(
            root=root,
            original=original,
            original_inventory=expected,
            bundle=bundle,
            bundle_inventory=bundle_before,
            archive_count=count,
            audit_path=audit_path,
            audit_sha256=audit_sha,
            sources=tuple(sources),
            source_sha256=source_hashes,
        )
    )
    result = {
        "schema": "shot-study-control-final-restoration/1",
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes,
        "scientific_audit": {
            "path": str(audit_path.relative_to(root)),
            "sha256": audit_sha,
            "rows": rows,
        },
        "bundle_files": bundle_before,
        "files_restored": count,
        "archive_manifest_sha256": sha(bundle / "archive_manifest.json"),
        "plan_sha256": original_report["plan_sha256"],
        "matrix_counts": original_report["counts"],
        "planned_cells": 72,
        "fresh_directory": True,
        "temporary_directory_removed": True,
        "all_file_hashes_and_sizes_equal": True,
        "full_restored_report_equals_original": True,
        "audit_plan_execution_cells_statuses_bound": True,
        "sources_originals_bundles_audit_unchanged": True,
        "limitations": [
            "Packaging/restoration acceptance only; scientific acceptance is the separately hash-bound completed audit.",
            "No simulations, scoring recomputation, supplemental dataset acceptance, or visual review performed.",
        ],
    }
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    sys.stdout.write(json.dumps(run(args.receipt), indent=2) + "\n")


if __name__ == "__main__":
    main()
