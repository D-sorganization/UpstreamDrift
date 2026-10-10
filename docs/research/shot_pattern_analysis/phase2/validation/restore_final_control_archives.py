"""Independently restore control-final archives and bind the completed audit.

Run from the repository root via ``python -m`` with a new receipt path.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from docs.research.shot_pattern_analysis.phase2.validation import (
    restore_final_primary_archives as primary_helpers,
)
from scripts.research.shot_pattern_analysis.archive_data import verify_archives
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
        root / "scripts/research/shot_pattern_analysis/archive_data.py",
        root / "src/tools/shot_pattern_analysis/matched_report.py",
        root / "src/tools/shot_pattern_analysis/provenance.py",
    ]
    source_hashes = {str(p.relative_to(root)): sha(p) for p in sources}
    audit_sha = sha(audit_path)
    audit = json.loads(audit_path.read_text())
    require(audit["pass"] is True and audit["complete"] is True, "audit incomplete")
    require(audit["failures"] == [], "scientific audit failures")
    require(audit["audited_cells"] == audit["planned_cells"] == 72, "audit count")
    require(audit["data_stage"] == "final", "audit stage")
    rows = sum(c["rows"] for c in audit["cells"].values())
    require(rows == 1_560_000, "audit rows")
    bundle_before = inventory(bundle)
    count = verify_archives(bundle)
    manifest = json.loads((bundle / "archive_manifest.json").read_text())
    require(manifest["source_name"] == "control_final_v1", "wrong dataset")
    expected = {
        name: {k: item[k] for k in ("sha256", "bytes")}
        for name, item in manifest["files"].items()
    }
    require(inventory(original) == expected, "original differs from archive inventory")
    original_report = load_matrix_report(original, require_complete=True)
    require(
        original_report["counts"] == {"completed": 52, "unavailable": 20, "not_run": 0},
        "matrix status counts",
    )
    require(original_report["planned_cells"] == 72, "matrix inventory")
    require(original_report["stage"] == "final", "matrix stage")
    require(
        original_report["plan_sha256"] == audit["plan_sha256"], "audit plan binding"
    )
    require(
        original_report["execution"] == audit["execution"], "audit execution binding"
    )
    require(
        original_report["cell_manifest_sha256"]
        == {name: cell["manifest_sha256"] for name, cell in audit["cells"].items()},
        "audit cell binding",
    )
    require(
        {c["cell"]: c["status"] for c in original_report["cells"]}
        == {name: cell["status"] for name, cell in audit["cells"].items()},
        "audit status binding",
    )
    with tempfile.TemporaryDirectory(prefix="p2-control-final-restore-") as temporary:
        destination = Path(temporary)
        require(not any(destination.iterdir()), "restoration directory not fresh")
        # The qualified verifier checks safe names, regular-file types and payloads.
        for part in manifest["archives"]:
            with ZipFile(bundle / part) as stream:
                stream.extractall(destination)
        require(
            {p.name for p in destination.iterdir()} == {"control_final_v1"},
            "extra root",
        )
        require(
            inventory(destination / "control_final_v1") == expected, "restored bytes"
        )
        restored = load_matrix_report(
            destination / "control_final_v1", require_complete=True
        )
        require(restored == original_report, "full restored report differs")
    require(not destination.exists(), "temporary restoration not cleaned")
    require(inventory(original) == expected, "original changed during review")
    require(verify_archives(bundle) == count, "archive count changed")
    require(inventory(bundle) == bundle_before, "bundle changed during review")
    require(sha(audit_path) == audit_sha, "audit receipt changed")
    require(
        {str(p.relative_to(root)): sha(p) for p in sources} == source_hashes,
        "review source changed",
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
