"""Audit fresh primary-final restoration through qualified archive/report APIs.

Run from the repository root with ``python -m`` and a new receipt path.
This reads saved data only; it neither simulates nor repeats the scientific audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from scripts.research.shot_pattern_analysis.archive_data import verify_archives
from src.tools.shot_pattern_analysis.matched_report import load_matrix_report


def sha(path: Path) -> str:
    """Hash bounded blocks without holding an archive in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def inventory(directory: Path) -> dict[str, dict[str, Any]]:
    entries = sorted(directory.rglob("*"))
    require(not any(path.is_symlink() for path in entries), "symlink in inventory")
    return {
        path.relative_to(directory.parent).as_posix(): {
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
        for path in entries
        if path.is_file()
    }


def run(output: Path) -> dict[str, Any]:
    require(not output.exists(), "receipt must be new")
    root = Path.cwd()
    phase = root / "docs/research/shot_pattern_analysis/phase2"
    original = phase / "final_v1"
    bundle = phase / "data_archives/final_primary_v1"
    audit_path = phase / "validation/independent_primary_final_audit.json"
    sources = [
        Path(__file__).resolve(),
        root / "scripts/research/shot_pattern_analysis/archive_data.py",
        root / "src/tools/shot_pattern_analysis/matched_report.py",
        root / "src/tools/shot_pattern_analysis/provenance.py",
    ]
    source_hashes = {str(p.relative_to(root)): sha(p) for p in sources}
    audit_sha = sha(audit_path)
    audit = json.loads(audit_path.read_text())
    require(audit["pass"] is True and audit["complete"] is True, "audit incomplete")
    require(audit["failures"] == [], "scientific audit failures")
    require(audit["audited_cells"] == audit["planned_cells"] == 324, "audit count")
    require(audit["data_stage"] == "final", "audit stage")
    require(sum(c["rows"] for c in audit["cells"].values()) == 8_100_000, "audit rows")
    bundle_before = inventory(bundle)
    count = verify_archives(bundle)
    manifest = json.loads((bundle / "archive_manifest.json").read_text())
    require(manifest["source_name"] == "final_v1", "wrong dataset")
    expected = {
        name: {k: item[k] for k in ("sha256", "bytes")}
        for name, item in manifest["files"].items()
    }
    require(inventory(original) == expected, "original differs from archive inventory")
    original_report = load_matrix_report(original, require_complete=True)
    require(
        original_report["counts"]
        == {"completed": 270, "unavailable": 54, "not_run": 0},
        "matrix status counts",
    )
    require(original_report["planned_cells"] == 324, "matrix inventory")
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
    with tempfile.TemporaryDirectory(prefix="p2-primary-final-restore-") as temporary:
        destination = Path(temporary)
        require(not any(destination.iterdir()), "restoration directory not fresh")
        # verify_archives above validates member paths, regular-file types and hashes.
        for part in manifest["archives"]:
            with ZipFile(bundle / part) as stream:
                stream.extractall(destination)
        require({p.name for p in destination.iterdir()} == {"final_v1"}, "extra root")
        require(inventory(destination / "final_v1") == expected, "restored file bytes")
        restored = load_matrix_report(destination / "final_v1", require_complete=True)
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
        "schema": "shot-study-primary-final-restoration/1",
        "status": "passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hashes,
        "scientific_audit": {
            "path": str(audit_path.relative_to(root)),
            "sha256": audit_sha,
            "rows": 8_100_000,
        },
        "bundle_files": bundle_before,
        "files_restored": count,
        "archive_manifest_sha256": sha(bundle / "archive_manifest.json"),
        "plan_sha256": original_report["plan_sha256"],
        "matrix_counts": original_report["counts"],
        "planned_cells": 324,
        "fresh_directory": True,
        "temporary_directory_removed": True,
        "all_file_hashes_and_sizes_equal": True,
        "full_restored_report_equals_original": True,
        "audit_plan_execution_cells_statuses_bound": True,
        "sources_originals_bundles_audit_unchanged": True,
        "limitations": [
            "Packaging/restoration acceptance only; scientific acceptance is the separately hash-bound prior audit.",
            "No simulations, scoring recomputation, control or supplemental dataset acceptance, or visual review performed.",
        ],
    }
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    result = run(args.receipt)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
