"""Verify exact restoration of existing pilots; write a new receipt path argument."""

import hashlib
import json
import tempfile
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile
from scripts.research.shot_pattern_analysis.archive_data import verify_archives
from src.tools.shot_pattern_analysis.matched_report import load_matrix_report

root = Path.cwd()
phase = root / "docs/research/shot_pattern_analysis/phase2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


result: dict[str, Any] = {
    "schema": "shot-study-restoration-review/1",
    "status": "pending",
    "scope": "Existing 32-shot pilot archives only; no scientific recomputation or final-run acceptance",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "datasets": {},
    "source_sha256": {
        str(p): sha(root / p)
        for p in map(
            Path,
            [
                "scripts/research/shot_pattern_analysis/archive_data.py",
                "src/tools/shot_pattern_analysis/matched_report.py",
                "src/tools/shot_pattern_analysis/provenance.py",
            ],
        )
    },
}
with tempfile.TemporaryDirectory(prefix="p2-pilot-restore-audit-") as tmp:
    destination = Path(tmp)
    for name in ("pilot_v1", "control_pilot_v1"):
        bundle = phase / "data_archives" / name
        manifest_path = bundle / "archive_manifest.json"
        manifest_sha = sha(manifest_path)
        count = verify_archives(bundle)
        manifest = json.loads(manifest_path.read_text())
        for part in manifest["archives"]:
            with ZipFile(bundle / part) as stream:
                stream.extractall(destination)
        assert {
            p.relative_to(destination).as_posix()
            for p in (destination / name).rglob("*")
            if p.is_file()
        } == set(manifest["files"])
        for member, expected in manifest["files"].items():
            path = destination / member
            assert path.stat().st_size == expected["bytes"], member
            assert sha(path) == expected["sha256"], member
        restored = load_matrix_report(destination / name, require_complete=True)
        original = load_matrix_report(phase / name, require_complete=True)
        assert restored == original, name
        assert sha(manifest_path) == manifest_sha
        assert verify_archives(bundle) == count
        result["datasets"][name] = {
            "files_restored": count,
            "archive_manifest_sha256": manifest_sha,
            "archive_parts": manifest["archives"],
            "matrix_counts": restored["counts"],
            "planned_cells": restored["planned_cells"],
            "restored_report_equals_local_report": True,
            "all_restored_file_bytes_match": True,
            "archive_verified_before_and_after": True,
        }
result["source_sha256"][str(Path(__file__).resolve().relative_to(root))] = sha(
    Path(__file__)
)
result["status"] = "passed"
result["limitations"] = [
    "This verifies packaging, restoration and source-bound report loading; prior independent scientific pilot audits remain separate.",
    "Final and supplemental datasets are not yet complete or archived.",
]
out = Path(sys.argv[1])
assert not out.exists()
out.write_text(json.dumps(result, indent=2) + "\n")
sys.stdout.write(json.dumps(result, indent=2) + "\n")
