"""Inventory a candidate transition; never issue simulation or comparison acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PACKAGE = "src/tools/shot_pattern_analysis/"
ALLOWED = {
    PACKAGE + name
    for name in (
        "matched_runner.py",
        "matched_matrix.py",
        "matched_supplement.py",
        "matched_export.py",
        "nominal_matching.py",
        "density.py",
        "academic_plots.py",
        "matched_report_plots.py",
        "provenance.py",
        "scenario_scoring.py",
        "course_score_vectors.py",
    )
}
LEGACY = "src/shared/python/physics/impact_model/utils.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new candidate inventory output")
    old = json.loads(args.acceptance.read_text())["execution"]
    new = json.loads(args.snapshot.read_text())
    old_map, new_map = old["source_sha256"], new["source_sha256"]
    changed = {
        path: {
            "old_sha256": old_map.get(path),
            "new_sha256": new_map.get(path),
            "classification": "unreachable_legacy"
            if path == LEGACY
            else "approved_nonprojection",
        }
        for path in sorted(old_map.keys() | new_map.keys())
        if old_map.get(path) != new_map.get(path)
    }
    disallowed = sorted(set(changed) - ALLOWED - {LEGACY})
    other_equal = {k: v for k, v in old.items() if k != "source_sha256"} == {
        k: v for k, v in new.items() if k != "source_sha256"
    }
    stale = [
        path
        for path, digest in new_map.items()
        if not (args.candidate_root / path).is_file()
        or sha(args.candidate_root / path) != digest
    ]
    native = Path(new["native_binary_path"])
    native_verified = native.is_file() and sha(native) == new["native_binary_sha256"]
    result = {
        "schema": "matched-delivery-transition-preparation/1",
        "status": "pending_independent_qualification_not_acceptance",
        "qualified_scope_requested": "matched-delivery-comparisons",
        "old_execution": old,
        "new_execution": new,
        "changed_sources": changed,
        "disallowed_changed_paths": disallowed,
        "unchanged_non_source_execution": other_equal,
        "candidate_snapshot_stale_paths": stale,
        "native_binary_verified": native_verified,
        "inventory_checks_pass": not disallowed
        and other_equal
        and not stale
        and native_verified,
        "acceptance_sha256": sha(args.acceptance),
        "candidate_snapshot_sha256": sha(args.snapshot),
        "script_sha256": sha(Path(__file__)),
        "remaining": [
            "Stable final source snapshot",
            "Per-change evidence qualification",
            "Focused software and rendered-figure gates",
            "Three qualified supplemental pilots",
            "Deterministic supplemental refinement",
            "Independent final acceptance",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
