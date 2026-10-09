"""Integrity-checked assembly of matched-study reports without rerunning physics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .provenance import assert_source_unchanged, manifest_files_match


def load_matrix_report(root: Path, *, require_complete: bool = True) -> dict:
    """Load every planned cell; reject tampering and preserve unavailable outcomes.

    Reports bind to the saved execution snapshot, not the current installation.
    Missing cells can be shown only when explicitly requesting a partial report.
    """
    root = Path(root)
    plan_path = root / "matrix_plan.json"
    plan = json.loads(plan_path.read_text())
    if not isinstance(plan.get("cells"), dict) or not plan["cells"]:
        raise ValueError("matrix plan must contain explicit cells")
    cells, manifests = [], {}
    for name, config in plan["cells"].items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("matrix cell name must be a simple directory name")
        directory = root / name
        manifest = directory / "manifest.json"
        if not directory.exists():
            cells.append({"cell": name, "config": config, "status": "not_run"})
            continue
        if not manifest_files_match(manifest):
            raise ValueError(f"cell integrity failure: {name}")
        start = json.loads((directory / "run_start.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        if start["config"] != config or summary.get("config") != config:
            raise ValueError(f"cell input mismatch: {name}")
        assert_source_unchanged(plan["execution"], start["execution"])
        if summary.get("status") not in ("completed", "unavailable"):
            raise ValueError(f"unrecognized cell status: {name}")
        if summary["status"] == "completed":
            for item in summary["summary"].values():
                if item["total_n"] != config["n_shots"]:
                    raise ValueError(f"population count mismatch: {name}")
        cells.append({**summary, "cell": name})
        manifests[name] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    counts = dict.fromkeys(("completed", "unavailable", "not_run"), 0)
    counts.update(Counter(cell["status"] for cell in cells))
    if require_complete and counts["not_run"]:
        raise ValueError("matrix incomplete: planned cells have not run")
    return {
        "schema": "matched-study-report/1",
        "stage": plan["stage"],
        "execution": plan["execution"],
        "planned_cells": len(cells),
        "counts": counts,
        "cells": cells,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "cell_manifest_sha256": manifests,
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "conditional geometric simulation; not empirical qualification",
    }
