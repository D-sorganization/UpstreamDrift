import sys

"""Save a compact receipt for the frozen public supplement loader."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.tools.shot_pattern_analysis.matched_supplement_pipeline import (
    load_verified_supplement_artifacts,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


phase = Path(
    "/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/"
    "docs/research/shot_pattern_analysis/phase2"
)
output = phase / "validation/supplement_v6_final_graphics_review_v2"
terminal_path = phase / "validation/supplement_rendering_revision_v6/exit.json"
terminal = json.loads(terminal_path.read_text())
if (terminal.get("process_exit_code"), terminal.get("status")) != (
    0,
    "completed_pending_full_independent_acceptance",
):
    raise ValueError("producer terminal gate did not pass")

result = load_verified_supplement_artifacts(phase / "supplement_report_v1")
comparisons = result["comparisons"]["comparisons"]
cells = result["report"]["unique_cells"]
figures = result["figure_ledger"]["figures"]
paths = [row["relative_path"] for row in figures if row.get("status") != "unavailable"]
numeric_path = (
    phase / "validation/protocol_name_recovery_v6/expected_numeric_hashes.json"
)
expected = json.loads(numeric_path.read_text())
actual_numeric = {name: sha(phase / "supplement_report_v1" / name) for name in expected}
if actual_numeric != expected:
    raise ValueError("seven numerical hashes differ from committed reference")
if not (
    len(comparisons) == 384
    and len(cells) == 26
    and len(figures) == len(paths) == len(set(paths)) == 6456
    and Counter(Path(path).suffix for path in paths)
    == Counter({".png": 2408, ".pdf": 2024, ".svg": 2024})
    and Counter(row["comparison_kind"] for row in comparisons)
    == Counter({"across_lean": 56, "curve_strength": 164, "shaft_rotation_sd": 164})
):
    raise ValueError("completed numerical/graphics inventory differs from review plan")

receipt = {
    "scope": "Strict Public Loader and Numeric/Inventory Gate; Visual Acceptance Pending",
    "status": "passed",
    "finished_utc": datetime.now(timezone.utc).isoformat(),
    "python": "/tmp/shot-pattern-validation-venv-20261009/bin/python",  # nosec B108
    "producer_checkout": "/home/dieterolson/.codex/worktrees/shot-pattern-presentation/UpstreamDrift",
    "producer_commit": terminal["candidate_commit"],
    "public_loader": "src.tools.shot_pattern_analysis.matched_supplement_pipeline.load_verified_supplement_artifacts",
    "terminal_path": str(terminal_path),
    "terminal_sha256": sha(terminal_path),
    "pipeline_receipt_sha256": terminal["pipeline_receipt_sha256"],
    "ledger_sha256": sha(phase / "supplement_report_v1/REPORT_SUPPLEMENT_FIGURES.json"),
    "expected_numeric_reference_path": str(numeric_path),
    "expected_numeric_reference_sha256": sha(numeric_path),
    "numeric_sha256": actual_numeric,
    "comparison_counts": dict(Counter(row["comparison_kind"] for row in comparisons)),
    "unique_cells": len(cells),
    "figure_count": len(figures),
    "figure_formats": dict(Counter(Path(path).suffix for path in paths)),
    "unique_normalized_paths": len(set(paths)),
    "source_sha256": terminal["producer_sha256"],
    "input_sha256": terminal["input_sha256"],
}
(output / "strict_loader_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
sys.stdout.write(
    json.dumps(
        {
            "status": "passed",
            "pairs": len(comparisons),
            "cells": len(cells),
            "assets": len(figures),
        }
    )
    + "\n"
)
