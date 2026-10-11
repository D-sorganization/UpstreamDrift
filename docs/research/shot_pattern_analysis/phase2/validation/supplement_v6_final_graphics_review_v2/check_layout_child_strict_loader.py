"""Record the frozen public loader's verdict on the derivative publication."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE = HERE.parents[1]
CHILD = PHASE / "supplement_report_v2_layout_repair"
PRODUCER = Path(
    "/home/dieterolson/.codex/worktrees/shot-pattern-presentation/UpstreamDrift"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    sys.path.insert(0, str(PRODUCER))
    from src.tools.shot_pattern_analysis.matched_supplement_pipeline import (
        load_verified_supplement_artifacts,
    )

    start = time.time()
    result = load_verified_supplement_artifacts(CHILD)
    ledger = result["figure_ledger"]
    receipt = {
        "schema": "shot-pattern-layout-child-strict-loader/1",
        "public_loader": "src.tools.shot_pattern_analysis.matched_supplement_pipeline.load_verified_supplement_artifacts",
        "producer_checkout": str(PRODUCER),
        "child": str(CHILD),
        "pipeline_sha256": sha(CHILD / "REPORT_SUPPLEMENT_PIPELINE.json"),
        "figure_ledger_sha256": sha(CHILD / "REPORT_SUPPLEMENT_FIGURES.json"),
        "figure_count": len(ledger["figures"]),
        "comparison_count": len(result["comparisons"]["comparisons"]),
        "elapsed_seconds": round(time.time() - start, 3),
        "verdict": "pass",
    }
    target = HERE / "layout_child_strict_loader_receipt.json"
    target.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    sys.stdout.write(
        json.dumps({"receipt": str(target), "sha256": sha(target), "verdict": "pass"})
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
