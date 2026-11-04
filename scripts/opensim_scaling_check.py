"""Run the OpenSim segment scaling check on a canonical tour capture (MATCH #14444).

Reads the C3D in a clean child process (``ezc3d`` and ``opensim`` cannot share a
process, #12065), gates every segment on the #13991 length-plausibility bounds and
writes ``scaling_check_metrics.json`` plus a run-manifest-v1 ``run_manifest.json``.

Usage:
    python3 scripts/opensim_scaling_check.py data/C3D_TA_Driver.c3d --out <dir>
        [--ud-commit <sha>] [--trend <jsonl> [--date YYYY-MM-DD]]

With ``--trend`` the run also appends a dated record to the model-matching trend
(``scripts/model_matching_trend.py append``, #14446). It is opt-in because the
trend file is tracked and accepts one record per date.

Exit code 0 means every segment is plausible, 1 means at least one failed.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
import logging
from pathlib import Path
import subprocess
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.model_matching_trend import main as trend_main  # noqa: E402
from scripts.diagnostics.frozen_capture_trc import (  # noqa: E402
    load_transferred_capture,
    prepare_capture_in_clean_process,
)
from src.engines.physics_engines.opensim.python.tour_matching.scale import (  # noqa: E402
    DEFAULT_NOMINAL_LENGTHS_M,
)
from src.engines.physics_engines.opensim.python.tour_matching.scaling_check import (  # noqa: E402
    run_scaling_check,
    sha256_file,
    write_scaling_check,
)
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    TourCapture,
)

logger = logging.getLogger(__name__)


def load_capture(c3d: Path) -> TourCapture:
    """Decode ``c3d`` in a child process that never imports ``opensim``."""
    with tempfile.TemporaryDirectory() as scratch:
        trc = Path(scratch) / "capture.trc"
        receipt = prepare_capture_in_clean_process(
            c3d, trc, python=Path(sys.executable)
        )
        return load_transferred_capture(trc, receipt)


def _head_commit() -> str:
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "rev-parse", "HEAD"],  # noqa: S607 - git on PATH
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return completed.stdout.strip()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scaling check and write metrics plus manifest; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("c3d", type=Path, help="canonical tour capture C3D")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--ud-commit", help="UpstreamDrift commit (default: HEAD)")
    parser.add_argument("--trend", type=Path, help="trend JSONL to append a record to")
    parser.add_argument("--date", default=date.today().isoformat(), help="trend date")
    args = parser.parse_args(argv)

    metrics = run_scaling_check(
        load_capture(args.c3d),
        capture_sha256=sha256_file(args.c3d),
        nominal_lengths_m=DEFAULT_NOMINAL_LENGTHS_M,
    )
    metrics_path, _ = write_scaling_check(
        metrics, args.out, ud_commit=args.ud_commit or _head_commit()
    )
    logger.info("scaling check %s: %s", metrics.verdict, metrics_path)
    if args.trend is not None:
        trend_main(["append", "--trend", str(args.trend), "--date", args.date])
    for name in metrics.failed_segments:
        logger.warning("%s: %s", name, "; ".join(metrics.segments[name].reasons))
    return 0 if metrics.verdict == "pass" else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
