"""Forward-dynamics neck-tracking evidence (OSV-3c, #11729) from pipeline runs.

    python3 -m scripts.summarize_fd_neck_tracking OUT.json RUN [RUN ...]

Each ``RUN`` is a pipeline run directory (``receipt.json``). The JSON keeps,
per run, the configuration (capture, gaze weight, neck reference), the marker
RMS of the IK reference and of the replay (whole swing, address to impact and
per segment) and the ``dynamics.head_gaze`` block, so the reference table is
regenerated from receipts rather than copied by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.shared.python.contracts import require


def _mm(value: float | None) -> float | None:
    return None if value is None else round(1000.0 * float(value), 2)


def run_row(run: Path) -> dict[str, Any]:
    """The evidence row of one run directory.

    Raises:
        ValueError: when the receipt lacks the forward-dynamics head-gaze block.
    """
    receipt = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
    dyn = receipt["dynamics"]
    block = dyn.get("head_gaze")
    require(block is not None, f"{run}: receipt has no dynamics.head_gaze block")
    ik = receipt["ik"]
    ik_rms = ik["reference"]["marker_rms_m"] if "reference" in ik else None
    return {
        "run": run.name,
        "capture": receipt["capture"],
        "gaze_weight": receipt["head_gaze"].get("gaze_weight"),
        "neck_reference": block.get("neck_reference"),
        "ik_marker_rms_mm": _mm(ik_rms),
        "fd_marker_rms_mm": _mm(dyn["marker_rms_m"]),
        "fd_address_to_impact_rms_mm": _mm(
            dyn["fd_phase"].get("fd_rms_address_to_impact_m")
        ),
        "fd_segment_rms_mm": {k: _mm(v) for k, v in dyn["segment_rms_m"].items()},
        "ik_head_gaze_address_to_impact": receipt["head_gaze"]["address_to_impact"],
        "fd_head_gaze": block,
    }


def summarize(runs: list[Path]) -> dict[str, Any]:
    """Evidence document over ``runs`` (sorted by capture, then run name)."""
    require(len(runs) > 0, "at least one run directory is required")
    rows = sorted((run_row(r) for r in runs), key=lambda r: (r["capture"], r["run"]))
    return {
        "issue": "#11729 (OSV-3c)",
        "note": (
            "forward-dynamics replay (MuJoCo computed-torque tracking); the "
            "gaze-schedule neck is a modelled behaviour, not measured data"
        ),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    ns = parser.parse_args(argv)
    ns.out.write_text(
        json.dumps(summarize(ns.runs), indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
