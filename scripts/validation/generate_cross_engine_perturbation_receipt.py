"""Generate the unified cross-engine perturbation receipt (SAP-6, #13795).

Runs :func:`run_unified_study` (in-house dynamics and MuJoCo on the canonical
triple pendulum) and writes the validated receipt JSON. ``--check`` regenerates
and fails when the committed receipt's CVs no longer match.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.shared.python.pendulum_simulator.unified_perturbation_study import (
    receipt_cvs_match,
    run_unified_study,
)

DEFAULT_RECEIPT = Path("docs/validation/cross_engine_perturbation_receipt.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    receipt = run_unified_study()
    if args.check:
        committed = json.loads(args.output.read_text(encoding="utf-8"))
        if not receipt_cvs_match(receipt, committed):
            print(f"{args.output} is stale; regenerate it", file=sys.stderr)
            return 1
        return 0
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({receipt['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
