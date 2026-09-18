"""OS-7: merge several driver receipts of one ladder into the top-level receipt.

Usage::

    python os7_merge_receipts.py --evidence <dir> --headline run2 \
        run1=receipt_run1.json run2=receipt_run2.json

Every number in the output comes from a driver receipt (the headline run's
document is copied verbatim; ``per_horizon`` is the concatenation of every
run's rows tagged with the run name). Pure Python, runs anywhere.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engines.physics_engines.opensim.python.tour_matching.moco_g1 import (  # noqa: E402
    merge_ladder_receipts,
    validate_os7_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--headline", required=True)
    parser.add_argument("runs", nargs="+", help="name=receipt.json, ladder order")
    args = parser.parse_args()
    runs = []
    for item in args.runs:
        name, _, filename = item.partition("=")
        document = json.loads((args.evidence / filename).read_text(encoding="utf-8"))
        runs.append((name, document))
    merged = merge_ladder_receipts(runs, headline=args.headline)
    validate_os7_receipt(merged)
    out = args.evidence / "receipt.json"
    out.write_text(json.dumps(merged, indent=2, default=str) + "\n", encoding="utf-8")
    print(out, "rows:", len(merged["per_horizon"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
