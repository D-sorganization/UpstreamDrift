#!/usr/bin/env python3
"""Compare two ground-support receipt.json files under canonical JSON (#14771).

Exits 0 when canonical UTF-8 payloads match, 1 otherwise.

Example::

    python3 scripts/ground_support/check_receipt_canonical_match.py \\
        path/to/receipt_a.json path/to/receipt_b.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.shared.python.motion_matching.receipt_identity import (  # noqa: E402
    canonical_receipt_bytes,
    load_receipt_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt_a", type=Path, help="First receipt.json path")
    parser.add_argument("receipt_b", type=Path, help="Second receipt.json path")
    args = parser.parse_args()
    a = load_receipt_json(args.receipt_a)
    b = load_receipt_json(args.receipt_b)
    if canonical_receipt_bytes(a) == canonical_receipt_bytes(b):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
