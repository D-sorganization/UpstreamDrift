#!/usr/bin/env python3
"""Run the lift-pack parity baseline and write the Markdown doc and JSON receipt.

Usage: python3 scripts/lifting/run_pack_parity_baseline.py [--out-dir DIR]
Needs the four ``*_Models`` checkouts (see ``LIFT_PACK_ROOT``) and the engines.

``--from-receipt PATH`` skips the engines: it re-derives the gaps and re-renders
both documents from an existing condensed receipt, so analysis or report changes
can be applied without re-measuring. Measured values are copied, never edited.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from src.shared.python.lifting.pack_audit.baseline import run_baseline  # noqa: E402
from src.shared.python.lifting.pack_audit.gaps import derive_gaps  # noqa: E402
from src.shared.python.lifting.pack_audit.model import Anthropometry  # noqa: E402
from src.shared.python.lifting.pack_audit.report import (  # noqa: E402
    condense,
    render_markdown,
)

logger = logging.getLogger("lift_pack_baseline")


def main(argv: list[str] | None = None) -> int:
    """Write ``PACK_PARITY_BASELINE.md`` and ``pack_parity_baseline.json``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs/development/lifting")
    ap.add_argument("--body-mass", type=float, default=80.0)
    ap.add_argument("--height", type=float, default=1.78)
    ap.add_argument("--plate-per-side", type=float, default=50.0)
    ap.add_argument(
        "--full-receipt", type=Path, help="also write the uncondensed receipt"
    )
    ap.add_argument(
        "--from-receipt",
        type=Path,
        help="re-render from this condensed receipt instead of running the engines",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.from_receipt:
        receipt = _load_receipt(args.from_receipt)
    else:
        anthro = Anthropometry(args.body_mass, args.height, args.plate_per_side)
        receipt = run_baseline(anthro)
        if args.full_receipt:
            args.full_receipt.write_text(
                json.dumps(receipt, indent=1), encoding="utf-8"
            )
    receipt["gaps"] = derive_gaps(receipt)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "PACK_PARITY_BASELINE.md").write_text(
        render_markdown(receipt), encoding="utf-8"
    )
    (args.out_dir / "pack_parity_baseline.json").write_text(
        json.dumps(condense(receipt), indent=1) + "\n", encoding="utf-8"
    )
    logger.info("wrote %s", args.out_dir)
    return 0


def _load_receipt(path: Path) -> dict[str, Any]:
    """Read a committed receipt; it must already be condensed (no pose data)."""
    if not path.is_file():
        raise FileNotFoundError(f"receipt not found: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or "results" not in receipt:
        raise ValueError(f"{path} is not a pack-parity receipt (no 'results')")
    if condense(receipt) != receipt:
        raise ValueError(f"{path} is not condensed; re-run with the engines")
    return receipt


if __name__ == "__main__":
    raise SystemExit(main())
