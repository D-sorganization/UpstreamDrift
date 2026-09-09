"""Validate a proposed offline catalog and print its deterministic review diff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.shared.python.club_data.catalog_io import MAX_EXCHANGE_BYTES, import_json
from src.shared.python.club_data.catalog_sources import (
    catalog_diff,
    load_public_catalog,
)


def main() -> int:
    """Read and validate only; never update a player's catalog or capture files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate", type=Path, nargs="?", help="Proposed versioned catalog JSON"
    )
    args = parser.parse_args()
    current = load_public_catalog()
    if args.candidate is None:
        print(
            f"Validated {len(current)} offline club builds with attributed source claims."
        )
        return 0
    if args.candidate.stat().st_size > MAX_EXCHANGE_BYTES:
        parser.error("Candidate exceeds the 8 MiB exchange limit")
    candidate = import_json(args.candidate.read_text(encoding="utf-8"))
    print(
        json.dumps(
            [
                change.model_dump(mode="json")
                for change in catalog_diff(current, candidate)
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
