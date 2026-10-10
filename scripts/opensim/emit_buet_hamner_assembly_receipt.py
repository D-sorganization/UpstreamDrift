"""Emit a source-bound BUET–Hamner native assembly diagnostic receipt."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from src.engines.physics_engines.opensim.python.native_buet_hamner_assembly import (
    BuetHamnerAssemblyRequest,
    assemble_buet_hamner,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buet-source", required=True, type=Path)
    parser.add_argument("--buet-sha256", required=True)
    parser.add_argument("--hamner-source", required=True, type=Path)
    parser.add_argument("--hamner-sha256", required=True)
    parser.add_argument("--derived-output", required=True, type=Path)
    parser.add_argument("--receipt-output", required=True, type=Path)
    args = parser.parse_args()
    if args.receipt_output.exists():
        raise FileExistsError(args.receipt_output)
    receipt = assemble_buet_hamner(
        BuetHamnerAssemblyRequest(
            args.buet_source,
            args.buet_sha256,
            args.hamner_source,
            args.hamner_sha256,
            args.derived_output,
        )
    )
    args.receipt_output.write_text(
        json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
