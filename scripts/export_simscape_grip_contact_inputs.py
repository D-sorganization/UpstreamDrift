#!/usr/bin/env python3
"""Export Simscape physical grip contact scenario inputs JSON (HRG-GS-M12, #13334).

Writes ``gs3dx_grip_contact_scenarios.json`` under
``exploratory_gs3dx/tools/generated/`` containing the ``hold`` and
``short_swing`` scenario inputs (hand trajectories, Hermite spline coefficients,
club parameters, initial state, solver settings, and topology mapping hashes)
for ``overlap``, ``interlock``, and ``ten_finger``.

Usage:
    python scripts/export_simscape_grip_contact_inputs.py
    python scripts/export_simscape_grip_contact_inputs.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.shared.python.grip_contact.simscape_grip_contact import (
    DEFAULT_SCENARIOS_JSON_PATH,
    export_scenarios_json,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export or check Simscape physical grip contact scenario inputs JSON."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the generated scenarios JSON matches the Python source of truth.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SCENARIOS_JSON_PATH,
        help="Path to gs3dx_grip_contact_scenarios.json.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rc = export_scenarios_json(path=args.output, check=bool(args.check))
    if rc != 0:
        print(
            "ERROR: Simscape grip contact scenarios JSON is out of date. "
            "Run `python scripts/export_simscape_grip_contact_inputs.py`.",
            file=sys.stderr,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
