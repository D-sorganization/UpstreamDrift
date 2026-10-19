#!/usr/bin/env python3
"""Generate the canonical Simscape GolfSwing3D grip topology JSON (HRG-GS-M11, #13333).

Writes ``gs3dx_grip_topology.json`` next to ``gs3dx_appearance.v1.json`` in
``src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/appearance/``
while keeping ``gs3dx_appearance.v1.json`` byte-identical for v1 consumers.

Usage:
    python3 scripts/export_simscape_grip_topology.py
    python3 scripts/export_simscape_grip_topology.py --check
    python3 scripts/export_simscape_grip_topology.py --output /tmp/out.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.grip_contact.simscape_grip_topology import (  # noqa: E402
    DEFAULT_GRIP_TOPOLOGY_JSON,
    GRIP_TOPOLOGY_SCHEMA_VERSION,
    build_grip_topology_payload,
    export_grip_topology,
    render_grip_topology_json,
    validate_grip_topology_payload,
)

SCHEMA_VERSION = GRIP_TOPOLOGY_SCHEMA_VERSION
DEFAULT_OUTPUT = DEFAULT_GRIP_TOPOLOGY_JSON


def build_payload() -> dict[str, Any]:
    """Build the deterministic ``simscape-grip-topology/v1`` payload."""
    return build_grip_topology_payload()


def validate_payload(payload: dict[str, Any]) -> None:
    """Validate payload schema and invariants."""
    validate_grip_topology_payload(payload)


def render_json(payload: dict[str, Any]) -> str:
    """Render deterministic JSON text with sorted keys and trailing newline."""
    return render_grip_topology_json(payload)


def export(output_path: Path, *, check: bool = False) -> int:
    """Write or check the grip topology JSON; return 0 on success, 1 on drift."""
    rc = export_grip_topology(output_path, check=check)
    if check and rc != 0:
        if not output_path.exists():
            print(
                f"ERROR: grip topology JSON missing at {output_path}",
                file=sys.stderr,
            )
        else:
            print(
                f"ERROR: grip topology JSON drifted at {output_path}. "
                "Re-run `python3 scripts/export_simscape_grip_topology.py`.",
                file=sys.stderr,
            )
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output grip topology JSON file.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the output file does not match generated JSON.",
    )
    args = parser.parse_args(argv)
    return export(args.output, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
