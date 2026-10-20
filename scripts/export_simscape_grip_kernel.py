#!/usr/bin/env python3
"""Generate the canonical Simscape grip contact MATLAB kernel and golden vectors (#13334).

Writes:
  - ``exploratory_gs3dx/tools/generated/gs3dx_grip_contact_kernel.m``
  - ``exploratory_gs3dx/tools/generated/gs3dx_grip_contact_params.json``
  - ``exploratory_gs3dx/tests/fixtures/gs3dx_grip_pair_golden.json``

Usage:
    python3 scripts/export_simscape_grip_kernel.py
    python3 scripts/export_simscape_grip_kernel.py --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.grip_contact.matlab_kernel import (  # noqa: E402
    DEFAULT_GOLDEN_JSON_PATH,
    DEFAULT_KERNEL_M_PATH,
    DEFAULT_PARAMS_JSON_PATH,
    export_matlab_kernel_bundle,
)


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI kernel export and freshness check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel-m",
        type=Path,
        default=DEFAULT_KERNEL_M_PATH,
        help="Path to the generated MATLAB kernel (.m).",
    )
    parser.add_argument(
        "--params-json",
        type=Path,
        default=DEFAULT_PARAMS_JSON_PATH,
        help="Path to the generated law and topology parameters JSON.",
    )
    parser.add_argument(
        "--golden-json",
        type=Path,
        default=DEFAULT_GOLDEN_JSON_PATH,
        help="Path to the generated golden vectors JSON fixture.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that the committed generated files match the Python source of truth.",
    )
    args = parser.parse_args(argv)
    rc = export_matlab_kernel_bundle(
        kernel_m_path=args.kernel_m,
        params_json_path=args.params_json,
        golden_json_path=args.golden_json,
        check=args.check,
    )
    if args.check and rc != 0:
        print(
            "ERROR: Simscape grip kernel bundle drifted or missing. "
            "Re-run `python3 scripts/export_simscape_grip_kernel.py`.",
            file=sys.stderr,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
