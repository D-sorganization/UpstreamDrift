#!/usr/bin/env python3
"""Build the Sidekick standalone binary via PyInstaller.

Usage
-----
    python scripts/packaging/build_sidekick_binary.py [--max-mb N] [--output-dir DIR]

The script:
1. Runs PyInstaller with ``sidekick.spec``.
2. Verifies the resulting binary is under ``--max-mb`` (default 250 MB).
3. Prints a summary line with the binary path and size.

Exit codes
----------
0  Build succeeded and binary is within the size budget.
1  Build failed or binary exceeds the size budget.

Design-by-Contract
------------------
Precondition:  PyInstaller is importable (``pip install pyinstaller``).
Postcondition: If exit code is 0, ``{output_dir}/sidekick[.exe]`` exists and
               its size ≤ max_mb * 1024 * 1024 bytes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

MAX_MB_DEFAULT = 250
SPEC_FILE = Path(__file__).parent.parent.parent / "sidekick.spec"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--max-mb",
        type=int,
        default=MAX_MB_DEFAULT,
        metavar="N",
        help=f"Maximum allowed binary size in MB (default: {MAX_MB_DEFAULT})",
    )
    p.add_argument(
        "--output-dir",
        default="dist",
        metavar="DIR",
        help="Directory where PyInstaller writes the binary (default: dist)",
    )
    return p.parse_args()


def _binary_name() -> str:
    return "sidekick.exe" if sys.platform == "win32" else "sidekick"


def main() -> int:
    args = _parse_args()

    assert SPEC_FILE.exists(), f"sidekick.spec not found at {SPEC_FILE}"
    assert args.max_mb > 0, f"--max-mb must be positive, got {args.max_mb}"

    env = {**os.environ, "SKIP_UI_BUILD": "1"}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--distpath",
            args.output_dir,
            str(SPEC_FILE),
        ],
        env=env,
        check=False,
    )

    if result.returncode != 0:
        print(
            f"::error::PyInstaller failed with exit code {result.returncode}",
            file=sys.stderr,
        )
        return 1

    binary = Path(args.output_dir) / _binary_name()
    if not binary.exists():
        print(f"::error::Binary not found at {binary}", file=sys.stderr)
        return 1

    size_bytes = binary.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    max_bytes = args.max_mb * 1024 * 1024

    print(f"Binary: {binary}")
    print(f"Size:   {size_mb:.1f} MB (budget: {args.max_mb} MB)")

    if size_bytes > max_bytes:
        print(
            f"::error::Binary size {size_mb:.1f} MB exceeds budget of {args.max_mb} MB. "
            "If this growth is intentional, update MAX_MB in sidekick.spec and add "
            "justification to the PR description.",
            file=sys.stderr,
        )
        return 1

    # Emit workflow summary line (GitHub Actions picks this up)
    summary_env = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_env:
        with open(summary_env, "a", encoding="utf-8") as fh:
            fh.write(f"| {sys.platform} | {binary.name} | {size_mb:.1f} MB |\n")

    assert binary.stat().st_size <= max_bytes, "postcondition: binary within budget"
    return 0


if __name__ == "__main__":
    sys.exit(main())
