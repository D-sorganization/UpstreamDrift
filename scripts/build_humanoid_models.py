#!/usr/bin/env python3
"""Cross-engine humanoid model build orchestrator.

This is the user-visible entry point for regenerating per-engine model files
from the shared anthropometric YAML at
``shared/models/golf_humanoid_dimensions.yaml``.

This script is owned long-term by issue **#4094 (PARITY-MODEL-BUILD)**; the
Drake-side hook landed here as part of issue **#4108 (DRAKE-1)**. Other
engines are wired by their own DRAKE-1-equivalents (see
``CROSS_ENGINE_PARITY_SPEC.md``).

Usage::

    python3 scripts/build_humanoid_models.py --engine drake
    python3 scripts/build_humanoid_models.py --engine drake --check  # CI gate

The ``--check`` mode regenerates the URDF into a temp dir and asserts the
on-disk file matches byte-for-byte; CI gate **#4129** uses this to forbid
hand-edits.
"""

from __future__ import annotations

import argparse
import filecmp
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_YAML = REPO_ROOT / "shared" / "models" / "golf_humanoid_dimensions.yaml"

# Make the repo root importable so this script works whether or not the
# package is pip-installed. CI invokes us as `python3 scripts/build_...`
# from the repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_drake(yaml_path: Path, *, check: bool) -> int:
    """Generate (or verify) the Drake humanoid URDF."""
    # Lazy import so that --engine mujoco doesn't pull in drake plumbing.
    from src.engines.physics_engines.drake.python.motion_matching.humanoid_urdf import (
        CANONICAL_URDF,
        build_humanoid_urdf,
    )

    if check:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_out = Path(tmp) / "golfer.urdf"
            build_humanoid_urdf(yaml_path=yaml_path, out_path=tmp_out)
            if not CANONICAL_URDF.exists():
                sys.stderr.write(f"FAIL: canonical URDF missing at {CANONICAL_URDF}\n")
                return 1
            if not filecmp.cmp(tmp_out, CANONICAL_URDF, shallow=False):
                sys.stderr.write(
                    "FAIL: regenerated URDF differs from on-disk file. "
                    "Run `python3 scripts/build_humanoid_models.py "
                    "--engine drake` and commit the result.\n"
                )
                return 1
            sys.stdout.write("OK: drake URDF matches regeneration.\n")
            return 0

    out = build_humanoid_urdf(yaml_path=yaml_path)
    sys.stdout.write(f"Wrote {out}\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        action="append",
        choices=["drake"],
        required=True,
        help="Engine(s) to regenerate. Repeatable.",
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=SHARED_YAML,
        help="Path to the shared humanoid dimensions YAML.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Regenerate to a temp dir and diff against the on-disk file. "
            "Exits non-zero if they differ. Used by CI gate #4129."
        ),
    )
    args = parser.parse_args(argv)

    if not args.yaml.exists():
        sys.stderr.write(f"YAML not found: {args.yaml}\n")
        return 2

    rc = 0
    for engine in args.engine:
        if engine == "drake":
            rc |= _build_drake(args.yaml, check=args.check)
        else:  # pragma: no cover - argparse rejects other values
            sys.stderr.write(f"Unsupported engine: {engine}\n")
            rc |= 2
    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
