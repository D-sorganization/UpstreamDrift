#!/usr/bin/env python3
"""Generate fleet agent-hook adapter files from fleet-hooks/manifest.yml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from shared_scripts.agent_hook_adapters import (
        SUPPORTED_TARGETS,
        check_adapters,
        generate_adapters,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        default="cursor,claude,codex,antigravity,gemini,pre-push",
        help="comma-separated adapter targets",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when generated files drift from committed copies",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: cwd)",
    )
    args = parser.parse_args(argv)
    targets = frozenset(t.strip() for t in args.targets.split(",") if t.strip())
    unknown = targets - SUPPORTED_TARGETS
    if unknown:
        print(f"unknown targets: {sorted(unknown)}", file=sys.stderr)
        return 2
    root = args.repo_root.resolve()
    if args.check:
        drift = check_adapters(root, targets=targets)
        if drift:
            for path in drift:
                print(f"drift: {path}", file=sys.stderr)
            return 1
        return 0
    generate_adapters(root, targets=targets, write=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
