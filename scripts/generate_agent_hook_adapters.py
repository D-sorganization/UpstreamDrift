#!/usr/bin/env python3
# ruff: noqa
# fmt: off
# flake8: noqa
# mypy: ignore-errors
"""Generate fleet agent-hook adapter files from fleet-hooks/manifest.yml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from shared_scripts.agent_hook_adapters import (  # noqa: PLC0415
        SUPPORTED_TARGETS,
        check_adapters,
        generate_adapters,
        load_runtime_descriptor,
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
    parser.add_argument(
        "--runtime-descriptor",
        type=Path,
        help="absolute protected host runtime.json installed outside the checkout",
    )
    args = parser.parse_args(argv)
    targets = frozenset(t.strip() for t in args.targets.split(",") if t.strip())
    unknown = targets - SUPPORTED_TARGETS
    if unknown:
        print(f"unknown targets: {sorted(unknown)}", file=sys.stderr)
        return 2
    root = args.repo_root.resolve()
    if args.check:
        runtime = (
            load_runtime_descriptor(args.runtime_descriptor, root)
            if args.runtime_descriptor
            else None
        )
        drift = check_adapters(root, targets=targets, runtime=runtime)
        if drift:
            for path in drift:
                print(f"drift: {path}", file=sys.stderr)
            return 1
        return 0
    if not args.runtime_descriptor:
        print("runtime is not installed; run scripts/install_trusted_agent_hooks.py", file=sys.stderr)
        return 2
    runtime = load_runtime_descriptor(args.runtime_descriptor, root)
    generate_adapters(root, targets=targets, runtime=runtime, write=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
