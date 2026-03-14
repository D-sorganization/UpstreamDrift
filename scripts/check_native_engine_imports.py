#!/usr/bin/env python3
"""Check that expected native engine modules can be imported.

This script is intended for CI and local troubleshooting of optional
native-engine environments used by integration and cross-engine tests.
It writes both JSON and Markdown summaries when output paths are provided.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

DEFAULT_CHECKS: dict[str, str] = {
    "mujoco": "mujoco",
    "drake": "pydrake.all",
    "pinocchio": "pinocchio",
}


def parse_checks(values: list[str] | None) -> dict[str, str]:
    """Parse `name=module.path` pairs into a check mapping."""
    if not values:
        return DEFAULT_CHECKS.copy()

    checks: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid check {value!r}; expected format name=module.path"
            )
        name, module_path = value.split("=", 1)
        name = name.strip()
        module_path = module_path.strip()
        if not name or not module_path:
            raise ValueError(
                f"Invalid check {value!r}; both name and module path are required"
            )
        checks[name] = module_path
    return checks


def run_checks(checks: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Import each module and collect status strings plus failures."""
    results: dict[str, str] = {}
    failures: list[str] = []

    for name, module_name in checks.items():
        try:
            importlib.import_module(module_name)
            results[name] = "ok"
        except Exception as exc:  # pragma: no cover - environment dependent
            results[name] = f"failed: {exc}"
            failures.append(f"{name}: {exc}")

    return results, failures


def write_outputs(
    results: dict[str, str],
    json_output: Path | None,
    markdown_output: Path | None,
) -> None:
    """Write machine-readable and human-readable output files."""
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if markdown_output is not None:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        lines = ["## Native Engine Import Check", ""]
        for name, status in results.items():
            icon = "OK" if status == "ok" else "FAIL"
            lines.append(f"- {icon} {name}: {status}")
        markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="append",
        help="Engine import check in the form name=module.path. May be repeated.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write JSON results.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        help="Optional path to write Markdown summary.",
    )
    return parser


def main() -> int:
    """Run the configured checks and return a shell-friendly exit code."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        checks = parse_checks(args.check)
    except ValueError as exc:
        parser.error(str(exc))

    results, failures = run_checks(checks)
    write_outputs(results, args.json_output, args.markdown_output)

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
