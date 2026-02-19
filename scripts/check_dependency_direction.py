#!/usr/bin/env python3
"""Check dependency direction via configurable architecture fitness rules."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_RULES_PATH = Path("scripts/config/dependency_direction_rules.json")


def load_rules(project_root: Path, rules_path: Path) -> dict:
    """Load dependency-direction rule configuration."""
    path = project_root / rules_path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _is_exception_active(exc: dict, *, now: datetime) -> bool:
    """Return True if an exception is currently valid."""
    expires_on = exc.get("expires_on")
    if not expires_on:
        return True
    try:
        expiry = datetime.fromisoformat(expires_on).replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return now <= expiry


def build_exception_index(config: dict) -> tuple[set[str], list[str]]:
    """Build active-exception index and list any expired/invalid exceptions."""
    active: set[str] = set()
    invalid: list[str] = []
    now = datetime.now(tz=timezone.utc)
    for exc in config.get("exceptions", []):
        rel_path = exc.get("path", "")
        owner = exc.get("owner", "").strip()
        reason = exc.get("reason", "").strip()
        if not rel_path or not owner or not reason:
            invalid.append(f"Invalid exception entry missing required fields: {exc}")
            continue
        if _is_exception_active(exc, now=now):
            active.add(rel_path.replace("\\", "/"))
        else:
            invalid.append(
                f"Expired exception for {rel_path} (owner={owner}, expires_on={exc.get('expires_on')})"
            )
    return active, invalid


def get_top_level_imports(filepath: Path) -> list[tuple[int, str]]:
    """Extract module-level imports from a Python file.

    Returns list of (line_number, module_path) tuples.
    Skips imports inside TYPE_CHECKING blocks.
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    results = []
    for node in ast.iter_child_nodes(tree):
        # Skip TYPE_CHECKING blocks
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                continue
            if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
                continue

        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module:
            results.append((node.lineno, node.module))

    return results


def check_rules(src_root: Path, rules_path: Path, verbose: bool = False) -> list[str]:
    """Check all rules and return list of violation messages."""
    project_root = src_root.parent
    config = load_rules(project_root, rules_path)
    active_exceptions, invalid_exceptions = build_exception_index(config)
    violations = list(invalid_exceptions)

    rules = config.get("rules", [])
    for rule in rules:
        source_dir = str(rule["source_dir"])
        forbidden_prefixes = tuple(rule["forbidden_prefixes"])
        description = str(rule["description"])
        source_path = src_root / source_dir

        if not source_path.exists():
            if verbose:
                logger.debug("  SKIP: %s does not exist", source_dir)
            continue

        for py_file in source_path.rglob("*.py"):
            rel_path = str(py_file.relative_to(src_root)).replace("\\", "/")

            # Skip active exceptions
            if rel_path in active_exceptions:
                if verbose:
                    logger.debug("  SKIP (exception): %s", rel_path)
                continue

            imports = get_top_level_imports(py_file)
            for lineno, module_path in imports:
                if module_path.startswith(forbidden_prefixes):
                    msg = f"{rel_path}:{lineno}: {module_path} [{description}]"
                    violations.append(msg)

    return violations


def main() -> int:
    """Check import dependency directions and report violations."""
    parser = argparse.ArgumentParser(description="Check dependency directions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--rules-path",
        type=Path,
        default=DEFAULT_RULES_PATH,
        help="Path to JSON rule config relative to project root",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    src_root = project_root / "src" if (project_root / "src").exists() else project_root

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if args.verbose:
        logger.debug("Checking dependency directions in: %s", src_root)
        logger.debug("")

    violations = check_rules(src_root, rules_path=args.rules_path, verbose=args.verbose)

    if violations:
        logger.error(
            "FAIL: %d dependency direction violation(s) found:\n", len(violations)
        )
        for v in violations:
            logger.error("  %s", v)
        logger.error("")
        logger.error("Fix: Move the import inside a function body, use TYPE_CHECKING,")
        logger.error("     or relocate the module to the correct layer.")
        return 1
    logger.info("OK: No dependency direction violations found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
