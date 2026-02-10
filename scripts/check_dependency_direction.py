#!/usr/bin/env python3
"""Check that dependency directions between top-level modules are correct.

This script performs static analysis using Python's AST module to detect
forbidden import directions. It does NOT execute any code, so it works
even if optional dependencies (MuJoCo, Drake, etc.) are not installed.

Forbidden directions (at module level, outside TYPE_CHECKING blocks):
  - shared -> engines  (engines should depend on shared, not the reverse)
  - shared -> robotics (robotics should depend on shared, not the reverse)
  - api -> launchers   (should use api.services.launcher_service instead)

Allowed exceptions:
  - Backward-compatible shim files that re-export from new locations
  - Imports inside function bodies (lazy imports)
  - Imports inside TYPE_CHECKING blocks

Usage:
    python scripts/check_dependency_direction.py
    python scripts/check_dependency_direction.py --verbose

Exit codes:
    0: No violations found
    1: Violations found
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Shim files that are explicitly allowed to import across boundaries
# because they exist solely for backward compatibility
ALLOWED_SHIMS = {
    "shared/python/engine_loaders.py",  # re-exports from src.engines.loaders
    "engines/common/capabilities.py",  # re-exports from src.shared.python.capabilities
}

# Rules: (source_dir_relative_to_src, forbidden_target_prefix, description)
RULES = [
    (
        "shared/python",
        ("src.engines", "engines"),
        "shared -> engines (inverted dependency)",
    ),
    (
        "shared/python",
        ("src.robotics", "robotics"),
        "shared -> robotics (inverted dependency)",
    ),
    (
        "api",
        ("src.launchers", "launchers"),
        "api -> launchers (should use api.services.launcher_service)",
    ),
]


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
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                results.append((node.lineno, node.module))

    return results


def check_rules(src_root: Path, verbose: bool = False) -> list[str]:
    """Check all rules and return list of violation messages."""
    violations = []

    for source_dir, forbidden_prefixes, description in RULES:
        source_path = src_root / source_dir

        if not source_path.exists():
            if verbose:
                print(f"  SKIP: {source_dir} does not exist")
            continue

        for py_file in source_path.rglob("*.py"):
            rel_path = str(py_file.relative_to(src_root)).replace("\\", "/")

            # Skip allowed shims
            if rel_path in ALLOWED_SHIMS:
                if verbose:
                    print(f"  SKIP (shim): {rel_path}")
                continue

            imports = get_top_level_imports(py_file)
            for lineno, module_path in imports:
                if module_path.startswith(forbidden_prefixes):
                    msg = f"{rel_path}:{lineno}: {module_path} [{description}]"
                    violations.append(msg)

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description="Check dependency directions")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Find project root (parent of scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    src_root = project_root / "src" if (project_root / "src").exists() else project_root

    if args.verbose:
        print(f"Checking dependency directions in: {src_root}")
        print()

    violations = check_rules(src_root, verbose=args.verbose)

    if violations:
        print(f"FAIL: {len(violations)} dependency direction violation(s) found:\n")
        for v in violations:
            print(f"  {v}")
        print()
        print("Fix: Move the import inside a function body, use TYPE_CHECKING,")
        print("     or relocate the module to the correct layer.")
        return 1
    else:
        print("OK: No dependency direction violations found.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
