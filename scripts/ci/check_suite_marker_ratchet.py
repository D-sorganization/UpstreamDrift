#!/usr/bin/env python3
"""Fail CI when tests without suite markers grow beyond the baseline.

The runtime pytest hook in ``tests/conftest.py`` reports missing suite markers
and can enforce either strict zero-unmarked mode or baseline-ratchet mode during
collection.  This script provides a faster CI gate: it statically scans test
source files for source-level tests that lack any marker from
``tests.support.suite_markers.SUITE_MARKERS``.

Usage::

    python scripts/ci/check_suite_marker_ratchet.py
    python scripts/ci/check_suite_marker_ratchet.py --update-baseline

Exit codes:
    0 - current unmarked tests are covered by the baseline
    1 - at least one net-new unmarked test was found
    2 - invalid arguments or invalid baseline shape
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from collections.abc import Iterable
from typing import TypeGuard

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tests.support.suite_markers import BASELINE_PATH, SUITE_MARKERS, normalize_nodeid

TESTS_ROOT = REPO_ROOT / "tests"
TEST_FILE_GLOBS = ("test_*.py", "*_test.py")


def _marker_name(expr: ast.AST) -> str | None:
    """Return the pytest marker name represented by *expr*, if any."""
    if isinstance(expr, ast.Call):
        return _marker_name(expr.func)
    if isinstance(expr, ast.Attribute):
        current: ast.AST = expr
        parts: list[str] = []
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        dotted = ".".join(reversed(parts))
        prefix = "pytest.mark."
        if dotted.startswith(prefix):
            return dotted.removeprefix(prefix).split(".", 1)[0]
    return None


def _suite_markers_from_expr(expr: ast.AST) -> set[str]:
    if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
        found: set[str] = set()
        for element in expr.elts:
            found.update(_suite_markers_from_expr(element))
        return found
    marker_name = _marker_name(expr)
    if marker_name in SUITE_MARKERS:
        return {marker_name}
    return set()


def _module_suite_markers(tree: ast.Module) -> set[str]:
    markers: set[str] = set()
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "pytestmark"
            for target in stmt.targets
        ):
            markers.update(_suite_markers_from_expr(stmt.value))
    return markers


def _decorator_suite_markers(
    node: ast.AsyncFunctionDef | ast.FunctionDef | ast.ClassDef,
) -> set[str]:
    markers: set[str] = set()
    for decorator in node.decorator_list:
        markers.update(_suite_markers_from_expr(decorator))
    return markers


def _is_test_function(
    node: ast.AST,
) -> TypeGuard[ast.AsyncFunctionDef | ast.FunctionDef]:
    return isinstance(
        node, (ast.AsyncFunctionDef, ast.FunctionDef)
    ) and node.name.startswith("test")


def _is_test_class(node: ast.AST) -> TypeGuard[ast.ClassDef]:
    return isinstance(node, ast.ClassDef) and node.name.startswith("Test")


def _nodeid(path: pathlib.Path, *parts: str) -> str:
    rel = path.relative_to(REPO_ROOT).as_posix()
    return "::".join((rel, *parts))


def _unmarked_in_file(path: pathlib.Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc

    module_markers = _module_suite_markers(tree)
    unmarked: list[str] = []

    for stmt in tree.body:
        if _is_test_function(stmt):
            if not module_markers and not _decorator_suite_markers(stmt):
                unmarked.append(_nodeid(path, stmt.name))
            continue

        if not _is_test_class(stmt):
            continue

        class_markers = module_markers | _decorator_suite_markers(stmt)
        for item in stmt.body:
            if _is_test_function(item):
                function_markers = class_markers | _decorator_suite_markers(item)
                if not function_markers:
                    unmarked.append(_nodeid(path, stmt.name, item.name))

    return unmarked


def iter_test_files(root: pathlib.Path = TESTS_ROOT) -> Iterable[pathlib.Path]:
    seen: set[pathlib.Path] = set()
    for pattern in TEST_FILE_GLOBS:
        for path in root.rglob(pattern):
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def collect_unmarked_nodeids(root: pathlib.Path | None = None) -> list[str]:
    scan_root = TESTS_ROOT if root is None else root
    nodeids: list[str] = []
    for path in sorted(iter_test_files(scan_root)):
        nodeids.extend(_unmarked_in_file(path))
    return sorted(normalize_nodeid(nodeid) for nodeid in nodeids)


def _load_baseline(path: pathlib.Path | None = None) -> list[str]:
    baseline_path = BASELINE_PATH if path is None else path
    if not baseline_path.exists():
        return []
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"baseline {baseline_path} must be a JSON object")
    nodeids = payload.get("unmarked_nodeids", [])
    if not isinstance(nodeids, list) or not all(isinstance(n, str) for n in nodeids):
        raise ValueError(
            f"baseline {baseline_path} must contain string list unmarked_nodeids"
        )
    return sorted(normalize_nodeid(nodeid) for nodeid in nodeids)


def _write_baseline(path: pathlib.Path, nodeids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Known tests without suite markers. This is a ratchet: CI allows "
            "entries to disappear but rejects new entries."
        ),
        "unmarked_nodeids": nodeids,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite the baseline with the current unmarked source-level tests.",
    )
    args = parser.parse_args(argv)

    current = collect_unmarked_nodeids()

    if args.update_baseline:
        _write_baseline(BASELINE_PATH, current)
        print(
            f"Updated {BASELINE_PATH.relative_to(REPO_ROOT)} with {len(current)} entries."
        )
        return 0

    baseline = set(_load_baseline())
    drift = [nodeid for nodeid in current if nodeid not in baseline]
    if drift:
        sys.stderr.write("Suite-marker ratchet FAILED: net-new unmarked tests found.\n")
        for nodeid in drift:
            sys.stderr.write(f"  - {nodeid}\n")
        sys.stderr.write(
            "Add an appropriate suite marker such as pytest.mark.unit, "
            "pytest.mark.integration, or pytest.mark.e2e. If paying down "
            "existing debt, run with --update-baseline after removing markers "
            "from the baseline only.\n"
        )
        return 1

    stale = sorted(baseline - set(current))
    if stale:
        print(
            f"Suite-marker baseline can shrink by {len(stale)} entr"
            f"{'y' if len(stale) == 1 else 'ies'}; run --update-baseline."
        )
    print(f"Suite-marker ratchet passed: {len(current)} unmarked tests, no drift.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
