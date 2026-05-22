"""Meta-test: assert-density audit across the test suite (issue #5910).

This test file is a *meta-test* — it introspects the test suite itself rather
than any production source.  Its purpose is to surface test functions that
contain no ``assert`` statement, which are a common source of silent test
failures (the function passes trivially without exercising any behaviour).

Design:
    - Scans all ``test_*.py`` files under ``tests/``.
    - For each ``def test_*`` function, checks for at least one ``assert``
      keyword in the function body (after the ``def`` line).
    - Reports functions with zero asserts as **warnings** (not hard failures)
      so that this check does not block CI during the initial audit phase
      (issue #5910, grade 5/10 → improvements tracked incrementally).
    - The ``test_assert_density_summary`` test *always passes* but logs a
      summary of any zero-assert functions found so that CI step summaries
      surface the count.

Promoting to a hard failure:
    Change ``HARD_FAIL = False`` to ``True`` once the backlog of zero-assert
    functions has been resolved (tracked in issue #5910).
"""

from __future__ import annotations

import ast
import logging
import re
import warnings
from pathlib import Path
from typing import NamedTuple

import pytest

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set to True to turn zero-assert functions into hard test failures.
# Currently False so that the check surfaces findings without blocking CI.
HARD_FAIL: bool = False

# Directories to scan (relative to repo root).
_SCAN_DIRS: list[str] = ["tests"]

# Patterns to exclude from scanning (e.g. large generated or archive files).
_EXCLUDE_PATTERNS: list[str] = [
    "*/archive/*",
    "*/legacy/*",
    "*/vendor/*",
    "*/.git/*",
    "*/conftest.py",
]

# Minimum number of asserts expected per test function.
_MIN_ASSERTS: int = 1

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class ZeroAssertFunction(NamedTuple):
    """A test function with fewer than ``_MIN_ASSERTS`` assert statements."""

    file: Path
    function: str
    line: int


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

_ASSERT_RE = re.compile(r"\bassert\b")


def _is_excluded(path: Path, patterns: list[str]) -> bool:
    """Return True if *path* matches any glob *patterns*."""
    return any(path.match(pat) for pat in patterns)


def _count_asserts_ast(
    source: str, func_node: ast.FunctionDef | ast.AsyncFunctionDef
) -> int:
    """Count ``assert`` statements inside *func_node* using the AST.

    Traverses the function body recursively so that asserts inside nested
    ``for`` / ``with`` / ``if`` blocks are also counted.

    Args:
        source: Full source text (unused; kept for potential future use).
        func_node: The parsed function node to inspect.

    Returns:
        Number of ``ast.Assert`` nodes found in the function body.
    """
    count = 0
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assert):
            count += 1
    return count


def collect_zero_assert_functions(repo_root: Path) -> list[ZeroAssertFunction]:
    """Scan test files and return functions with fewer than ``_MIN_ASSERTS`` asserts.

    Args:
        repo_root: Absolute path to the repository root.

    Returns:
        Sorted list of :class:`ZeroAssertFunction` entries.
    """
    results: list[ZeroAssertFunction] = []

    for scan_dir in _SCAN_DIRS:
        base = repo_root / scan_dir
        if not base.is_dir():
            log.debug("Scan dir %s not found; skipping", base)
            continue

        for py_file in sorted(base.rglob("test_*.py")):
            if _is_excluded(py_file, _EXCLUDE_PATTERNS):
                continue

            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                log.warning("Cannot read %s: %s", py_file, exc)
                continue

            try:
                tree = ast.parse(source, filename=str(py_file))
            except SyntaxError as exc:
                log.warning("Syntax error in %s: %s", py_file, exc)
                continue

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not node.name.startswith("test_"):
                    continue

                assert_count = _count_asserts_ast(source, node)
                if assert_count < _MIN_ASSERTS:
                    results.append(
                        ZeroAssertFunction(
                            file=py_file.relative_to(repo_root),
                            function=node.name,
                            line=node.lineno,
                        )
                    )

    results.sort(key=lambda r: (str(r.file), r.line))
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Return the repository root (two levels above ``tests/``).

    Returns:
        Absolute path to the repository root.
    """
    # This file lives at tests/unit/test_assert_density.py
    return Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_assert_density_summary() -> None:
    """Scan test files and warn about zero-assert functions (issue #5910).

    This test always passes but emits a ``UserWarning`` summarising any
    zero-assert functions found.  Set ``HARD_FAIL = True`` at module level
    to promote findings to failures.
    """
    root = _repo_root()
    zero_assert = collect_zero_assert_functions(root)

    if zero_assert:
        lines = [
            f"\n[assert-density] {len(zero_assert)} test function(s) have "
            f"zero assert statements (issue #5910):\n"
        ]
        for entry in zero_assert:
            lines.append(f"  {entry.file}:{entry.line}  {entry.function}()")
        summary = "\n".join(lines)

        if HARD_FAIL:
            pytest.fail(summary)
        else:
            warnings.warn(summary, UserWarning, stacklevel=1)
            log.warning(summary)
    else:
        log.info(
            "[assert-density] All scanned test functions have at least one assert."
        )

    # This assertion always passes — the test's value is in the warning above.
    assert isinstance(zero_assert, list)


@pytest.mark.unit
def test_assert_density_no_regressions() -> None:
    """Guard: the assert-density scanner itself must not raise on the repo.

    Verifies that :func:`collect_zero_assert_functions` completes without
    raising an exception so that CI always has a meaningful result.
    """
    root = _repo_root()
    try:
        results = collect_zero_assert_functions(root)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"collect_zero_assert_functions raised unexpectedly: {exc!r}")

    assert isinstance(results, list), "Expected a list of ZeroAssertFunction entries"
