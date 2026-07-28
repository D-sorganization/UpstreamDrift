#!/usr/bin/env python3
"""Reject collected test functions whose body is only a no-op placeholder."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = Path("pyproject.toml")


def iter_test_files(repo_root: Path) -> list[Path]:
    """Return configured pytest test files under pyproject testpaths."""
    testpaths = _load_testpaths(repo_root / PYPROJECT)
    test_files: list[Path] = []
    for testpath in testpaths:
        root = repo_root / testpath
        if root.is_file() and root.name.startswith("test_") and root.suffix == ".py":
            test_files.append(root)
        elif root.is_dir():
            test_files.extend(sorted(root.rglob("test_*.py")))
    return sorted(set(test_files))


def find_noop_tests(repo_root: Path) -> list[str]:
    """Return repo-relative findings for no-op test functions."""
    findings: list[str] = []
    for test_file in iter_test_files(repo_root):
        relative_path = test_file.relative_to(repo_root).as_posix()
        try:
            tree = ast.parse(
                test_file.read_text(encoding="utf-8"), filename=relative_path
            )
        except SyntaxError as exc:
            findings.append(f"{relative_path}:{exc.lineno}: syntax error: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if not node.name.startswith("test_"):
                continue
            noop_kind = _noop_test_kind(node)
            if noop_kind is not None:
                findings.append(
                    f"{relative_path}:{node.lineno}: {node.name} is a no-op "
                    f"placeholder ({noop_kind})"
                )
    return findings


def _load_testpaths(pyproject_path: Path) -> list[str]:
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found: {pyproject_path}")
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    testpaths = data["tool"]["pytest"]["ini_options"]["testpaths"]
    if not isinstance(testpaths, list) or not all(
        isinstance(path, str) for path in testpaths
    ):
        raise ValueError("pyproject pytest testpaths must be a list of strings")
    return testpaths


def _effective_body(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.stmt]:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _noop_test_kind(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    body = _effective_body(node)
    if len(body) != 1:
        return None
    statement = body[0]
    if isinstance(statement, ast.Pass):
        return "pass"
    if (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is Ellipsis
    ):
        return "ellipsis"
    if isinstance(statement, ast.Assert) and _is_constant_true(statement.test):
        return "assert-true"
    return None


def _is_constant_true(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is True
    return False


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Repository root to audit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    findings = find_noop_tests(args.repo_root.resolve())
    if findings:
        print("No-op test guard failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print("No no-op placeholder tests found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
