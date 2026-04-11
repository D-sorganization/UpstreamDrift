"""Regression test for plot_engine test return annotations."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPED_FILES = (
    "src/shared/python/plot_engine/tests/test_plotly_converter.py",
    "src/shared/python/plot_engine/tests/test_matplotlib_renderer.py",
)


def _missing_return_annotations() -> list[str]:
    violations: list[str] = []

    for relative_path in SCOPED_FILES:
        path = REPO_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.returns is None
            ):
                violations.append(f"{relative_path}:{node.lineno}")

    return violations


def test_plot_engine_tests_have_return_annotations() -> None:
    assert not _missing_return_annotations()
