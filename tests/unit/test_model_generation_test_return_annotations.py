"""Regression test for model_generation test return annotations."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPED_FILES = (
    "src/shared/python/model_generation/tests/test_api.py",
    "src/shared/python/model_generation/tests/test_cli.py",
    "src/shared/python/model_generation/tests/test_editor.py",
    "src/shared/python/model_generation/tests/test_github_importer.py",
    "src/shared/python/model_generation/tests/test_library.py",
    "src/shared/python/model_generation/tests/test_simscape.py",
    "src/shared/python/model_generation/tests/test_unified_loader.py",
)


def _missing_return_annotations() -> list[str]:
    violations: list[str] = []

    for relative_path in SCOPED_FILES:
        path = REPO_ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.returns is None
            ):
                violations.append(f"{relative_path}:{node.lineno}")

    return violations


def test_model_generation_tests_have_return_annotations() -> None:
    assert not _missing_return_annotations()
