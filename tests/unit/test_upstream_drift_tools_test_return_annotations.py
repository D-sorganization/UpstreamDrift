"""Regression test for sidekick test return annotations."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPED_FILES = (
    "src/shared/python/sidekick/tests/lab/bio/test_c3d_reader_fixed.py",
    "src/shared/python/sidekick/tests/calculators/electrical/test_electrical_model.py",
    "src/shared/python/sidekick/tests/calculators/conversion/test_conversion.py",
    "src/shared/python/sidekick/process_calculators/psa_package/test_psa_model.py",
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


def test_sidekick_tests_have_return_annotations() -> None:
    assert not _missing_return_annotations()
