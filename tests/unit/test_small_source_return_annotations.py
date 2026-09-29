"""Regression test for a bounded source-level return-annotation slice."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPED_FILES = (
    "src/shared/python/model_generation/explorer/__init__.py",
    "src/shared/python/data_io/output_manager.py",
    "src/shared/python/output_manager.py",
    "src/shared/python/pendulum_simulator/optimizer_gpu.py",
    "src/tools/video_analyzer/types.py",
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


def test_small_source_slice_has_return_annotations() -> None:
    assert not _missing_return_annotations()
