"""Regression coverage for sidekick test import isolation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


@pytest.mark.unit
def test_sidekick_tests_do_not_clear_global_module_cache() -> None:
    root = Path("src/shared/python/sidekick/tests")
    offenders: list[str] = []

    for path in root.rglob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "clear"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "modules"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "sys"
            ):
                offenders.append(f"{path}:{node.lineno}")

    assert offenders == []
