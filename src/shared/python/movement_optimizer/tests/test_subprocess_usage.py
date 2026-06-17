"""Regression tests for safe subprocess usage."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCES = (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src", PROJECT_ROOT / "tests")


def _subprocess_calls(tree: ast.AST) -> list[ast.Call]:
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
        ):
            calls.append(node)
    return calls


def test_subprocess_calls_do_not_use_shell_true() -> None:
    offenders: list[str] = []
    for root in PYTHON_SOURCES:
        for source in root.rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for call in _subprocess_calls(tree):
                for keyword in call.keywords:
                    if (
                        keyword.arg == "shell"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    ):
                        offenders.append(f"{source.relative_to(PROJECT_ROOT)}:{call.lineno}")

    assert offenders == []


def test_subprocess_calls_use_sequence_arguments() -> None:
    offenders: list[str] = []
    for root in PYTHON_SOURCES:
        for source in root.rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for call in _subprocess_calls(tree):
                if not call.args:
                    continue
                first_arg = call.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    offenders.append(f"{source.relative_to(PROJECT_ROOT)}:{call.lineno}")

    assert offenders == []
