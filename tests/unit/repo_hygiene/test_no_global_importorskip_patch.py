from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.unit


def _assigns_pytest_importorskip(node: ast.AST) -> bool:
    targets: list[ast.AST]
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign | ast.AugAssign):
        targets = [node.target]
    else:
        return False
    return any(_is_pytest_importorskip(target) for target in targets)


def _calls_setattr_pytest_importorskip(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "pytest"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "importorskip"
    )


def _is_pytest_importorskip(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "importorskip"
        and isinstance(node.value, ast.Name)
        and node.value.id == "pytest"
    )


def test_conftests_do_not_patch_pytest_importorskip_globally() -> None:
    offenders: list[str] = []
    for conftest in sorted((REPO_ROOT / "tests").rglob("conftest.py")):
        tree = ast.parse(conftest.read_text(encoding="utf-8"), filename=str(conftest))
        for node in ast.walk(tree):
            if _assigns_pytest_importorskip(node) or _calls_setattr_pytest_importorskip(
                node
            ):
                rel_path = conftest.relative_to(REPO_ROOT).as_posix()
                line = getattr(node, "lineno", 0)
                offenders.append(f"{rel_path}:{line}")

    assert offenders == []
