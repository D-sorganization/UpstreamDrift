"""AST guard: grip-loss modules must not embed numeric force/friction literals (#15128)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_MODULES = tuple(
    Path(__file__)
    .resolve()
    .parents[3]
    .glob("src/shared/python/grip_contact/grip_loss_*.py")
)


def _force_literal_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target_text = " ".join(
            t.id if isinstance(t, ast.Name) else "" for t in node.targets
        ).lower()
        if not any(token in target_text for token in ("friction", "force", "squeeze")):
            continue
        if isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, (int, float)
        ):
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("path", sorted(_MODULES), ids=lambda p: p.name)
def test_grip_loss_modules_avoid_force_literals(path: Path) -> None:
    assert not _force_literal_lines(path), f"numeric force literals in {path.name}"
