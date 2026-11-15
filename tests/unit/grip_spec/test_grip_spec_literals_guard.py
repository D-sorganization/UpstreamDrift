"""AST guard: grip_spec must not embed numeric force/contact literals (#15123)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PKG = Path(__file__).resolve().parents[3] / "src/shared/python/grip_spec"
_FORCE_MODULE = _PKG / "providers.py"
_ALLOWED_GEOMETRY_FILES = frozenset(
    {"defaults.py", "serialization.py", "types.py", "validation.py", "projection.py"}
)


def _assignment_targets(node: ast.Assign | ast.AnnAssign) -> list[ast.expr]:
    if isinstance(node, ast.AnnAssign):
        return [node.target]
    return list(node.targets)


def _force_literal_nodes(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if isinstance(node, ast.AnnAssign) and node.value is None:
            continue
        targets = " ".join(
            getattr(t, "id", "")
            or getattr(getattr(t, "attr", None), "__str__", lambda: "")()
            for t in _assignment_targets(node)
        ).lower()
        if not any(
            token in targets for token in ("force", "stiff", "friction", "preload")
        ):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            hits.append(node.lineno)
    return hits


def test_no_numeric_force_literals_in_grip_providers_module() -> None:
    violations = _force_literal_nodes(_FORCE_MODULE)
    assert not violations, violations
    text = _FORCE_MODULE.read_text(encoding="utf-8")
    assert not re.search(
        r"\b(stiffness|friction|preload|force_limit)\w*\s*=\s*[0-9]",
        text,
        flags=re.IGNORECASE,
    )


@pytest.mark.parametrize("module_path", sorted(_PKG.glob("*.py")), ids=lambda p: p.name)
def test_geometry_modules_do_not_define_force_constants(module_path: Path) -> None:
    if module_path.name not in _ALLOWED_GEOMETRY_FILES:
        pytest.skip("providers-only gate for force literals")
    if module_path.name != "providers.py":
        assert not _force_literal_nodes(module_path)
