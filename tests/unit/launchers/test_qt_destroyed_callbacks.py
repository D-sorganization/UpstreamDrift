"""Qt destroyed callbacks must not close over ``self``.

The three GUI widgets that connect ``destroyed`` used a lambda that loads
``self`` from the method cell. Tearing the widget down clears that cell and
raises ``NameError: cannot access free variable 'self'``, which pytest-qt
reports as a failure of whichever test is running.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[3]
DESTROYED_CALLBACKS = (
    "src/launchers/cross_engine_dashboard.py",
    "src/tools/model_explorer/gui.py",
    "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/gui/core/main_widget.py",
)


def _destroyed_lambdas(path: Path) -> list[ast.Lambda]:
    """Return lambdas passed to ``destroyed.connect``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[ast.Lambda] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "connect":
            continue
        owner = func.value
        if not isinstance(owner, ast.Attribute) or owner.attr != "destroyed":
            continue
        callback = node.args[0]
        if isinstance(callback, ast.Lambda):
            found.append(callback)
    return found


def _loads_self(node: ast.AST) -> bool:
    """True when ``node`` loads the name ``self``."""
    return any(
        isinstance(child, ast.Name)
        and child.id == "self"
        and isinstance(child.ctx, ast.Load)
        for child in ast.walk(node)
    )


def test_destroyed_callbacks_do_not_close_over_self() -> None:
    """Each destroyed lambda binds the widget by value instead of loading self.

    ``destroyed`` may emit the QObject being torn down. That payload must not
    replace the bound widget, or cleanup runs on a bare ``QWidget``.
    """
    for relative in DESTROYED_CALLBACKS:
        path = REPO / relative
        lambdas = _destroyed_lambdas(path)
        assert lambdas, relative
        for callback in lambdas:
            assert not _loads_self(callback.body), relative
            assert callback.args.vararg is not None, relative
