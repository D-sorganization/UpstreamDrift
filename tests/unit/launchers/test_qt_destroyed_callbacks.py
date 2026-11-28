"""Qt destroyed callbacks must not close over a method's ``self`` cell.

Tearing down a widget can clear the method cell and raise
``NameError: cannot access free variable 'self'`` if a callback closes over it.
Use value-bound lambdas or a named helper wrapped in ``functools.partial``.
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


def _destroyed_callbacks(path: Path) -> list[ast.expr]:
    """Return callbacks passed to ``destroyed.connect``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[ast.expr] = []
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
    """Each destroyed callback avoids a closure over its method's ``self``.

    Lambdas bind the widget in a default argument, while the dashboard uses a
    factory that returns ``functools.partial`` around a module-level helper.
    ``destroyed`` may emit the QObject being torn down, so each callback must
    accept and ignore that signal payload.
    """
    for relative in DESTROYED_CALLBACKS:
        path = REPO / relative
        callbacks = _destroyed_callbacks(path)
        assert callbacks, relative
        for callback in callbacks:
            if isinstance(callback, ast.Lambda):
                assert not _loads_self(callback.body), relative
                assert callback.args.vararg is not None, relative
                continue

            assert isinstance(callback, ast.Call), relative
            assert isinstance(callback.func, ast.Name), relative
            assert callback.func.id == "dashboard_destroyed_slot", relative
            assert len(callback.args) == 1, relative
            assert isinstance(callback.args[0], ast.Name), relative
            assert callback.args[0].id == "self", relative

            theme_module = REPO / "src/launchers/cross_engine_dashboard_theme.py"
            theme_tree = ast.parse(theme_module.read_text(encoding="utf-8"))
            factory = next(
                node
                for node in theme_tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "dashboard_destroyed_slot"
            )
            returns = [
                node for node in ast.walk(factory) if isinstance(node, ast.Return)
            ]
            assert len(returns) == 1, relative
            returned = returns[0].value
            assert isinstance(returned, ast.Call), relative
            assert isinstance(returned.func, ast.Name), relative
            assert returned.func.id == "partial", relative
            assert len(returned.args) == 2, relative
            assert isinstance(returned.args[0], ast.Name), relative
            assert returned.args[0].id == "_release_dashboard_theme_manager", relative
            assert isinstance(returned.args[1], ast.Name), relative
            assert returned.args[1].id == "window", relative
            assert not _loads_self(returned), relative

            helper = next(
                node
                for node in theme_tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_release_dashboard_theme_manager"
            )
            assert not _loads_self(helper), relative
            assert helper.args.vararg is not None, relative
            assert helper.args.vararg.arg == "_args", relative
            release_calls = [
                node
                for node in ast.walk(helper)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "release_theme_manager"
            ]
            assert len(release_calls) == 1, relative
            assert len(release_calls[0].args) == 1, relative
            assert isinstance(release_calls[0].args[0], ast.Name), relative
            assert release_calls[0].args[0].id == "window", relative
