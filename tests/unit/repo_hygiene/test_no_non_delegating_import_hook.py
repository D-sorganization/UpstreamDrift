"""Guard against non-delegating ``builtins.__import__`` replacements (UD #9474).

Why this rule exists
--------------------
A test that installs its own ``builtins.__import__`` does not only intercept
the imports performed by the code under test.  It intercepts *every* import
performed by every frame that runs while the patch is installed -- including
pytest's own lazy imports.

``_pytest.nodes._repr_failure_py`` does ``from _pytest.fixtures import
FixtureLookupError`` while formatting a failed test.  If the installed hook
raises ``ImportError`` for names it does not recognise, that lazy import fails,
pytest cannot build the failure report, and the *entire session* dies with
``INTERNALERROR`` and exit code 3.  When that happens pytest never prints the
``short test summary info`` section, never prints the ``FAILURES`` section and
never writes the coverage report -- the run reports a failure count with no way
to learn which tests failed.

That is exactly what happened to ``tests (3.12)`` on ``main``: 305 failures
were counted and not one of them was named, because
``tests/unit/installer/test_build_installer.py::test_detect_physics_engines``
installed a hook that raised ``ImportError`` unconditionally.

The rule is therefore: a function installed as ``builtins.__import__`` must
delegate to the real import for names it does not handle.  Code under test
should expose a narrow, named seam (see
``installer.windows.build_installer._module_available``) so that tests never
need to replace the interpreter's import hook at all.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TESTS_ROOT = _REPO_ROOT / "tests"

# This ledger only ever ratchets down.  Adding an entry requires an owner and a
# tracking issue; the guard above explains why an entry is a latent
# session-killing defect rather than a style preference.
_PREEXISTING_NON_DELEGATING_HOOKS: frozenset[tuple[str, str]] = frozenset()


def _iter_test_sources() -> Iterator[tuple[Path, ast.Module]]:
    for path in sorted(_TESTS_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        yield path, tree


def _names_bound_to_real_import(tree: ast.Module) -> set[str]:
    """Local names that hold a reference to the genuine ``__import__``."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if (
            (isinstance(value, ast.Name) and value.id == "__import__")
            or (isinstance(value, ast.Attribute) and value.attr == "__import__")
            or (
                # e.g. ``original = module.__builtins__["__import__"]``
                isinstance(value, ast.Subscript)
                and isinstance(value.slice, ast.Constant)
                and value.slice.value == "__import__"
            )
        ):
            bound.add(target.id)
    return bound


def _functions_installed_as_import_hook(tree: ast.Module) -> set[str]:
    """Names of locally defined functions patched over ``builtins.__import__``."""
    installed: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        callee = (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        )
        if callee not in {"setattr", "patch", "setitem"}:
            continue
        targets_import_hook = any(
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and arg.value in {"__import__", "builtins.__import__"}
            for arg in node.args
        )
        if not targets_import_hook:
            continue
        installed.update(arg.id for arg in node.args if isinstance(arg, ast.Name))
    return installed


def _delegates_to_real_import(
    func: ast.FunctionDef | ast.AsyncFunctionDef, real: set[str]
) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        if isinstance(callee, ast.Name) and (
            callee.id in real or callee.id == "__import__"
        ):
            return True
        if isinstance(callee, ast.Attribute) and callee.attr == "__import__":
            return True
    return False


def _find_non_delegating_hooks() -> list[tuple[str, str, int]]:
    offenders: list[tuple[str, str, int]] = []
    for path, tree in _iter_test_sources():
        installed = _functions_installed_as_import_hook(tree)
        if not installed:
            continue
        real = _names_bound_to_real_import(tree)
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        for name in sorted(installed):
            func = functions.get(name)
            if func is None:
                continue
            if not _delegates_to_real_import(func, real):
                relative = path.relative_to(_REPO_ROOT).as_posix()
                offenders.append((relative, name, func.lineno))
    return offenders


def test_import_hooks_delegate_to_the_real_import() -> None:
    """No test may install an import hook that fails unrecognised names.

    Such a hook also intercepts pytest's own lazy imports and aborts the whole
    session with ``INTERNALERROR``, destroying the failure summary for every
    other test in the run (UD #9474).
    """
    offenders = [
        (path, name, lineno)
        for path, name, lineno in _find_non_delegating_hooks()
        if (path, name) not in _PREEXISTING_NON_DELEGATING_HOOKS
    ]
    assert not offenders, (
        "These functions are installed as builtins.__import__ but raise for "
        "names they do not recognise. They will abort the entire pytest "
        "session (INTERNALERROR, exit code 3) as soon as any test fails while "
        "they are installed. Delegate to the saved real __import__, or patch a "
        "narrow seam in the code under test instead:\n"
        + "\n".join(
            f"  {path}:{lineno} -> {name}()" for path, name, lineno in offenders
        )
    )


def test_ledger_only_lists_hooks_that_still_exist() -> None:
    """The allowlist must ratchet down: stale entries are a failure."""
    present = {(path, name) for path, name, _ in _find_non_delegating_hooks()}
    stale = sorted(_PREEXISTING_NON_DELEGATING_HOOKS - present)
    assert not stale, (
        "Remove these fixed entries from _PREEXISTING_NON_DELEGATING_HOOKS; "
        f"the ledger only ratchets down: {stale}"
    )
