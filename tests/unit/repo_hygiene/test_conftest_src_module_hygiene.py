"""No ``conftest.py`` may pivot ``sys.modules["src"]`` for the whole process.

General hygiene guard for issues #9402 / #9404 / #9409.

``sys.modules`` is process-global. Under ``pytest-xdist`` one worker
collects and runs tests from many directories, so a conftest that rebinds
the top-level ``sys.modules["src"]`` entry at import time (the shape of the
#9402 bug, fixed for the two known offenders in #9404) silently poisons
every later test in that worker: ``import src`` returns an unrelated
package and the shared-alias installer in ``src/__init__.py`` never runs.

``tests/unit/repo_hygiene/test_no_permanent_src_module_shadow.py`` pins the
*known* pivot conftests to the scoped enter/exit pattern. This module is the
general rule the #9409 readiness work asks for -- it applies to every
conftest in the repository, present and future:

1. **Static (module scope):** no top-level statement (including inside
   top-level ``if`` / ``try`` / ``with`` / loops) may assign to, delete,
   ``pop`` or ``update`` the ``"src"`` key of ``sys.modules``.
2. **Static (function scope):** a conftest that touches
   ``sys.modules["src"]`` inside a function must implement the scoped pivot
   contract from #9404: ``_enter_pivot`` / ``_exit_pivot`` plus the three
   pytest hooks that confine the pivot to the conftest's own directory.
3. **Dynamic:** importing any such conftest under a private module name
   must leave ``sys.modules.get("src")`` bit-for-bit unchanged.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKIP_DIR_NAMES = frozenset(
    {"archive", "legacy", "node_modules", "vendor", ".git", "__pycache__"}
)
_REQUIRED_PIVOT_NAMES = (
    "_enter_pivot",
    "_exit_pivot",
    "pytest_make_collect_report",
    "pytest_runtest_setup",
    "pytest_runtest_teardown",
)
_MUTATING_CALLS = frozenset(
    {"pop", "update", "setdefault", "__setitem__", "__delitem__"}
)


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def _pytest_testpaths() -> list[Path]:
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    raw = data["tool"]["pytest"]["ini_options"]["testpaths"]
    return [_REPO_ROOT / entry for entry in raw]


def _iter_conftests(roots: list[Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("conftest.py")):
            if any(
                part in _SKIP_DIR_NAMES for part in path.relative_to(_REPO_ROOT).parts
            ):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def _all_conftests() -> list[Path]:
    return list(_iter_conftests([_REPO_ROOT / "tests", *_pytest_testpaths()]))


# --------------------------------------------------------------------------
# Static analysis
# --------------------------------------------------------------------------


def _sys_module_aliases(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Return (names bound to the ``sys`` module, names bound to ``sys.modules``)."""
    sys_names: set[str] = {"sys"}
    modules_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "sys":
                    sys_names.add(alias.asname or "sys")
        elif isinstance(node, ast.ImportFrom) and node.module == "sys":
            for alias in node.names:
                if alias.name == "modules":
                    modules_names.add(alias.asname or "modules")
    return sys_names, modules_names


def _is_sys_modules(
    node: ast.AST, sys_names: set[str], modules_names: set[str]
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in modules_names
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "modules"
        and isinstance(node.value, ast.Name)
        and node.value.id in sys_names
    )


def _is_src_key(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "src"


def _is_src_subscript(
    node: ast.AST, sys_names: set[str], modules_names: set[str]
) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and _is_sys_modules(node.value, sys_names, modules_names)
        and _is_src_key(node.slice)
    )


def _call_mutates_src(
    node: ast.AST, sys_names: set[str], modules_names: set[str]
) -> bool:
    """``sys.modules.pop("src")``, ``.update({"src": ...})``, ``.setdefault("src", ...)``."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr not in _MUTATING_CALLS:
        return False
    if not _is_sys_modules(node.func.value, sys_names, modules_names):
        return False
    if node.func.attr == "update":
        for arg in node.args:
            if isinstance(arg, ast.Dict) and any(_is_src_key(k) for k in arg.keys if k):
                return True
        return any(kw.arg == "src" for kw in node.keywords)
    return bool(node.args) and _is_src_key(node.args[0])


def _statement_mutates_src(
    stmt: ast.stmt, sys_names: set[str], modules_names: set[str]
) -> bool:
    if isinstance(stmt, ast.Assign):
        return any(_is_src_subscript(t, sys_names, modules_names) for t in stmt.targets)
    if isinstance(stmt, (ast.AugAssign, ast.AnnAssign)):
        return _is_src_subscript(stmt.target, sys_names, modules_names)
    if isinstance(stmt, ast.Delete):
        return any(_is_src_subscript(t, sys_names, modules_names) for t in stmt.targets)
    if isinstance(stmt, ast.Expr):
        return _call_mutates_src(stmt.value, sys_names, modules_names)
    return False


def _iter_module_scope_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Yield statements executed at import time (never descend into defs)."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for field in ("body", "orelse", "finalbody"):
            nested = getattr(stmt, field, None)
            if isinstance(nested, list):
                yield from _iter_module_scope_statements(nested)
        for handler in getattr(stmt, "handlers", []) or []:
            yield from _iter_module_scope_statements(handler.body)
        for case in getattr(stmt, "cases", []) or []:
            yield from _iter_module_scope_statements(case.body)


def module_scope_src_pivots(source: str) -> list[int]:
    """Line numbers of import-time statements that mutate ``sys.modules['src']``."""
    tree = ast.parse(source)
    sys_names, modules_names = _sys_module_aliases(tree)
    return [
        stmt.lineno
        for stmt in _iter_module_scope_statements(tree.body)
        if _statement_mutates_src(stmt, sys_names, modules_names)
    ]


def references_src_module_entry(source: str) -> bool:
    """True when the file touches ``sys.modules['src']`` anywhere (read or write)."""
    tree = ast.parse(source)
    sys_names, modules_names = _sys_module_aliases(tree)
    for node in ast.walk(tree):
        if _is_src_subscript(node, sys_names, modules_names):
            return True
        if _call_mutates_src(node, sys_names, modules_names):
            return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_sys_modules(node.func.value, sys_names, modules_names)
            and node.args
            and _is_src_key(node.args[0])
        ):
            return True
    return False


def missing_pivot_contract_names(source: str) -> list[str]:
    tree = ast.parse(source)
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    defined |= {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    return [name for name in _REQUIRED_PIVOT_NAMES if name not in defined]


# --------------------------------------------------------------------------
# Dynamic probe
# --------------------------------------------------------------------------


def _load_conftest(path: Path) -> ModuleType:
    """Execute a conftest under a private module name (mirrors #9404's probe)."""
    rel = path.relative_to(_REPO_ROOT).with_suffix("")
    name = "_conftest_src_hygiene_probe__" + "__".join(rel.parts)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


_CONFTESTS = _all_conftests()
_PIVOT_CONFTESTS = [
    p for p in _CONFTESTS if references_src_module_entry(p.read_text(encoding="utf-8"))
]


# --------------------------------------------------------------------------
# Repository tests
# --------------------------------------------------------------------------


def test_conftests_were_discovered() -> None:
    """Guard the guard: an empty discovery would make every test below vacuous."""
    assert _REPO_ROOT / "tests" / "conftest.py" in _CONFTESTS
    assert any("c3d_viewer" in _rel(p) for p in _PIVOT_CONFTESTS), (
        "expected the #9404 c3d_viewer pivot conftest to be classified as a pivot"
    )


@pytest.mark.parametrize("conftest_path", _CONFTESTS, ids=_rel)
def test_no_conftest_pivots_src_at_module_scope(conftest_path: Path) -> None:
    """Import-time rebinding of ``sys.modules['src']`` is the #9402 bug."""
    lines = module_scope_src_pivots(conftest_path.read_text(encoding="utf-8"))
    assert not lines, (
        f"{_rel(conftest_path)} mutates sys.modules['src'] at module scope "
        f"(lines {lines}). Scope the pivot with _enter_pivot/_exit_pivot and the "
        "pytest_make_collect_report/pytest_runtest_setup/pytest_runtest_teardown "
        "hooks instead (see #9404)."
    )


@pytest.mark.parametrize("conftest_path", _PIVOT_CONFTESTS, ids=_rel)
def test_pivot_conftests_implement_scoped_contract(conftest_path: Path) -> None:
    missing = missing_pivot_contract_names(conftest_path.read_text(encoding="utf-8"))
    assert not missing, (
        f"{_rel(conftest_path)} touches sys.modules['src'] but does not define "
        f"{missing}; every pivot must use the directory-scoped pattern from #9404."
    )


@pytest.mark.parametrize("conftest_path", _PIVOT_CONFTESTS, ids=_rel)
def test_importing_pivot_conftest_leaves_src_identity_untouched(
    conftest_path: Path,
) -> None:
    previous = sys.modules.get("src")
    try:
        _load_conftest(conftest_path)
        assert sys.modules.get("src") is previous, (
            f"importing {_rel(conftest_path)} rebound sys.modules['src'] -- the "
            "pivot leaks into every later test in the same xdist worker (#9402)."
        )
    finally:
        if sys.modules.get("src") is not previous:
            if previous is not None:
                sys.modules["src"] = previous
            else:
                sys.modules.pop("src", None)


# --------------------------------------------------------------------------
# Self-tests for the static checker
# --------------------------------------------------------------------------


def test_static_check_rejects_module_scope_assignment(tmp_path: Path) -> None:
    bad = tmp_path / "conftest.py"
    bad.write_text('import sys\n\nsys.modules["src"] = object()\n', encoding="utf-8")
    assert module_scope_src_pivots(bad.read_text(encoding="utf-8")) == [3]


@pytest.mark.parametrize(
    "snippet",
    [
        'import sys\nif True:\n    sys.modules["src"] = object()\n',
        'import sys\ntry:\n    del sys.modules["src"]\nexcept KeyError:\n    pass\n',
        'import sys\nsys.modules.pop("src", None)\n',
        'import sys\nsys.modules.update({"src": object()})\n',
        'import sys as _sys\n_sys.modules["src"] = object()\n',
        'from sys import modules\nmodules["src"] = object()\n',
    ],
    ids=["if-block", "try-del", "pop", "update", "sys-alias", "from-import"],
)
def test_static_check_rejects_other_module_scope_shapes(snippet: str) -> None:
    assert module_scope_src_pivots(snippet), snippet


def test_static_check_allows_function_scope_pivot() -> None:
    good = (
        "import sys\n"
        "def _enter_pivot():\n"
        '    sys.modules["src"] = object()\n'
        "def _exit_pivot():\n"
        '    sys.modules.pop("src", None)\n'
    )
    assert module_scope_src_pivots(good) == []
    assert references_src_module_entry(good)
    assert missing_pivot_contract_names(good) == [
        "pytest_make_collect_report",
        "pytest_runtest_setup",
        "pytest_runtest_teardown",
    ]


def test_static_check_ignores_unrelated_sys_modules_use() -> None:
    src = 'import sys\nsys.modules["pydrake"] = object()\nx = sys.modules.get("os")\n'
    assert module_scope_src_pivots(src) == []
    assert not references_src_module_entry(src)
