"""``bunkershot3d`` must be imported under exactly one module name.

Both ``.`` and ``src`` are on ``pythonpath`` (``pyproject.toml``), so
``bunkershot3d.x`` and ``src.bunkershot3d.x`` both resolve — to **different**
module objects holding **different** class objects. When one module raises
``src.bunkershot3d.exceptions.BackendNotImplementedError`` and another catches
``bunkershot3d.exceptions.BackendNotImplementedError``, the ``except`` clause
does not match.

That is not hypothetical: it was the cause of the ``test_non_mock_run_fails_
instead_of_fabricating`` failure on main, and
``simulation_backends/wrench_extractor.py`` imported a second copy of the whole
package tree while its own docstring named the canonical path.

Canonical name: ``bunkershot3d``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FORBIDDEN_ROOT = "src.bunkershot3d"

#: Directories scanned for the forbidden import form.
_SCAN_ROOTS = ("src", "tests", "scripts", "notebooks")


def _python_files() -> list[Path]:
    files: list[Path] = []
    for root in _SCAN_ROOTS:
        base = _REPO_ROOT / root
        if not base.is_dir():
            continue
        files.extend(
            p
            for p in base.rglob("*.py")
            if "vendor" not in p.parts and ".codex-worktrees" not in p.parts
        )
    return files


def _imported_names(tree: ast.AST) -> list[str]:
    """Return every dotted module name imported by ``tree``."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def test_no_module_imports_bunkershot3d_under_the_src_prefix() -> None:
    """Every import of the package must use the canonical ``bunkershot3d`` root."""
    offenders: list[str] = []
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # not our concern here
            continue
        for name in _imported_names(tree):
            if name == _FORBIDDEN_ROOT or name.startswith(_FORBIDDEN_ROOT + "."):
                offenders.append(f"{path.relative_to(_REPO_ROOT).as_posix()} -> {name}")

    assert not offenders, (
        "These modules import bunkershot3d under the 'src.' prefix, which creates a "
        "second copy of the package with distinct class objects:\n  "
        + "\n  ".join(sorted(offenders))
        + "\nImport 'bunkershot3d.…' instead."
    )


def test_package_modules_do_not_mutate_sys_path() -> None:
    """Library code must not rewrite ``sys.path`` at import time.

    ``calibration/calibrate_all.py`` used to ``sys.path.insert(0, ...)`` on
    import, which is what made the duplicate package reachable in the first
    place. Path setup belongs to ``pyproject.toml``'s ``pythonpath``.
    """
    package_root = _REPO_ROOT / "src" / "bunkershot3d"
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "path"
                and isinstance(value.value, ast.Name)
                and value.value.id == "sys"
                and node.attr in {"insert", "append", "extend"}
            ):
                rel = path.relative_to(_REPO_ROOT).as_posix()
                offenders.append(f"{rel}:{node.lineno} sys.path.{node.attr}")

    assert not offenders, (
        "bunkershot3d modules must not mutate sys.path at import time:\n  "
        + "\n  ".join(sorted(offenders))
    )


def test_exception_identity_is_stable_across_the_public_api() -> None:
    """The exception re-exported from the package root is the same object.

    Guards against a future refactor reintroducing a second import root: if
    ``bunkershot3d.BackendNotImplementedError`` ever stops being the very class
    raised by the submodule, ``except`` clauses across the package break
    silently.
    """
    import bunkershot3d
    from bunkershot3d.exceptions import BackendNotImplementedError

    assert bunkershot3d.BackendNotImplementedError is BackendNotImplementedError
