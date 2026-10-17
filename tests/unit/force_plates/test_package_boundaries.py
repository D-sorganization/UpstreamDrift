"""OF-01/OF-02 (#13061): the force-plate package stays engine-free and AGPL-free.

open-fp is AGPL-3.0; UpstreamDrift integrates with it only through files and
its network API.  No module here may import an ``open_fp`` package, a physics
engine or a GUI toolkit.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

PACKAGE = (
    Path(__file__).resolve().parents[3] / "src" / "shared" / "python" / "force_plates"
)
FORBIDDEN_ROOTS = {
    "open_fp",
    "mujoco",
    "pydrake",
    "pinocchio",
    "opensim",
    "PyQt6",
    "matplotlib",
}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_package_modules_exist() -> None:
    assert (PACKAGE / "__init__.py").is_file()


@pytest.mark.parametrize(
    "module",
    sorted(PACKAGE.rglob("*.py")),
    ids=lambda p: p.relative_to(PACKAGE).as_posix(),
)
def test_module_imports_no_engine_gui_or_agpl_package(module: Path) -> None:
    assert not (_imported_roots(module) & FORBIDDEN_ROOTS)
