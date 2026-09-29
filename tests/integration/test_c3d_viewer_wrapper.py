"""Integration test for the C3D viewer entry-point wrapper.

Pins the package-pivot wrapper added in PR #4595. The wrapper inserts the
engine ``src/`` onto ``sys.path`` so the viewer's relative imports resolve
when invoked as a flat script.

These are source-checkout bootstrap checks only; they do not initialize the
Qt viewer and therefore belong in the fast unit lane.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = (
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "python"
    / "src"
    / "apps"
    / "run_c3d_viewer.py"
)
EXPECTED_VIEWER_MODULE = (
    "src.engines.Simscape_Multibody_Models.3D_Golf_Model.python.src.apps.c3d_viewer"
)
EXPECTED_SHARED_PYTHON_ROOT = REPO_ROOT / "src" / "shared" / "python"


def _check_viewer_deps_available() -> bool:
    """Check if optional C3D viewer dependencies are available.

    The C3D viewer requires pandas and GUI stack. Skip the test if these
    optional dependencies are not installed rather than hard-failing.
    """
    try:
        import pandas  # noqa: F401
    except ImportError:
        return False
    try:
        from PyQt5 import QtWidgets  # noqa: F401
    except ImportError:
        try:
            from PyQt6 import QtWidgets  # noqa: F401
        except ImportError:
            return False
    return True


@pytest.mark.unit
def test_wrapper_imports_viewer_without_src_package_pivot() -> None:
    """The wrapper must not delete or repoint the repo ``src`` package."""
    tree = ast.parse(WRAPPER.read_text(encoding="utf-8"), filename=str(WRAPPER))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    deleted_sys_modules = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Delete)
        for target in node.targets
        if (
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "modules"
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "sys"
        )
    ]
    assert deleted_sys_modules == []

    viewer_module_values = [
        assignment.value.value
        for assignment in tree.body
        if isinstance(assignment, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_VIEWER_MODULE"
            for target in assignment.targets
        )
        and isinstance(assignment.value, ast.Constant)
        and isinstance(assignment.value.value, str)
    ]
    import_module_calls = [
        call
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "import_module"
    ]

    assert EXPECTED_VIEWER_MODULE in viewer_module_values
    assert import_module_calls


@pytest.mark.unit
def test_wrapper_bootstrap_exposes_shared_python_root_for_sidekick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone execution must resolve the in-checkout ``sidekick`` package.

    ``sidekick`` is a top-level package rooted at ``src/shared/python``.  The
    wrapper is launched as a script, so it cannot rely on pytest's path setup
    or the launcher-managed subprocess environment to expose that directory.
    """
    shared_root = str(EXPECTED_SHARED_PYTHON_ROOT)
    for module_name in list(sys.modules):
        if module_name == "sidekick" or module_name.startswith("sidekick."):
            monkeypatch.delitem(sys.modules, module_name)
    monkeypatch.setattr(
        sys,
        "path",
        [
            entry
            for entry in sys.path
            if "shared/python" not in entry.replace("\\", "/")
        ],
    )
    spec = importlib.util.spec_from_file_location("_c3d_viewer_wrapper", WRAPPER)
    assert spec is not None and spec.loader is not None
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)

    wrapper._ensure_import_paths()

    assert shared_root in sys.path
    sidekick_spec = importlib.util.find_spec("sidekick")
    assert sidekick_spec is not None and sidekick_spec.origin is not None
    assert (
        Path(sidekick_spec.origin)
        .resolve()
        .is_relative_to(EXPECTED_SHARED_PYTHON_ROOT.resolve())
    )
