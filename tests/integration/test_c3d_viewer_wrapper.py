"""Integration test for the C3D viewer entry-point wrapper.

Pins the package-pivot wrapper added in PR #4595. The wrapper inserts the
engine ``src/`` onto ``sys.path`` so the viewer's relative imports resolve
when invoked as a flat script.

Marked ``slow`` and ``requires_gl`` so the fast headless suite skips it
even though the test runs under ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.requires_gl]

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

    import_module_args = [
        call.args[0].value
        for call in calls
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "import_module"
            and call.args
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        )
    ]
    assert EXPECTED_VIEWER_MODULE in import_module_args
