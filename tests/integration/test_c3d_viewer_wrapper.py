"""Integration test for the C3D viewer entry-point wrapper.

Pins the package-pivot wrapper added in PR #4595. The wrapper inserts the
engine ``src/`` onto ``sys.path`` so the viewer's relative imports resolve
when invoked as a flat script.

Marked ``slow`` and ``requires_gl`` so the fast headless suite skips it
even though the test runs under ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations

import ast
import os
import runpy
import subprocess
import sys
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

    # ``import_module`` is called with a module-level constant, so resolve
    # simple ``Name`` arguments through the module's string assignments.
    string_constants = {
        target.id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    import_module_args = []
    for call in calls:
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "import_module"
            and call.args
        ):
            continue
        argument = call.args[0]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
            import_module_args.append(argument.value)
        elif isinstance(argument, ast.Name) and argument.id in string_constants:
            import_module_args.append(string_constants[argument.id])
    assert EXPECTED_VIEWER_MODULE in import_module_args


@pytest.mark.unit
def test_wrapper_declares_shared_python_import_root() -> None:
    """The wrapper must seed ``src/shared/python`` (issue #8088).

    Shared helpers reachable from the viewer import the flat ``sidekick``
    package by bare name, which only resolves when ``src/shared/python`` is an
    import root. Omitting it made a source-checkout run depend on the caller
    having exported ``PYTHONPATH``.
    """
    namespace = runpy.run_path(str(WRAPPER), run_name="wrapper_probe")

    roots = [Path(str(root)) for root in namespace["_IMPORT_ROOTS"]]  # type: ignore[call-overload]
    assert REPO_ROOT / "src" / "shared" / "python" in roots
    assert REPO_ROOT / "src" in roots
    assert REPO_ROOT in roots


@pytest.mark.unit
def test_wrapper_import_paths_are_applied_in_declared_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_ensure_import_paths`` must not invert the declared precedence."""
    namespace = runpy.run_path(str(WRAPPER), run_name="wrapper_probe")
    roots = [str(root) for root in namespace["_IMPORT_ROOTS"]]  # type: ignore[call-overload]

    monkeypatch.setattr("sys.path", ["/unrelated"])
    namespace["_ensure_import_paths"]()  # type: ignore[operator]

    import sys as _sys

    assert _sys.path[: len(roots)] == roots


@pytest.mark.slow
def test_wrapper_starts_from_source_checkout_without_pythonpath() -> None:
    """Importing the viewer through the wrapper needs no external PYTHONPATH."""
    if not _check_viewer_deps_available():
        pytest.skip("C3D viewer GUI dependencies unavailable")

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["QT_QPA_PLATFORM"] = "offscreen"

    program = (
        "import runpy, sys;"
        f"ns = runpy.run_path({str(WRAPPER)!r}, run_name='wrapper_probe');"
        "ns['_load_main']();"
        "print('WRAPPER_OK')"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(REPO_ROOT.parent),
    )

    assert "WRAPPER_OK" in result.stdout, result.stderr[-2000:]
