"""Every ``ezc3d`` import must route through the shared guard (#12065).

Importing the Python ``ezc3d`` extension next to a native OpenSim SDK fails
with an opaque Windows DLL error.  ``import_ezc3d`` in
``src/shared/python/data_io/native_dll_guard.py`` is the single place that
diagnoses it, so no other module may import ``ezc3d`` directly (DRY, LoD).
Tests obtain the module through ``tests/helpers/ezc3d_support.py``.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCAN_DIRS = ("src", "tests", "scripts")
SANCTIONED = {
    "src/shared/python/data_io/native_dll_guard.py",
    "src/shared/python/sidekick/utils/native_dll_guard.py",
    "tests/helpers/ezc3d_support.py",
}
GUARD_FACADE = "src/shared/python/data_io/native_dll_guard.py"
CANONICAL_GUARD_MODULE = "src.shared.python.sidekick.utils.native_dll_guard"
EZC3D_MODULE = "ezc3d"
pytestmark = pytest.mark.gate


def direct_ezc3d_imports(source: str) -> list[int]:
    """Return line numbers of ``import ezc3d`` / ``from ezc3d import`` statements."""
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] == "ezc3d" for a in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and (node.module or "").split(".")[0] == "ezc3d":
                lines.append(node.lineno)
    return lines


def importorskip_ezc3d_calls(source: str) -> list[int]:
    """Return line numbers of ``pytest.importorskip("ezc3d")`` style calls."""
    lines: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "importorskip" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == EZC3D_MODULE:
            lines.append(node.lineno)
    return lines


def test_importorskip_detector_flags_ezc3d_calls() -> None:
    assert importorskip_ezc3d_calls(
        'import pytest\npytest.importorskip("ezc3d")\n'
    ) == [2]
    assert importorskip_ezc3d_calls(
        'x = pytest.importorskip("ezc3d", reason="r")\n'
    ) == [1]
    assert importorskip_ezc3d_calls("importorskip('ezc3d')\n") == [1]
    assert importorskip_ezc3d_calls('pytest.importorskip("numpy")\n') == []


def test_detector_flags_direct_imports() -> None:
    assert direct_ezc3d_imports("import ezc3d\n") == [1]
    assert direct_ezc3d_imports("x = 1\nimport ezc3d as e\n") == [2]
    assert direct_ezc3d_imports("from ezc3d import c3d\n") == [1]
    assert direct_ezc3d_imports("import numpy\nfrom x import ezc3d\n") == []


def _candidate_files() -> list[str]:
    """Tracked ``.py`` files under the scanned roots that mention ``ezc3d``."""
    proc = subprocess.run(  # noqa: S603 - fixed argv, read-only git query
        ["git", "grep", "-l", "ezc3d", "--", *(f"{d}/*.py" for d in SCAN_DIRS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode in {0, 1}, proc.stderr
    return sorted(line for line in proc.stdout.splitlines() if line)


def test_no_module_imports_ezc3d_outside_the_guard() -> None:
    offenders: list[str] = []
    for rel in _candidate_files():
        if rel in SANCTIONED:
            continue
        source = (ROOT / rel).read_text(encoding="utf-8")
        offenders.extend(f"{rel}:{n}" for n in direct_ezc3d_imports(source))
    assert not offenders, (
        "direct ezc3d imports bypass the shared guard; use "
        "src.shared.python.data_io.native_dll_guard.import_ezc3d (src) or "
        "tests.helpers.ezc3d_support (tests):\n  " + "\n  ".join(offenders)
    )


def test_no_test_uses_importorskip_for_ezc3d() -> None:
    """``importorskip`` hides DLL conflicts as skips; use ``require_ezc3d``."""
    offenders: list[str] = []
    for rel in _candidate_files():
        if rel in SANCTIONED or rel == "tests/architecture/test_ezc3d_import_guard.py":
            continue
        source = (ROOT / rel).read_text(encoding="utf-8")
        offenders.extend(f"{rel}:{n}" for n in importorskip_ezc3d_calls(source))
    assert not offenders, (
        "pytest.importorskip('ezc3d') converts a native DLL conflict into a "
        "silent skip; use tests.helpers.ezc3d_support.require_ezc3d:\n  "
        + "\n  ".join(offenders)
    )


def test_data_io_guard_is_a_pure_reexport_of_the_tools_guard() -> None:
    """The data_io guard must not duplicate the Tools implementation (DRY)."""
    tree = ast.parse((ROOT / GUARD_FACADE).read_text(encoding="utf-8"))
    defined = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    ]
    assert not defined, f"{GUARD_FACADE} re-implements: {defined}"


def test_public_guard_names_are_the_canonical_objects() -> None:
    import importlib

    facade = importlib.import_module("src.shared.python.data_io.native_dll_guard")
    canonical = importlib.import_module(CANONICAL_GUARD_MODULE)
    for name in ("import_ezc3d", "loaded_dll_path", "NativeDllConflictError"):
        assert getattr(facade, name) is getattr(canonical, name), name
