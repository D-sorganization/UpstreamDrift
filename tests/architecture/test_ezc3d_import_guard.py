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
    "tests/helpers/ezc3d_support.py",
}

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
