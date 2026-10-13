"""Tests for the shared OpenSim/ezc3d native DLL conflict guard (#12065).

Importing OpenSim on Windows loads its bundled ``ezc3d.dll``.  A later
``import ezc3d`` then fails with ``DLL load failed ... specified procedure
could not be found`` because Windows reuses the already-loaded same-named DLL.
The guard must surface that as an explicit, diagnosable error instead of a
bare ImportError that call sites turn into a silent skip.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from src.shared.python.data_io import native_dll_guard as facade
from src.shared.python.sidekick.utils import native_dll_guard as guard

from tests.helpers.ezc3d_support import require_ezc3d

pytestmark = pytest.mark.unit

DLL_MSG = "DLL load failed while importing _ezc3d: The specified procedure could not be found."


def _fake_importer(exc: BaseException | None):
    def _import(name: str):
        assert name == "ezc3d"
        if exc is not None:
            raise exc
        return "ezc3d-module"

    return _import


def test_import_ezc3d_returns_module_on_success() -> None:
    assert guard.import_ezc3d(importer=_fake_importer(None)) == "ezc3d-module"


def test_missing_package_stays_plain_import_error() -> None:
    with pytest.raises(ModuleNotFoundError) as info:
        guard.import_ezc3d(importer=_fake_importer(ModuleNotFoundError("ezc3d")))
    assert not isinstance(info.value, guard.NativeDllConflictError)


def test_dll_failure_with_opensim_loaded_raises_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "opensim", object())
    monkeypatch.setattr(
        guard, "loaded_dll_path", lambda name: r"C:\OpenSim\bin\ezc3d.dll"
    )
    with pytest.raises(guard.NativeDllConflictError) as info:
        guard.import_ezc3d(importer=_fake_importer(ImportError(DLL_MSG)))
    message = str(info.value)
    assert r"C:\OpenSim\bin\ezc3d.dll" in message
    assert "separate process" in message
    assert DLL_MSG in message
    assert isinstance(info.value, ImportError)


def test_dll_failure_without_opensim_is_not_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "opensim", raising=False)
    monkeypatch.setattr(guard, "loaded_dll_path", lambda name: None)
    with pytest.raises(ImportError) as info:
        guard.import_ezc3d(importer=_fake_importer(ImportError(DLL_MSG)))
    assert not isinstance(info.value, guard.NativeDllConflictError)


def test_foreign_loaded_dll_triggers_conflict_even_without_opensim_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "opensim", raising=False)
    monkeypatch.setattr(guard, "loaded_dll_path", lambda name: r"C:\Other\ezc3d.dll")
    with pytest.raises(guard.NativeDllConflictError):
        guard.import_ezc3d(importer=_fake_importer(ImportError(DLL_MSG)))


def test_loaded_dll_path_unknown_dll_is_none() -> None:
    assert guard.loaded_dll_path("definitely_not_a_loaded_dll_12065.dll") is None


def test_loaded_dll_path_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        guard.loaded_dll_path("")


def _repo_env(repo: Path) -> dict[str, str]:
    """Child-process env mirroring pytest's ``pythonpath = [..., "src"]``.

    The Tools guard lives under ``src.shared.python.sidekick``, whose canonical
    ``shared.python`` alias needs ``src`` importable, as in every real entry point.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    src = str(repo / "src")
    env["PYTHONPATH"] = src if not existing else os.pathsep.join([src, existing])
    return env


def _opensim_ezc3d_dll() -> str | None:
    hits = sorted(glob.glob(r"C:\OpenSim*\bin\ezc3d.dll"))
    return hits[-1] if hits else None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL loader behaviour")
def test_native_first_then_ezc3d_in_fresh_process_is_explicit() -> None:
    """Load OpenSim's ezc3d.dll first, then import ezc3d via the guard."""
    dll = _opensim_ezc3d_dll()
    if dll is None:
        pytest.skip("no OpenSim install with bundled ezc3d.dll")
    require_ezc3d()
    script = textwrap.dedent(
        """
        import ctypes, os, sys
        dll = sys.argv[1]
        os.add_dll_directory(os.path.dirname(dll))
        ctypes.WinDLL(dll)
        from src.shared.python.data_io import native_dll_guard as g
        try:
            g.import_ezc3d()
            print("OK")
        except g.NativeDllConflictError as exc:
            assert dll.lower() in str(exc).lower(), str(exc)
            print("CONFLICT")
        """
    )
    repo = Path(__file__).resolve().parents[3]
    proc = subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-c", script, dll],
        cwd=repo,
        env=_repo_env(repo),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() in {"OK", "CONFLICT"}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DLL loader behaviour")
def test_ezc3d_first_then_native_dll_never_raises_in_fresh_process() -> None:
    """The opposite order must be diagnosable too (no uncaught ImportError)."""
    require_ezc3d()
    script = (
        "from src.shared.python.data_io import native_dll_guard as g;"
        "g.import_ezc3d(); print(bool(g.loaded_dll_path('ezc3d.dll')))"
    )
    repo = Path(__file__).resolve().parents[3]
    proc = subprocess.run(  # noqa: S603 - fixed argv, test-only
        [sys.executable, "-c", script],
        cwd=repo,
        env=_repo_env(repo),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "True"


def test_data_io_facade_exposes_the_canonical_guard_objects() -> None:
    """The stable ``data_io`` path re-exports Tools' guard (no duplicate)."""
    for name in facade.__all__:
        assert getattr(facade, name) is getattr(guard, name), name
