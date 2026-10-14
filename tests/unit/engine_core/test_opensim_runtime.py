"""Tests for the single production route to the OpenSim SDK (#12065).

``import_opensim`` is the src-side counterpart of ``import_ezc3d``: it keeps
plain ``import opensim`` semantics (absence stays a ``ModuleNotFoundError``,
any other failure an ``ImportError``) but turns the documented Windows
``ezc3d.dll`` ABI conflict into a typed, diagnosable error.
"""

from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from src.shared.python.data_io.native_dll_guard import NativeDllConflictError
from src.shared.python.engine_core import opensim_runtime

pytestmark = pytest.mark.unit

DLL_MSG = (
    "DLL load failed while importing _simulation: "
    "The specified procedure could not be found."
)


def _fake_importer(exc: BaseException | None, module: object = "opensim-module"):
    seen: list[str] = []

    def _import(name: str) -> Any:
        seen.append(name)
        if exc is not None:
            raise exc
        return module

    _import.seen = seen  # type: ignore[attr-defined]
    return _import


def test_returns_module_on_success() -> None:
    importer = _fake_importer(None)
    assert opensim_runtime.import_opensim(importer=importer) == "opensim-module"
    assert importer.seen == ["opensim"]  # type: ignore[attr-defined]


def test_imports_named_submodule() -> None:
    importer = _fake_importer(None)
    opensim_runtime.import_opensim("opensim._simulation", importer=importer)
    assert importer.seen == ["opensim._simulation"]  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", ["", "ezc3d", "opensim_models", "numpy.opensim"])
def test_rejects_non_opensim_module_names(name: str) -> None:
    with pytest.raises(ValueError, match="opensim"):
        opensim_runtime.import_opensim(name, importer=_fake_importer(None))


def test_missing_package_stays_module_not_found() -> None:
    error = ModuleNotFoundError("No module named 'opensim'", name="opensim")
    with pytest.raises(ModuleNotFoundError) as info:
        opensim_runtime.import_opensim(importer=_fake_importer(error))
    assert info.value is error


def test_dll_failure_with_ezc3d_hosted_raises_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ezc3d", ModuleType("ezc3d"))
    monkeypatch.setattr(
        opensim_runtime, "_ezc3d_dll_owner", lambda: r"C:\py\ezc3d\ezc3d.dll"
    )
    with pytest.raises(NativeDllConflictError) as info:
        opensim_runtime.import_opensim(importer=_fake_importer(ImportError(DLL_MSG)))
    message = str(info.value)
    assert DLL_MSG in message
    assert r"C:\py\ezc3d\ezc3d.dll" in message
    assert "separate process" in message
    assert isinstance(info.value, ImportError)


def test_dll_failure_without_foreign_runtime_is_not_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "ezc3d", raising=False)
    monkeypatch.setattr(opensim_runtime, "_ezc3d_dll_owner", lambda: None)
    error = ImportError(DLL_MSG)
    with pytest.raises(ImportError) as info:
        opensim_runtime.import_opensim(importer=_fake_importer(error))
    assert info.value is error


def test_non_dll_import_error_is_not_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ezc3d", ModuleType("ezc3d"))
    error = ImportError("cannot import name 'Model'")
    with pytest.raises(ImportError) as info:
        opensim_runtime.import_opensim(importer=_fake_importer(error))
    assert info.value is error


def test_default_importer_honours_sys_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(Model=object)
    monkeypatch.setitem(sys.modules, "opensim", fake)
    assert opensim_runtime.import_opensim() is fake


def test_default_importer_honours_blocked_sys_modules_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "opensim", None)
    with pytest.raises(ImportError):
        opensim_runtime.import_opensim()


def test_default_importer_goes_through_builtins_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patching ``builtins.__import__`` still simulates absence, as before."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "opensim":
            raise ImportError("simulated missing opensim")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "opensim", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="simulated missing opensim"):
        opensim_runtime.import_opensim()


def test_foreign_dll_without_ezc3d_module_raises_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "ezc3d", raising=False)
    monkeypatch.setattr(opensim_runtime, "_ezc3d_dll_owner", lambda: "other/ezc3d.dll")
    with pytest.raises(NativeDllConflictError, match="other/ezc3d.dll"):
        opensim_runtime.import_opensim(importer=_fake_importer(ImportError(DLL_MSG)))


def test_module_import_is_stdlib_only() -> None:
    """Routing an import through here must not add import-time weight."""

    probe = (
        "import sys; import src.shared.python.engine_core.opensim_runtime; "
        "heavy = [m for m in ('numpy', 'src.shared.python.data_io') "
        "if m in sys.modules]; assert not heavy, heavy"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[3],
    )
    assert result.returncode == 0, result.stderr
