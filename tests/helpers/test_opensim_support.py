"""OpenSim test imports are held at the native-runtime boundary (#12065).

A test (or test module) that needs the OpenSim SDK must never load it into a
pytest process that already hosts the Python ``ezc3d`` extension: the two ship
ABI-incompatible ``ezc3d.dll`` builds.  ``require_opensim`` is the single
test-side route; under a co-hosting conflict it skips with the boundary reason
*before* any import, and it skips for absence only when OpenSim is genuinely
not installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest

from scripts.diagnostics.native_runtime_identity import RUNTIME_ROLES
from tests.helpers import native_runtime_boundary as boundary
from tests.helpers import opensim_support as support

pytestmark = pytest.mark.unit

_MODULE_LEVEL_TEST = """
from tests.helpers.opensim_support import require_opensim

osim = require_opensim()


def test_never_collected():
    raise AssertionError("module body ran past the boundary")
"""


@pytest.fixture
def clean_runtimes(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """A process view in which neither runtime nor a foreign ezc3d.dll is loaded."""
    for name in RUNTIME_ROLES:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(boundary, "loaded_dll_path", lambda name: None)
    return monkeypatch


def _raise(exc: BaseException):
    def _importer() -> types.ModuleType:
        raise exc

    return _importer


def _forbid_import(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden() -> types.ModuleType:
        raise AssertionError("opensim import attempted inside an ezc3d process")

    monkeypatch.setattr(support, "import_opensim", _forbidden)


def test_require_opensim_skips_at_boundary_without_importing(
    clean_runtimes: pytest.MonkeyPatch,
) -> None:
    clean_runtimes.setitem(sys.modules, "ezc3d", types.ModuleType("ezc3d"))
    _forbid_import(clean_runtimes)
    with pytest.raises(pytest.skip.Exception) as info:
        support.require_opensim()
    assert str(info.value.msg).startswith(boundary.BOUNDARY_SKIP_PREFIX)
    assert info.value.allow_module_level is True


def test_test_module_importing_opensim_skips_under_conflict(
    clean_runtimes: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Executing a module that loads OpenSim at import time skips the module."""
    clean_runtimes.setitem(sys.modules, "ezc3d", types.ModuleType("ezc3d"))
    _forbid_import(clean_runtimes)
    source = tmp_path / "test_simulated_opensim_module.py"
    source.write_text(_MODULE_LEVEL_TEST, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(source.stem, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(pytest.skip.Exception) as info:
        spec.loader.exec_module(module)
    assert str(info.value.msg).startswith(boundary.BOUNDARY_SKIP_PREFIX)
    assert "opensim" not in sys.modules


def test_require_opensim_returns_module_in_clean_process(
    clean_runtimes: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("opensim")
    clean_runtimes.setattr(support, "import_opensim", lambda: fake)
    assert support.require_opensim() is fake


def test_require_opensim_skips_only_when_not_installed(
    clean_runtimes: pytest.MonkeyPatch,
) -> None:
    missing = ModuleNotFoundError("No module named 'opensim'", name="opensim")
    clean_runtimes.setattr(support, "import_opensim", _raise(missing))
    with pytest.raises(pytest.skip.Exception, match=support.SKIP_REASON):
        support.require_opensim()


def test_missing_sdk_dependency_is_not_absence(
    clean_runtimes: pytest.MonkeyPatch,
) -> None:
    """OpenSim present but one of its own modules missing is a broken install."""
    broken = ModuleNotFoundError("No module named '_simbody'", name="_simbody")
    clean_runtimes.setattr(support, "import_opensim", _raise(broken))
    with pytest.raises(ModuleNotFoundError) as info:
        support.require_opensim()
    assert info.value is broken


def test_other_import_errors_fail_loudly(clean_runtimes: pytest.MonkeyPatch) -> None:
    error = ImportError("DLL load failed while importing _simbody")
    clean_runtimes.setattr(support, "import_opensim", _raise(error))
    with pytest.raises(ImportError) as info:
        support.require_opensim()
    assert info.value is error
