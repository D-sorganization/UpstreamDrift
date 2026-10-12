"""The shared ezc3d test helper skips only when ezc3d is not installed."""

from __future__ import annotations

import pytest

from src.shared.python.data_io.native_dll_guard import NativeDllConflictError
from tests.helpers import ezc3d_support as support

pytestmark = pytest.mark.unit


def _raise(exc: BaseException):
    def _importer() -> None:
        raise exc

    return _importer


def test_require_skips_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(support, "import_ezc3d", _raise(ModuleNotFoundError("ezc3d")))
    with pytest.raises(pytest.skip.Exception):
        support.require_ezc3d()
    assert support.ezc3d_installed() is False


@pytest.mark.parametrize(
    "error",
    [ImportError("DLL load failed"), NativeDllConflictError("conflict")],
)
def test_other_import_errors_fail_loudly(
    monkeypatch: pytest.MonkeyPatch, error: ImportError
) -> None:
    monkeypatch.setattr(support, "import_ezc3d", _raise(error))
    with pytest.raises(ImportError) as info:
        support.require_ezc3d()
    assert info.value is error
    with pytest.raises(ImportError):
        support.ezc3d_installed()
