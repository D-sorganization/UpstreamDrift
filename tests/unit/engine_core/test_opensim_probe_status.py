"""Structured OpenSim probe status (#12883): never report a DLL clash as missing."""

from __future__ import annotations

import pytest
from src.shared.python.data_io.native_dll_guard import NativeDllConflictError
from src.shared.python.engine_core.opensim_runtime import (
    OpenSimProbe,
    OpenSimState,
    probe_opensim,
)

pytestmark = pytest.mark.unit


def _raiser(exc: BaseException):
    def _importer():
        raise exc

    return _importer


@pytest.mark.unit
def test_probe_installed() -> None:
    result = probe_opensim(lambda: object())
    assert result.state is OpenSimState.INSTALLED
    assert result.available and result.reason is None


@pytest.mark.unit
def test_probe_missing_is_not_import_failed() -> None:
    result = probe_opensim(_raiser(ModuleNotFoundError("No module named 'opensim'")))
    assert result.state is OpenSimState.MISSING
    assert not result.available
    assert "opensim" in (result.reason or "")


@pytest.mark.unit
def test_probe_dll_conflict_reports_real_reason() -> None:
    msg = "DLL load failed while importing _common; ezc3d.dll is already loaded"
    result = probe_opensim(_raiser(NativeDllConflictError(msg)))
    assert result.state is OpenSimState.IMPORT_FAILED
    assert result.dll_conflict is True
    assert result.reason == msg


@pytest.mark.unit
def test_probe_other_import_error_is_import_failed() -> None:
    result = probe_opensim(_raiser(ImportError("boom")))
    assert result.state is OpenSimState.IMPORT_FAILED
    assert result.dll_conflict is False
    assert result.reason == "boom"


@pytest.mark.unit
def test_probe_contract_rejects_inconsistent_state() -> None:
    with pytest.raises(ValueError):
        OpenSimProbe(OpenSimState.INSTALLED, "oops")
    with pytest.raises(ValueError):
        OpenSimProbe(OpenSimState.MISSING)
