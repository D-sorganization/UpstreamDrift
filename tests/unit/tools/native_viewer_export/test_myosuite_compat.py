"""MyoSuite 2.x / 3.x renderer import compatibility (issue #11997)."""

from __future__ import annotations

import subprocess
import sys
import types
from typing import Any

import pytest

from src.tools.native_viewer_export.backends import myosuite_arena
from src.tools.native_viewer_export.backends.myosuite_arena import (
    MyoSuiteArenaBackend,
)
from src.tools.native_viewer_export.backends.myosuite_compat import (
    MJ_RENDERER_MODULES,
    PROBE_CODE,
    import_mj_renderer,
)

pytestmark = pytest.mark.unit

NEW, OLD = MJ_RENDERER_MODULES


def _fake_module(marker: str) -> types.ModuleType:
    module = types.ModuleType("fake")
    module.MJRenderer = type("MJRenderer", (), {"marker": marker})  # type: ignore[attr-defined]
    return module


def test_prefers_3x_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, NEW, _fake_module("3x"))
        mp.setitem(sys.modules, OLD, _fake_module("2x"))
        assert import_mj_renderer().marker == "3x"


def test_falls_back_to_2x_renderer() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, NEW, None)  # None makes the import raise
        mp.setitem(sys.modules, OLD, _fake_module("2x"))
        assert import_mj_renderer().marker == "2x"


def test_missing_renderer_raises_import_error_listing_modules() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, NEW, None)
        mp.setitem(sys.modules, OLD, None)
        with pytest.raises(ImportError) as excinfo:
            import_mj_renderer()
    for name in MJ_RENDERER_MODULES:
        assert name in str(excinfo.value)


def test_unavailable_reason_probes_worker_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        seen["argv"] = argv
        return subprocess.CompletedProcess(
            argv, 1, "", "ModuleNotFoundError: no MyoSuite MJRenderer found"
        )

    monkeypatch.setattr(myosuite_arena, "myosuite_python", lambda: "/fake/python")
    monkeypatch.setattr(myosuite_arena.subprocess, "run", fake_run)
    reason = MyoSuiteArenaBackend().unavailable_reason()
    assert reason is not None
    assert "MJRenderer" in reason
    assert seen["argv"] == ["/fake/python", "-c", PROBE_CODE]


def test_unavailable_reason_none_when_probe_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(myosuite_arena, "myosuite_python", lambda: "/fake/python")
    monkeypatch.setattr(
        myosuite_arena.subprocess,
        "run",
        lambda argv, **_: subprocess.CompletedProcess(argv, 0, "", ""),
    )
    assert MyoSuiteArenaBackend().unavailable_reason() is None
