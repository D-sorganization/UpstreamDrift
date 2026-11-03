"""Tests for the clean-environment install smoke test (#14275)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.ci import check_clean_install_smoke as smoke_mod

pytestmark = [pytest.mark.unit, pytest.mark.core_only]


def test_check_python_version_accepts_supported_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 11, 9, "final", 0))
    passed, message = smoke_mod.check_python_version()
    assert passed is True
    assert "3.11" in message


def test_check_python_version_rejects_unsupported_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 10, 12, "final", 0))
    passed, message = smoke_mod.check_python_version()
    assert passed is False
    assert "requires >= 3.11" in message


def test_check_core_dependencies_succeeds_when_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_import = MagicMock()
    monkeypatch.setattr(smoke_mod.importlib, "import_module", fake_import)
    results = smoke_mod.check_core_dependencies()
    assert len(results) == len(smoke_mod.CORE_REQUIRED_PACKAGES)
    assert all(ok for _, ok, _ in results)


def test_check_core_dependencies_detects_missing_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import(name: str) -> types.ModuleType:
        if name == "fastapi":
            raise ImportError("No module named 'fastapi'")
        return types.ModuleType(name)

    monkeypatch.setattr(smoke_mod.importlib, "import_module", fake_import)
    results = smoke_mod.check_core_dependencies()
    fastapi_result = next(r for r in results if r[0] == "fastapi")
    assert fastapi_result[1] is False
    assert (
        "missing" in fastapi_result[2].lower() or "failed" in fastapi_result[2].lower()
    )


def test_check_core_import_surface_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smoke_mod.importlib, "import_module", MagicMock())
    results = smoke_mod.check_core_import_surface()
    assert len(results) == len(smoke_mod.CORE_SURFACE_MODULES)
    assert all(ok for _, ok, _ in results)


def test_check_core_import_surface_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import(name: str) -> types.ModuleType:
        if "mujoco" in name:
            raise ImportError("DLL failed to load")
        return types.ModuleType(name)

    monkeypatch.setattr(smoke_mod.importlib, "import_module", fake_import)
    results = smoke_mod.check_core_import_surface()
    mujoco_result = next(r for r in results if "mujoco" in r[0])
    assert mujoco_result[1] is False


def test_check_forbidden_engine_isolation_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for mod in list(sys.modules):
        if any(
            mod == pkg or mod.startswith(f"{pkg}.")
            for pkg in smoke_mod.FORBIDDEN_ENGINES
        ):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    results = smoke_mod.check_forbidden_engine_isolation()
    assert all(ok for _, ok, _ in results)


def test_check_forbidden_engine_isolation_detects_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "drake", types.ModuleType("drake"))
    results = smoke_mod.check_forbidden_engine_isolation()
    drake_result = next(r for r in results if r[0] == "drake")
    assert drake_result[1] is False
    assert "loaded" in drake_result[2]


def test_check_cli_entrypoints_executes_help(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> MagicMock:
        mock = MagicMock()
        mock.returncode = 0
        mock.stdout = "Upstream Drift CLI help"
        mock.stderr = ""
        return mock

    monkeypatch.setattr(smoke_mod.subprocess, "run", fake_run)
    results = smoke_mod.check_cli_entrypoints(python_bin=Path(sys.executable))
    assert len(results) >= 2
    assert all(ok for _, ok, _ in results)


def test_run_clean_install_smoke_checks_aggregates_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke_mod, "check_python_version", lambda: (True, "Python OK"))
    monkeypatch.setattr(
        smoke_mod, "check_core_dependencies", lambda: [("numpy", True, "ok")]
    )
    monkeypatch.setattr(
        smoke_mod, "check_core_import_surface", lambda: [("src.api", True, "ok")]
    )
    monkeypatch.setattr(
        smoke_mod,
        "check_forbidden_engine_isolation",
        lambda: [("drake", True, "isolated")],
    )
    monkeypatch.setattr(
        smoke_mod,
        "check_cli_entrypoints",
        lambda **_: [("upstream-drift --help", True, "ok")],
    )

    report = smoke_mod.run_clean_install_smoke_checks(check_cli=True)
    assert report["status"] == "passed"
    assert report["summary"]["passed"] == report["summary"]["total"]


def test_main_cli_returns_zero_on_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        smoke_mod,
        "run_clean_install_smoke_checks",
        lambda **_: {
            "status": "passed",
            "summary": {"passed": 5, "total": 5},
            "checks": {},
        },
    )
    code = smoke_mod.main(["--json"])
    assert code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "passed"


def test_main_cli_returns_one_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        smoke_mod,
        "run_clean_install_smoke_checks",
        lambda **_: {
            "status": "failed",
            "summary": {"passed": 4, "total": 5},
            "checks": {},
        },
    )
    code = smoke_mod.main([])
    assert code == 1
