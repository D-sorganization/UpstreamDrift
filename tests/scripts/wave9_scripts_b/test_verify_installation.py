"""Tests for scripts/ci/verify_installation.py."""

from __future__ import annotations

import sys
import types

import pytest

from scripts.ci import verify_installation as mod


def test_check_python_version_ok() -> None:
    ok, msg = mod.check_python_version()
    # The test suite already runs on 3.10+, so this is always True
    assert ok
    assert "✓" in msg


def test_check_virtualenv_returns_tuple() -> None:
    ok, msg = mod.check_virtualenv()
    assert isinstance(ok, bool)
    assert isinstance(msg, str)


def test_check_import_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = types.ModuleType("fake_pkg_xyz")
    fake_mod.__version__ = "1.2.3"
    monkeypatch.setitem(sys.modules, "fake_pkg_xyz", fake_mod)
    ok, msg = mod.check_import("fake_pkg_xyz")
    assert ok
    assert "1.2.3" in msg


def test_check_import_uses_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = types.ModuleType("real_path_abc")
    fake_mod.PYQT_VERSION_STR = "6.5"
    monkeypatch.setitem(sys.modules, "real_path_abc", fake_mod)
    ok, msg = mod.check_import("Display", "real_path_abc", "PYQT_VERSION_STR")
    assert ok
    assert "6.5" in msg
    assert "Display" in msg


def test_check_import_unknown_version(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = types.ModuleType("noversion_pkg")
    monkeypatch.setitem(sys.modules, "noversion_pkg", fake_mod)
    ok, msg = mod.check_import("noversion_pkg")
    assert ok
    assert "unknown" in msg


def test_check_import_missing() -> None:
    ok, msg = mod.check_import("definitely_not_a_real_module_xyz_12345")
    assert not ok
    assert "✗" in msg


def test_check_import_rejects_non_string() -> None:
    with pytest.raises(ValueError):
        mod.check_import(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mod.check_import("x", import_path=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mod.check_import("x", version_attr=5)  # type: ignore[arg-type]


def test_main_runs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["verify"])
    # Stub check_import to avoid heavy actual imports
    monkeypatch.setattr(mod, "check_import", lambda *a, **k: (True, "ok"))
    code = mod.main()
    assert code == 0


def test_main_json_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["verify", "--json"])
    monkeypatch.setattr(mod, "check_import", lambda *a, **k: (True, "ok"))
    code = mod.main()
    assert code == 0
    captured = capsys.readouterr()
    assert '"python_ok"' in captured.out


def test_main_failure_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["verify"])
    monkeypatch.setattr(mod, "check_import", lambda *a, **k: (False, "x"))
    assert mod.main() == 1
