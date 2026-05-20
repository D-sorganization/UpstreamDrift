"""Tests for scripts/ci/check_subsystem_status.py."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.ci import check_subsystem_status as mod


def test_load_subsystem_registry(tmp_path: Path) -> None:
    p = tmp_path / "reg.yaml"
    p.write_text(
        yaml.safe_dump({"subsystems": [{"name": "a", "status": "production"}]})
    )
    out = mod.load_subsystem_registry(str(p))
    assert out == [{"name": "a", "status": "production"}]


def test_load_subsystem_registry_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        mod.load_subsystem_registry(str(tmp_path / "nope.yaml"))
    assert exc.value.code == 2


def test_run_tests_for_path_skips_missing(tmp_path: Path) -> None:
    success, msg = mod.run_tests_for_path(str(tmp_path / "no.py"))
    assert success
    assert "SKIP" in msg


def test_run_tests_for_path_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "t.py").write_text("x=1\n")

    class FakeRes:
        returncode = 0
        stdout = "1 passed\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeRes())
    ok, output = mod.run_tests_for_path(str(tmp_path / "t.py"))
    assert ok
    assert "passed" in output


def test_run_tests_for_path_verbose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "t.py").write_text("x=1\n")

    class FakeRes:
        returncode = 1
        stdout = "out"
        stderr = "err"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeRes())
    ok, output = mod.run_tests_for_path(str(tmp_path / "t.py"), verbose=True)
    assert not ok
    assert "out" in output
    assert "err" in output


def test_run_tests_for_path_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "t.py").write_text("x=1\n")

    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, msg = mod.run_tests_for_path(str(tmp_path / "t.py"))
    assert not ok
    assert "TIMEOUT" in msg


def test_run_tests_for_path_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "t.py").write_text("x=1\n")

    def fake_run(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, msg = mod.run_tests_for_path(str(tmp_path / "t.py"))
    assert not ok
    assert "ERROR" in msg


def test_check_subsystem_status_all_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "t.py").write_text("x=1\n")
    monkeypatch.setattr(mod, "run_tests_for_path", lambda p, v=False: (True, "ok"))
    subs = [
        {"name": "s1", "status": "production", "test_paths": [str(tmp_path / "t.py")]}
    ]
    assert mod.check_subsystem_status(subs) is True


def test_check_subsystem_status_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "run_tests_for_path", lambda p, v=False: (False, "no"))
    subs = [{"name": "s1", "status": "production", "test_paths": ["p"]}]
    assert mod.check_subsystem_status(subs) is False


def test_check_subsystem_status_skips_non_production(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mod,
        "run_tests_for_path",
        lambda p, v=False: pytest.fail("should not be called"),
    )
    subs = [{"name": "s1", "status": "alpha", "test_paths": ["p"]}]
    assert mod.check_subsystem_status(subs, verbose=True) is True


def test_check_subsystem_status_warns_no_test_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mod, "run_tests_for_path", lambda p, v=False: (True, ""))
    subs = [{"name": "s1", "status": "production", "test_paths": []}]
    assert mod.check_subsystem_status(subs) is True


def test_check_subsystem_status_targets_specific(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        mod,
        "run_tests_for_path",
        lambda p, v=False: (calls.append(p), (True, "ok"))[1],
    )
    subs = [
        {"name": "s1", "status": "production", "test_paths": ["a"]},
        {"name": "s2", "status": "production", "test_paths": ["b"]},
    ]
    mod.check_subsystem_status(subs, target_subsystem="s2")
    assert calls == ["b"]


def test_main_no_subsystems(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"subsystems": []}))
    monkeypatch.setattr("sys.argv", ["x", "--registry", str(p)])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "load_subsystem_registry", lambda r: [])
    assert mod.main() == 2


def test_main_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["x"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        mod,
        "load_subsystem_registry",
        lambda r: [{"name": "s", "status": "alpha", "test_paths": []}],
    )
    assert mod.main() == 0
