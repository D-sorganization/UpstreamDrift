"""Tests for scripts/check_no_print_calls.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_no_print_calls as mod


def test_in_production_path() -> None:
    roots = ("src/api", "src/shared/python")
    assert mod._in_production_path("src/api/x.py", roots)
    assert mod._in_production_path("src/shared/python/z/y.py", roots)
    assert not mod._in_production_path("tests/x.py", roots)
    assert not mod._in_production_path("src/api2/x.py", roots)
    # equal to root
    assert mod._in_production_path("src/api", roots)


def test_is_excluded() -> None:
    assert mod._is_excluded("src/api/examples/x.py")
    assert mod._is_excluded("src/api/tutorials/x.py")
    assert not mod._is_excluded("src/api/foo/x.py")


def test_find_print_calls_detects_top_level(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text("import sys\nprint('hi')\ndef f():\n    print('x')\n    return 1\n")
    assert mod.find_print_calls(p) == [2, 4]


def test_find_print_calls_ignores_attr_print(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text("import logging\nlogging.print = 1\nclass X: pass\nX.print()\n")
    assert mod.find_print_calls(p) == []


def test_find_print_calls_handles_syntax_error(tmp_path: Path) -> None:
    p = tmp_path / "broken.py"
    p.write_text("def x(:\n")
    assert mod.find_print_calls(p) == []


def test_find_print_calls_handles_unicode_error(tmp_path: Path) -> None:
    p = tmp_path / "bin.py"
    p.write_bytes(b"\xff\xfe\x00not utf8")
    assert mod.find_print_calls(p) == []


def test_find_print_calls_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "e.py"
    p.write_text("")
    assert mod.find_print_calls(p) == []


def _init_repo(tmp_path: Path) -> Path:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@e",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "init",
        ],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


def test_changed_python_files_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _init_repo(tmp_path)
    # Create files
    (repo / "src" / "api").mkdir(parents=True)
    (repo / "src" / "api" / "good.py").write_text("x = 1\n")
    (repo / "src" / "api" / "examples").mkdir()
    (repo / "src" / "api" / "examples" / "ex.py").write_text("x = 1\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "t.py").write_text("x = 1\n")
    (repo / "src" / "api" / "deleted.py").write_text("x = 1\n")

    import subprocess

    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@e",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "add",
        ],
        cwd=repo,
        check=True,
    )
    # Tag base
    subprocess.run(["git", "branch", "base"], cwd=repo, check=True)
    # Modify one file, delete another
    (repo / "src" / "api" / "good.py").write_text("y = 2\n")
    (repo / "src" / "api" / "deleted.py").unlink()
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@e",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "-m",
            "change",
        ],
        cwd=repo,
        check=True,
    )

    files = mod.changed_python_files(repo, "base", ("src/api",))
    rel = [str(p.relative_to(repo)).replace("\\", "/") for p in files]
    assert "src/api/good.py" in rel
    assert "src/api/deleted.py" not in rel  # no longer exists


def test_run_git_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        mod._run_git(["log", "nope"], tmp_path)


def test_main_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod, "changed_python_files", lambda *a, **k: [])
    monkeypatch.setattr("sys.argv", ["check"])
    assert mod.main() == 0


def test_main_reports_violations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "x.py"
    bad.write_text("print('hi')\n")
    monkeypatch.setattr(mod, "changed_python_files", lambda *a, **k: [bad])
    monkeypatch.setattr(mod.Path, "relative_to", lambda self, _r: Path(self.name))
    monkeypatch.setattr("sys.argv", ["check"])
    assert mod.main() == 1


def test_main_fallback_to_head1(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_changed(repo, base, roots):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("no origin/main")
        return []

    monkeypatch.setattr(mod, "changed_python_files", fake_changed)
    monkeypatch.setattr("sys.argv", ["check"])
    assert mod.main() == 0
    assert calls["n"] == 2


def test_main_fallback_fails_when_base_is_head1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_changed(repo, base, roots):
        raise RuntimeError("nope")

    monkeypatch.setattr(mod, "changed_python_files", fake_changed)
    monkeypatch.setattr("sys.argv", ["check", "--base-ref", "HEAD~1"])
    assert mod.main() == 1
