"""Tests for scripts/check_root_clutter.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_root_clutter as mod


def test_allowlist_is_frozen() -> None:
    assert isinstance(mod.ALLOWLIST, frozenset)
    assert "README.md" in mod.ALLOWLIST
    assert "pyproject.toml" in mod.ALLOWLIST


def _fake_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_scripts = tmp_path / "scripts"
    fake_scripts.mkdir()
    fake_script = fake_scripts / "check_root_clutter.py"
    fake_script.write_text("# fake")
    # Patch __file__ via the module's resolution path
    monkeypatch.setattr(mod, "__file__", str(fake_script))
    return tmp_path


def test_main_passes_with_allowed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _fake_root(tmp_path, monkeypatch)
    (root / "README.md").write_text("hi")
    (root / "LICENSE").write_text("x")
    (root / ".hidden").write_text("x")  # hidden is skipped
    (root / "subdir").mkdir()  # dir is skipped
    assert mod.main() == 0


def test_main_fails_for_disallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _fake_root(tmp_path, monkeypatch)
    (root / "junk.txt").write_text("x")
    (root / "scratch.py").write_text("x")
    rc = mod.main()
    assert rc == 1
    err = capsys.readouterr().err
    assert "junk.txt" in err
    assert "scratch.py" in err
    assert "Disallowed files" in err


def test_main_ignores_dotfiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fake_root(tmp_path, monkeypatch)
    (root / ".env").write_text("x")
    (root / ".gitignore").write_text("x")
    assert mod.main() == 0


def test_main_ignores_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fake_root(tmp_path, monkeypatch)
    (root / "src").mkdir()
    (root / "docs").mkdir()
    (root / "scripts").mkdir(exist_ok=True)
    assert mod.main() == 0
