"""Tests for CLI provider discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.python.ai.cli_providers import discovery


def test_discover_returns_empty_when_nothing_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(discovery.shutil, "which", lambda _name: None)
    monkeypatch.setattr(discovery.os.path, "expanduser", lambda _p: str(tmp_path))
    assert discovery.discover_cli_providers() == []


def test_discover_finds_claude_via_which(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_which(name: str) -> str | None:
        return "/usr/local/bin/claude" if name == "claude" else None

    monkeypatch.setattr(discovery.shutil, "which", fake_which)
    monkeypatch.setattr(discovery.os.path, "expanduser", lambda _p: str(tmp_path))

    descriptors = discovery.discover_cli_providers()
    ids = [d.id for d in descriptors]
    assert "claude-cli" in ids
    claude = next(d for d in descriptors if d.id == "claude-cli")
    assert claude.executable_path == "/usr/local/bin/claude"
    assert claude.transport == "stdio"


def test_discover_finds_codex_via_which(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_which(name: str) -> str | None:
        return "/usr/local/bin/codex" if name == "codex" else None

    monkeypatch.setattr(discovery.shutil, "which", fake_which)
    monkeypatch.setattr(discovery.os.path, "expanduser", lambda _p: str(tmp_path))

    descriptors = discovery.discover_cli_providers()
    ids = [d.id for d in descriptors]
    assert "codex-cli" in ids


def test_discover_finds_cline_via_config_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".cline").mkdir()
    monkeypatch.setattr(discovery.shutil, "which", lambda _n: None)
    monkeypatch.setattr(discovery.os.path, "expanduser", lambda _p: str(tmp_path))

    descriptors = discovery.discover_cli_providers()
    cline_descs = [d for d in descriptors if d.id == "cline"]
    assert len(cline_descs) == 1
    assert cline_descs[0].transport == "socket"
    assert cline_descs[0].working_dir_aware is False


def test_discover_finds_claude_via_config_dir_when_binary_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".claude").mkdir()
    monkeypatch.setattr(discovery.shutil, "which", lambda _n: None)
    monkeypatch.setattr(discovery.os.path, "expanduser", lambda _p: str(tmp_path))

    descriptors = discovery.discover_cli_providers()
    assert any(d.id == "claude-cli" for d in descriptors)


def test_discover_windows_exe_suffix_handled_by_which(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # shutil.which on Windows returns the .exe path automatically; we
    # only need to confirm we don't mangle the path.
    def fake_which(name: str) -> str | None:
        if name == "claude":
            return r"C:\Users\u\bin\claude.exe"
        return None

    monkeypatch.setattr(discovery.shutil, "which", fake_which)
    monkeypatch.setattr(discovery.os.path, "expanduser", lambda _p: str(tmp_path))

    descriptors = discovery.discover_cli_providers()
    claude = next(d for d in descriptors if d.id == "claude-cli")
    assert claude.executable_path.endswith("claude.exe")
