"""Unit tests for root allowlist checks in scripts/check_docs_governance.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.check_docs_governance import (
    ROOT,
    _load_root_allowlist,
    _stale_root_entries,
    _tracked_root_entries,
    _unexpected_root_entries,
    main,
)

pytestmark = pytest.mark.unit


def test_unexpected_root_entries_flagged() -> None:
    tracked = ["AGENTS.md", "README.md", "rogue.txt", "untracked_dir"]
    allowlist = ["AGENTS.md", "README.md"]
    unexpected = _unexpected_root_entries(tracked, allowlist)
    assert unexpected == ["rogue.txt", "untracked_dir"]


def test_stale_root_entries_flagged() -> None:
    tracked = ["AGENTS.md", "README.md"]
    allowlist = ["AGENTS.md", "README.md", "obsolete.txt"]
    stale = _stale_root_entries(tracked, allowlist)
    assert stale == ["obsolete.txt"]


def test_exact_match_passes() -> None:
    tracked = ["AGENTS.md", "README.md", "src"]
    allowlist = ["AGENTS.md", "README.md", "src"]
    assert _unexpected_root_entries(tracked, allowlist) == []
    assert _stale_root_entries(tracked, allowlist) == []


def test_loader_rejects_missing_entries(tmp_path: Path) -> None:
    config_file = tmp_path / "root_allowlist.json"
    config_file.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing 'entries'"):
        _load_root_allowlist(config_file)


def test_loader_rejects_non_string_entries(tmp_path: Path) -> None:
    config_file = tmp_path / "root_allowlist.json"
    config_file.write_text(
        json.dumps({"schema_version": 1, "entries": ["AGENTS.md", 42]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="string"):
        _load_root_allowlist(config_file)


def test_loader_rejects_duplicate_entries(tmp_path: Path) -> None:
    config_file = tmp_path / "root_allowlist.json"
    config_file.write_text(
        json.dumps({"schema_version": 1, "entries": ["AGENTS.md", "AGENTS.md"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"(?i)duplicate"):
        _load_root_allowlist(config_file)


def test_loader_rejects_names_containing_slash(tmp_path: Path) -> None:
    config_file = tmp_path / "root_allowlist.json"
    config_file.write_text(
        json.dumps({"schema_version": 1, "entries": ["docs/README.md"]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="/"):
        _load_root_allowlist(config_file)


def test_tracked_root_entries_returns_none_on_git_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_run = MagicMock(return_value=MagicMock(returncode=1, stdout=""))
    monkeypatch.setattr("subprocess.run", mock_run)
    assert _tracked_root_entries() is None


def test_real_repo_root_allowlist_passes() -> None:
    tracked = _tracked_root_entries()
    assert tracked is not None
    allowlist = _load_root_allowlist()
    assert _unexpected_root_entries(tracked, allowlist) == []
    assert _stale_root_entries(tracked, allowlist) == []


def test_main_reports_unexpected_entries(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "scripts.check_docs_governance._load_root_allowlist",
        lambda *args, **kwargs: ["README.md"],
    )
    monkeypatch.setattr(
        "scripts.check_docs_governance._tracked_root_entries",
        lambda: ["README.md", "unexpected_file.txt"],
    )
    monkeypatch.setattr(
        "scripts.check_docs_governance._duplicate_process_directories",
        list,
    )
    monkeypatch.setattr(
        "scripts.check_docs_governance.REQUIRED_FILES",
        [],
    )
    ret = main()
    assert ret == 1
    captured = capsys.readouterr()
    assert (
        "Unexpected repository root entries (add to scripts/config/root_allowlist.json with review):\n- unexpected_file.txt"
        in captured.err
    )


def test_main_reports_stale_entries(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "scripts.check_docs_governance._load_root_allowlist",
        lambda *args, **kwargs: ["README.md", "stale_entry.txt"],
    )
    monkeypatch.setattr(
        "scripts.check_docs_governance._tracked_root_entries",
        lambda: ["README.md"],
    )
    monkeypatch.setattr(
        "scripts.check_docs_governance._duplicate_process_directories",
        list,
    )
    monkeypatch.setattr(
        "scripts.check_docs_governance.REQUIRED_FILES",
        [],
    )
    ret = main()
    assert ret == 1
    captured = capsys.readouterr()
    assert (
        "Stale repository root allowlist entries (remove from scripts/config/root_allowlist.json):\n- stale_entry.txt"
        in captured.err
    )


def test_main_fails_on_git_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "scripts.check_docs_governance._load_root_allowlist",
        lambda *args, **kwargs: ["README.md"],
    )
    monkeypatch.setattr(
        "scripts.check_docs_governance._tracked_root_entries",
        lambda: None,
    )
    monkeypatch.setattr(
        "scripts.check_docs_governance._duplicate_process_directories",
        list,
    )
    monkeypatch.setattr(
        "scripts.check_docs_governance.REQUIRED_FILES",
        [],
    )
    ret = main()
    assert ret == 1
    captured = capsys.readouterr()
    assert (
        "Failed to list tracked repository root entries with git ls-tree"
        in captured.err
    )
