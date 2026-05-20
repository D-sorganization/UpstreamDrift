"""Tests for scripts/check_module_size_budget.py."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts import check_module_size_budget as mod


def test_count_lines(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text("a\nb\nc\n")
    assert mod.count_lines(f) == 3


def test_count_lines_no_trailing_newline(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text("a\nb\nc")
    assert mod.count_lines(f) == 3


def test_should_skip() -> None:
    assert mod.should_skip(Path("a/.git/c.py"), {".git"})
    assert not mod.should_skip(Path("a/b/c.py"), {".git"})


def test_iter_python_files(tmp_path: Path) -> None:
    (tmp_path / "src" / "a").mkdir(parents=True)
    (tmp_path / "src" / "a" / "x.py").touch()
    (tmp_path / "src" / "a" / "__pycache__").mkdir()
    (tmp_path / "src" / "a" / "__pycache__" / "y.py").touch()
    (tmp_path / "src" / "a" / "y.txt").touch()
    files = list(mod.iter_python_files(("src",), {"__pycache__"}, tmp_path))
    rel = sorted(str(p.relative_to(tmp_path)).replace("\\", "/") for p in files)
    assert rel == ["src/a/x.py"]


def test_iter_python_files_missing_root(tmp_path: Path) -> None:
    files = list(mod.iter_python_files(("nonexistent",), set(), tmp_path))
    assert files == []


def test_get_owner_default(tmp_path: Path) -> None:
    assert mod._get_owner("src/x.py", tmp_path) == "Unknown"


def test_get_owner_from_codeowners(tmp_path: Path) -> None:
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "CODEOWNERS").write_text(
        "# comment\n\nsrc/ @team-a\nsrc/api/ @team-b\n"
    )
    assert mod._get_owner("src/api/foo.py", tmp_path) == "@team-b"
    assert mod._get_owner("src/other.py", tmp_path) == "@team-a"


def test_get_owner_with_leading_slash(tmp_path: Path) -> None:
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "CODEOWNERS").write_text("/src/api/ @team-b\n")
    assert mod._get_owner("src/api/foo.py", tmp_path) == "@team-b"


def test_exception_is_active_no_date() -> None:
    assert mod._exception_is_active({"path": "x"})


def test_exception_is_active_future() -> None:
    future = (date.today() + timedelta(days=10)).isoformat()
    assert mod._exception_is_active({"expires_on": future})


def test_exception_is_active_past() -> None:
    past = (date.today() - timedelta(days=1)).isoformat()
    assert not mod._exception_is_active({"expires_on": past})


def test_collect_active_exceptions_valid() -> None:
    future = (date.today() + timedelta(days=30)).isoformat()
    config = {
        "exceptions": [
            {
                "path": "src/big.py",
                "owner": "@me",
                "reason": "tracked in issue #123",
                "expires_on": future,
            }
        ]
    }
    active, invalid = mod._collect_active_exceptions(config)
    assert "src/big.py" in active
    assert invalid == []


def test_collect_active_exceptions_invalid_missing_field() -> None:
    config = {"exceptions": [{"path": "x", "owner": "", "reason": "issue 1"}]}
    active, invalid = mod._collect_active_exceptions(config)
    assert active == {}
    assert len(invalid) == 1


def test_collect_active_exceptions_missing_issue_in_reason() -> None:
    config = {"exceptions": [{"path": "x", "owner": "@m", "reason": "just because"}]}
    active, invalid = mod._collect_active_exceptions(config)
    assert active == {}
    assert any("missing linked issue" in s for s in invalid)


def test_collect_active_exceptions_expired() -> None:
    past = (date.today() - timedelta(days=10)).isoformat()
    config = {
        "exceptions": [
            {
                "path": "x",
                "owner": "@m",
                "reason": "issue #1",
                "expires_on": past,
            }
        ]
    }
    active, invalid = mod._collect_active_exceptions(config)
    assert active == {}
    assert any("Expired" in s for s in invalid)


def test_collect_active_exceptions_bad_date() -> None:
    config = {
        "exceptions": [
            {
                "path": "x",
                "owner": "@m",
                "reason": "issue #1",
                "expires_on": "not-a-date",
            }
        ]
    }
    active, invalid = mod._collect_active_exceptions(config)
    assert active == {}
    assert any("Invalid expires_on" in s for s in invalid)


def test_collect_active_exceptions_legacy_reason() -> None:
    config = {"exceptions": [{"path": "x", "owner": "@m", "reason": "legacy module"}]}
    active, _ = mod._collect_active_exceptions(config)
    assert "x" in active


def test_load_baseline_missing(tmp_path: Path) -> None:
    assert mod.load_baseline(tmp_path / "nope.json") == {}


def test_load_baseline_present(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"max_lines": 500}))
    assert mod.load_baseline(p) == {"max_lines": 500}


def _setup_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    return tmp_path


def test_main_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _setup_repo(tmp_path, monkeypatch)
    (repo / "src").mkdir()
    (repo / "src" / "small.py").write_text("a\n" * 10)
    monkeypatch.setattr("sys.argv", ["check", "--baseline", "missing.json"])
    assert mod.main() == 0


def test_main_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _setup_repo(tmp_path, monkeypatch)
    (repo / "src").mkdir()
    (repo / "src" / "huge.py").write_text("a\n" * 100)
    monkeypatch.setattr(
        "sys.argv",
        ["check", "--baseline", "missing.json", "--max-lines", "10"],
    )
    assert mod.main() == 1


def test_main_watchlist_zone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    repo = _setup_repo(tmp_path, monkeypatch)
    (repo / "src").mkdir()
    # 95 lines with budget 100 (95% -> watchlist 90<=count<=100)
    (repo / "src" / "near.py").write_text("a\n" * 95)
    monkeypatch.setattr(
        "sys.argv",
        ["check", "--baseline", "missing.json", "--max-lines", "100"],
    )
    import logging

    with caplog.at_level(logging.INFO):
        assert mod.main() == 0
    assert any("WATCHLIST" in r.message for r in caplog.records)


def test_main_with_too_many_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _setup_repo(tmp_path, monkeypatch)
    (repo / "src").mkdir()
    (repo / "scripts" / "config").mkdir(parents=True)
    excs = [
        {
            "path": f"src/x{i}.py",
            "owner": "@m",
            "reason": "issue #1",
        }
        for i in range(10)
    ]
    baseline = repo / "baseline.json"
    baseline.write_text(json.dumps({"max_lines": 1000, "exceptions": excs}))
    monkeypatch.setattr("sys.argv", ["check", "--baseline", "baseline.json"])
    assert mod.main() == 1


def test_main_uses_exception_to_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _setup_repo(tmp_path, monkeypatch)
    (repo / "src").mkdir()
    (repo / "src" / "big.py").write_text("a\n" * 100)
    baseline = repo / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "max_lines": 10,
                "exceptions": [
                    {
                        "path": "src/big.py",
                        "owner": "@m",
                        "reason": "issue #1",
                    }
                ],
            }
        )
    )
    monkeypatch.setattr("sys.argv", ["check", "--baseline", "baseline.json"])
    assert mod.main() == 0
