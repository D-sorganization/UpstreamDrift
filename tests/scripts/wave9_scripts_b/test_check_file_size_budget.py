"""Tests for scripts/ci/check_file_size_budget.py."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts.ci import check_file_size_budget as mod


def test_exception_is_active_no_expiry() -> None:
    assert mod._exception_is_active({})


def test_exception_is_active_future() -> None:
    future = (date.today() + timedelta(days=10)).isoformat()
    assert mod._exception_is_active({"expires_on": future})


def test_exception_is_active_past() -> None:
    past = (date.today() - timedelta(days=1)).isoformat()
    assert not mod._exception_is_active({"expires_on": past})


def test_line_count(tmp_path: Path) -> None:
    p = tmp_path / "x.py"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    assert mod._line_count(p) == 3


def test_load_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "scripts" / "config"
    config_dir.mkdir(parents=True)
    config_path = Path("scripts/config/budget.json")
    (tmp_path / config_path).write_text(json.dumps({"max_lines": 100}))
    out = mod._load_config(tmp_path, config_path)
    assert out["max_lines"] == 100


def test_get_owner_no_codeowners(tmp_path: Path) -> None:
    assert mod._get_owner("src/x.py", tmp_path) == "Unknown"


def test_get_owner_match(tmp_path: Path) -> None:
    co = tmp_path / ".github" / "CODEOWNERS"
    co.parent.mkdir()
    co.write_text("# comment\n\nsrc/ @team-a\n")
    assert mod._get_owner("src/x.py", tmp_path) == "@team-a"


def test_get_owner_no_match(tmp_path: Path) -> None:
    co = tmp_path / ".github" / "CODEOWNERS"
    co.parent.mkdir()
    co.write_text("docs/ @docs\n")
    assert mod._get_owner("src/x.py", tmp_path) == "Unknown"


def test_collect_active_exceptions_missing_fields() -> None:
    cfg = {"exceptions": [{"path": "", "owner": "@x", "reason": "issue #1"}]}
    _, invalid = mod._collect_active_exceptions(cfg)
    assert any("Invalid exception" in m for m in invalid)


def test_collect_active_exceptions_missing_issue_link() -> None:
    cfg = {"exceptions": [{"path": "a.py", "owner": "@x", "reason": "just because"}]}
    _, invalid = mod._collect_active_exceptions(cfg)
    assert any("missing linked issue" in m for m in invalid)


def test_collect_active_exceptions_expired() -> None:
    past = (date.today() - timedelta(days=1)).isoformat()
    cfg = {
        "exceptions": [
            {
                "path": "a.py",
                "owner": "@x",
                "reason": "issue #5",
                "expires_on": past,
            }
        ]
    }
    active, invalid = mod._collect_active_exceptions(cfg)
    assert active == {}
    assert any("Expired" in m for m in invalid)


def test_collect_active_exceptions_bad_date() -> None:
    cfg = {
        "exceptions": [
            {
                "path": "a.py",
                "owner": "@x",
                "reason": "issue #5",
                "expires_on": "garbage",
            }
        ]
    }
    _, invalid = mod._collect_active_exceptions(cfg)
    assert any("Invalid expires_on" in m for m in invalid)


def test_collect_active_exceptions_active() -> None:
    cfg = {
        "exceptions": [
            {"path": "a.py", "owner": "@x", "reason": "issue #5"},
            {
                "path": "b.py",
                "owner": "@y",
                "reason": "decomposition pending",
            },
        ]
    }
    active, invalid = mod._collect_active_exceptions(cfg)
    assert set(active.keys()) == {"a.py", "b.py"}
    assert invalid == []


def test_run_git_failure(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        mod._run_git(["this-is-not-a-real-git-cmd"], tmp_path)


def test_main_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "scripts" / "config"
    config_dir.mkdir(parents=True)
    cfg = config_dir / "budget.json"
    cfg.write_text(json.dumps({"max_lines": 100, "exceptions": []}))
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(mod, "_changed_python_files", lambda r, b: [])
    monkeypatch.setattr(
        "sys.argv",
        ["x", "--config-path", "scripts/config/budget.json"],
    )
    assert mod.main() == 0


def test_main_violation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "scripts" / "config"
    config_dir.mkdir(parents=True)
    cfg = config_dir / "budget.json"
    cfg.write_text(json.dumps({"max_lines": 2, "exceptions": []}))
    big = tmp_path / "big.py"
    big.write_text("a\nb\nc\nd\ne\n")
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(mod, "_changed_python_files", lambda r, b: [big])
    monkeypatch.setattr(
        "sys.argv",
        ["x", "--config-path", "scripts/config/budget.json"],
    )
    assert mod.main() == 1


def test_main_skips_tests_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "scripts" / "config"
    config_dir.mkdir(parents=True)
    cfg = config_dir / "budget.json"
    cfg.write_text(json.dumps({"max_lines": 1, "exceptions": []}))
    (tmp_path / "tests").mkdir()
    big = tmp_path / "tests" / "big.py"
    big.write_text("a\nb\nc\n")
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(mod, "_changed_python_files", lambda r, b: [big])
    monkeypatch.setattr(
        "sys.argv",
        ["x", "--config-path", "scripts/config/budget.json"],
    )
    assert mod.main() == 0


def test_main_watchlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "scripts" / "config"
    config_dir.mkdir(parents=True)
    cfg = config_dir / "budget.json"
    cfg.write_text(json.dumps({"max_lines": 10, "exceptions": []}))
    p = tmp_path / "warn.py"
    p.write_text("\n".join(["x"] * 9) + "\n")
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(mod, "_changed_python_files", lambda r, b: [p])
    monkeypatch.setattr(
        "sys.argv",
        ["x", "--config-path", "scripts/config/budget.json"],
    )
    assert mod.main() == 0


def test_main_fallback_on_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "scripts" / "config"
    config_dir.mkdir(parents=True)
    cfg = config_dir / "budget.json"
    cfg.write_text(json.dumps({"max_lines": 100, "exceptions": []}))
    calls = {"n": 0}

    def fake(r, base):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("no origin/main")
        return []

    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(mod, "_changed_python_files", fake)
    monkeypatch.setattr(
        "sys.argv",
        ["x", "--config-path", "scripts/config/budget.json"],
    )
    assert mod.main() == 0
    assert calls["n"] == 2


def test_main_too_many_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "scripts" / "config"
    config_dir.mkdir(parents=True)
    cfg = config_dir / "budget.json"
    excs = [{"path": f"f{i}.py", "owner": "@x", "reason": "issue #1"} for i in range(6)]
    cfg.write_text(json.dumps({"max_lines": 100, "exceptions": excs}))
    monkeypatch.setattr(mod, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(mod, "_changed_python_files", lambda r, b: [])
    monkeypatch.setattr(
        "sys.argv",
        ["x", "--config-path", "scripts/config/budget.json"],
    )
    assert mod.main() == 1
