"""Tests for scripts/check_doc_size_budget.py."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts import check_doc_size_budget as mod


def test_budget_exception_from_json_valid() -> None:
    exc = mod.BudgetException.from_json(
        {"path": "docs/x.md", "owner": "@me", "expires_on": "2099-01-01"}
    )
    assert exc.path == "docs/x.md"
    assert exc.owner == "@me"
    assert exc.expires_on == date(2099, 1, 1)


def test_budget_exception_normalizes_windows_path() -> None:
    exc = mod.BudgetException.from_json(
        {"path": "docs\\sub\\x.md", "owner": "@me", "expires_on": "2099-01-01"}
    )
    assert exc.path == "docs/sub/x.md"


def test_budget_exception_missing_path() -> None:
    with pytest.raises(ValueError, match="missing path"):
        mod.BudgetException.from_json(
            {"path": "", "owner": "@me", "expires_on": "2099-01-01"}
        )


def test_budget_exception_bad_owner() -> None:
    with pytest.raises(ValueError, match="owner must be a GitHub handle"):
        mod.BudgetException.from_json(
            {"path": "docs/x.md", "owner": "me", "expires_on": "2099-01-01"}
        )


def test_budget_exception_bad_date() -> None:
    with pytest.raises(ValueError, match="expires_on must be YYYY-MM-DD"):
        mod.BudgetException.from_json(
            {"path": "docs/x.md", "owner": "@me", "expires_on": "tomorrow"}
        )


def test_is_active() -> None:
    today = date(2026, 1, 1)
    exc = mod.BudgetException("p", "@o", today)
    assert exc.is_active(today)
    assert exc.is_active(today - timedelta(days=1))
    assert not exc.is_active(today + timedelta(days=1))


def _setup_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(mod, "CONFIG_PATH", tmp_path / "config.json")
    return tmp_path


def test_main_passes_when_no_offenders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _setup_root(tmp_path, monkeypatch)
    (root / "docs").mkdir()
    (root / "docs" / "small.md").write_text("hi")
    assert mod.main() == 0
    assert "passed" in capsys.readouterr().out


def test_main_excludes_node_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _setup_root(tmp_path, monkeypatch)
    (root / "node_modules" / "x").mkdir(parents=True)
    big = root / "node_modules" / "x" / "big.md"
    big.write_text("x" * (mod.MAX_BYTES + 1))
    assert mod.main() == 0


def test_main_fails_for_oversize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _setup_root(tmp_path, monkeypatch)
    (root / "docs").mkdir()
    big = root / "docs" / "big.md"
    big.write_text("x" * (mod.MAX_BYTES + 1))
    assert mod.main() == 1
    assert "big.md" in capsys.readouterr().err


def test_main_with_active_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _setup_root(tmp_path, monkeypatch)
    (root / "docs").mkdir()
    big = root / "docs" / "big.md"
    big.write_text("x" * (mod.MAX_BYTES + 1))
    (root / "config.json").write_text(
        json.dumps(
            {
                "max_bytes": mod.MAX_BYTES,
                "exceptions": [
                    {
                        "path": "docs/big.md",
                        "owner": "@me",
                        "expires_on": "2099-01-01",
                    }
                ],
            }
        )
    )
    assert mod.main() == 0


def test_main_with_expired_exception_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _setup_root(tmp_path, monkeypatch)
    (root / "docs").mkdir()
    big = root / "docs" / "big.md"
    big.write_text("x" * (mod.MAX_BYTES + 1))
    (root / "config.json").write_text(
        json.dumps(
            {
                "exceptions": [
                    {
                        "path": "docs/big.md",
                        "owner": "@me",
                        "expires_on": "2000-01-01",
                    }
                ]
            }
        )
    )
    assert mod.main() == 1


def test_main_with_bad_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _setup_root(tmp_path, monkeypatch)
    (root / "config.json").write_text("not json")
    assert mod.main() == 1
    assert "invalid doc size budget config" in capsys.readouterr().err


def test_main_with_custom_max_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _setup_root(tmp_path, monkeypatch)
    (root / "docs").mkdir()
    f = root / "docs" / "x.md"
    f.write_text("x" * 100)
    (root / "config.json").write_text(json.dumps({"max_bytes": 10}))
    assert mod.main() == 1


def test_iter_documents_only_doc_extensions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _setup_root(tmp_path, monkeypatch)
    (root / "docs").mkdir()
    (root / "docs" / "a.md").write_text("a")
    (root / "docs" / "b.qmd").write_text("b")
    (root / "docs" / "c.txt").write_text("c")
    docs = mod._iter_documents()
    names = sorted(p.name for p in docs)
    assert names == ["a.md", "b.qmd"]
