"""Tests for scripts/ci/check_hardcoded_style_ratchet.py (issue #8885).

Verifies the ratchet:
  * ignores hex literals in files that never call setStyleSheet
  * passes when the count equals baseline
  * passes (and reports) when the count decreases
  * fails when the count grows
  * exits 2 on configuration errors
  * --update-baseline rewrites only the count field
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "ci" / "check_hardcoded_style_ratchet.py"


def _load_ratchet_module() -> object:
    """Load the script as a module (it lives outside the package tree)."""
    spec = importlib.util.spec_from_file_location(
        "hardcoded_style_ratchet", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hardcoded_style_ratchet"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_baseline(path: pathlib.Path, count: int) -> None:
    path.write_text(
        json.dumps({"_comment": "test", "count": count}, indent=2) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def ratchet_env(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load_ratchet_module()
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    monkeypatch.setattr(mod, "TOOLS_DIR", tools_dir)
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "SELF_EXEMPT", set())
    return mod, tools_dir, baseline_path


def test_hex_literals_outside_setstylesheet_files_are_not_counted(
    ratchet_env,
) -> None:
    mod, tools_dir, baseline_path = ratchet_env
    (tools_dir / "constants.py").write_text(
        "BGR_RED = (0, 0, 255)\n# not a stylesheet: #ff00ff #00ffcc\n",
        encoding="utf-8",
    )
    _write_baseline(baseline_path, 0)
    assert mod.main([]) == 0


def test_hex_literal_in_a_setstylesheet_file_is_counted(ratchet_env) -> None:
    mod, tools_dir, baseline_path = ratchet_env
    (tools_dir / "gui.py").write_text(
        'widget.setStyleSheet("background-color: #2E7D32;")\n',
        encoding="utf-8",
    )
    _write_baseline(baseline_path, 1)
    assert mod.main([]) == 0


def test_passes_when_count_equals_baseline(ratchet_env) -> None:
    mod, tools_dir, baseline_path = ratchet_env
    (tools_dir / "gui.py").write_text(
        'w.setStyleSheet("color: #111111; background: #222222;")\n',
        encoding="utf-8",
    )
    _write_baseline(baseline_path, 2)
    assert mod.main([]) == 0


def test_passes_and_reports_when_count_decreases(ratchet_env, caplog) -> None:
    mod, tools_dir, baseline_path = ratchet_env
    (tools_dir / "gui.py").write_text(
        'w.setStyleSheet("color: #111111;")\n',
        encoding="utf-8",
    )
    _write_baseline(baseline_path, 5)
    with caplog.at_level("INFO"):
        assert mod.main([]) == 0
    assert any("improved" in r.message.lower() for r in caplog.records)


def test_fails_when_count_grows(ratchet_env, caplog) -> None:
    mod, tools_dir, baseline_path = ratchet_env
    (tools_dir / "gui.py").write_text(
        'w.setStyleSheet("color: #111111; background: #222222; border: #333;")\n',
        encoding="utf-8",
    )
    _write_baseline(baseline_path, 1)
    with caplog.at_level("ERROR"):
        assert mod.main([]) == 1
    assert any("violation" in r.message.lower() for r in caplog.records)


def test_exit_2_when_baseline_missing(ratchet_env, caplog) -> None:
    mod, tools_dir, baseline_path = ratchet_env
    (tools_dir / "gui.py").write_text("x = 1\n", encoding="utf-8")
    with caplog.at_level("ERROR"):
        assert mod.main([]) == 2
    assert any("baseline" in r.message.lower() for r in caplog.records)


def test_exit_2_when_baseline_missing_count_key(ratchet_env, caplog) -> None:
    mod, tools_dir, baseline_path = ratchet_env
    (tools_dir / "gui.py").write_text("x = 1\n", encoding="utf-8")
    baseline_path.write_text(json.dumps({"_comment": "no count"}), encoding="utf-8")
    with caplog.at_level("ERROR"):
        assert mod.main([]) == 2
    assert any("count" in r.message.lower() for r in caplog.records)


def test_update_baseline_rewrites_count(ratchet_env) -> None:
    mod, tools_dir, baseline_path = ratchet_env
    (tools_dir / "gui.py").write_text(
        'w.setStyleSheet("color: #111111;")\n',
        encoding="utf-8",
    )
    _write_baseline(baseline_path, 5)
    assert mod.main(["--update-baseline"]) == 0
    new = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert new["count"] == 1


def test_update_baseline_blocked_on_regression(ratchet_env) -> None:
    mod, tools_dir, baseline_path = ratchet_env
    (tools_dir / "gui.py").write_text(
        'w.setStyleSheet("color: #111111; background: #222222;")\n',
        encoding="utf-8",
    )
    _write_baseline(baseline_path, 1)
    assert mod.main(["--update-baseline"]) == 1
    new = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert new["count"] == 1  # unchanged


def test_real_repo_count_is_at_or_below_the_committed_baseline() -> None:
    """Smoke test against the real tree: the ratchet must pass as committed."""
    mod = _load_ratchet_module()
    assert mod.main([]) == 0
