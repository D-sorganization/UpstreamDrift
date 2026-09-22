"""Tests for the GUI thread-blocking ratchet script (issue #8880).

Modelled on ``tests/unit/ux/test_coverage_ratchet.py`` and
``tests/unit/scripts/test_error_handling_ratchet.py`` -- same shape: a
live-workspace smoke check plus unit tests against the detection logic and
the baseline file mechanics, driven through a temp directory rather than
the real ``src/tools`` tree.
"""

from __future__ import annotations

import json
from importlib import util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "ci" / "check_gui_thread_blocking_ratchet.py"


def _load_ratchet_module():
    """Import the ratchet script as a module without executing main()."""
    spec = util.spec_from_file_location("gui_ratchet_under_test", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load ratchet script")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ratchet_main_returns_an_int_against_live_workspace():
    """Smoke check: invoking ``main([])`` against the real tree returns 0 or 1.

    The committed baseline (``scripts/config/gui_thread_blocking_baseline.json``)
    is the authoritative signal enforced as its own CI step; this just proves
    the script runs cleanly end to end.
    """
    mod = _load_ratchet_module()
    rc = mod.main([])
    assert rc in (0, 1)


# ---------------------------------------------------------------------------
# Detection heuristic
# ---------------------------------------------------------------------------


def test_migrated_file_is_not_flagged():
    """A file that imports async_action is never flagged, however it's wired."""
    mod = _load_ratchet_module()
    text = (
        "from src.tools.async_action import AsyncActionBar\n"
        "self._btn = QPushButton('Go')\n"
        "self._btn.clicked.connect(self._run_async)\n"
    )
    assert mod._needs_migration(text) is False


def test_unmigrated_button_handler_is_flagged():
    """A bare button-to-handler wire with no async_action import is flagged."""
    mod = _load_ratchet_module()
    text = (
        "self._btn = QPushButton('Go')\nself._btn.clicked.connect(self._run_inline)\n"
    )
    assert mod._needs_migration(text) is True


def test_noqa_marker_suppresses_the_flag():
    """A file that already backgrounds work another way can opt out."""
    mod = _load_ratchet_module()
    text = (
        "# noqa: gui-thread/ok -- QProcess-backed, not inline compute.\n"
        "self._btn = QPushButton('Go')\n"
        "self._btn.clicked.connect(self._run_inline)\n"
    )
    assert mod._needs_migration(text) is False


def test_file_with_no_buttons_is_not_flagged():
    """A gui.py with no push buttons at all (e.g. a pure display widget)."""
    mod = _load_ratchet_module()
    text = "class Widget(QWidget):\n    pass\n"
    assert mod._needs_migration(text) is False


# ---------------------------------------------------------------------------
# Baseline file mechanics
# ---------------------------------------------------------------------------


def test_ratchet_initial_baseline_seed(tmp_path, monkeypatch):
    """--update-baseline with no existing file writes the current count."""
    mod = _load_ratchet_module()
    tools_root = tmp_path / "tools"
    (tools_root / "leaky_tool").mkdir(parents=True)
    (tools_root / "leaky_tool" / "gui.py").write_text(
        "self._btn = QPushButton('Go')\nself._btn.clicked.connect(self._run)\n"
    )
    baseline_path = tmp_path / "baseline.json"
    monkeypatch.setattr(mod, "TOOLS_ROOT", tools_root)
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline_path)

    rc = mod.main(["--update-baseline"])

    assert rc == 0
    written = json.loads(baseline_path.read_text())
    assert written == {"unmigrated_gui_files": 1}


def test_ratchet_fails_when_count_exceeds_baseline(tmp_path, monkeypatch, capsys):
    """A new un-migrated file pushes the count above a zero baseline."""
    mod = _load_ratchet_module()
    tools_root = tmp_path / "tools"
    (tools_root / "leaky_tool").mkdir(parents=True)
    (tools_root / "leaky_tool" / "gui.py").write_text(
        "self._btn = QPushButton('Go')\nself._btn.clicked.connect(self._run)\n"
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"unmigrated_gui_files": 0}))
    monkeypatch.setattr(mod, "TOOLS_ROOT", tools_root)
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline_path)

    rc = mod.main([])
    captured = capsys.readouterr()

    assert rc == 1
    assert "GUI thread-blocking ratchet FAILED" in captured.err
    assert "leaky_tool" in captured.err


def test_ratchet_passes_at_or_below_baseline(tmp_path, monkeypatch):
    """A migrated file (imports async_action) does not regress the count."""
    mod = _load_ratchet_module()
    tools_root = tmp_path / "tools"
    (tools_root / "clean_tool").mkdir(parents=True)
    (tools_root / "clean_tool" / "gui.py").write_text(
        "from src.tools.async_action import AsyncActionBar\n"
        "self._btn = QPushButton('Go')\n"
        "self._btn.clicked.connect(self._run_async)\n"
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"unmigrated_gui_files": 0}))
    monkeypatch.setattr(mod, "TOOLS_ROOT", tools_root)
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline_path)

    rc = mod.main([])

    assert rc == 0


def test_ratchet_update_baseline_only_shrinks(tmp_path, monkeypatch):
    """--update-baseline never raises an existing baseline (lower-only ratchet)."""
    mod = _load_ratchet_module()
    tools_root = tmp_path / "tools"
    tools_root.mkdir(parents=True)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"unmigrated_gui_files": 10000}))
    monkeypatch.setattr(mod, "TOOLS_ROOT", tools_root)
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline_path)

    mod.main(["--update-baseline"])

    updated = json.loads(baseline_path.read_text())
    assert updated["unmigrated_gui_files"] <= 10000
