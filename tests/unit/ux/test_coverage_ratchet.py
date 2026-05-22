"""Tests for the UX coverage ratchet script (epic #5968, Phase 0.5/7.1)."""

from __future__ import annotations

import json
from importlib import util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "ci" / "check_ux_coverage_ratchet.py"


def _load_ratchet_module():
    """Import the ratchet script as a module without executing main()."""
    spec = util.spec_from_file_location("ux_ratchet_under_test", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load ratchet script")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ratchet_passes_against_existing_baseline():
    """The ratchet must currently pass — if it ever fails on main, CI
    blocks every PR.  This test catches accidental baseline edits.
    """
    mod = _load_ratchet_module()
    assert mod.main([]) == 0


def test_ratchet_initial_baseline_seed(tmp_path, monkeypatch):
    """--update-baseline with no existing file writes current counts."""
    mod = _load_ratchet_module()
    baseline_path = tmp_path / "baseline.json"
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline_path)
    rc = mod.main(["--update-baseline"])
    assert rc == 0
    assert baseline_path.exists()
    written = json.loads(baseline_path.read_text())
    # All expected pattern names are present.
    for name in mod.PY_PATTERNS:
        assert name in written
    for name in mod.WEB_PATTERNS:
        assert name in written


def test_ratchet_fails_when_count_exceeds_baseline(tmp_path, monkeypatch, capsys):
    """If the live count exceeds the baseline, exit code is 1."""
    mod = _load_ratchet_module()
    baseline_path = tmp_path / "baseline.json"
    # Seed an artificially low baseline so the live count exceeds it.
    baseline_path.write_text(
        json.dumps(dict.fromkeys({**mod.PY_PATTERNS, **mod.WEB_PATTERNS}, 0)),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline_path)
    rc = mod.main([])
    captured = capsys.readouterr()
    assert rc == 1
    assert "UX coverage ratchet FAILED" in captured.err


def test_ratchet_update_baseline_only_shrinks(tmp_path, monkeypatch):
    """Running --update-baseline on an existing baseline never grows
    a count (lower-only ratchet, matching error_handling_ratchet)."""
    mod = _load_ratchet_module()
    baseline_path = tmp_path / "baseline.json"
    # Seed an artificially high baseline.
    high = dict.fromkeys({**mod.PY_PATTERNS, **mod.WEB_PATTERNS}, 10000)
    baseline_path.write_text(json.dumps(high), encoding="utf-8")
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline_path)
    mod.main(["--update-baseline"])
    updated = json.loads(baseline_path.read_text())
    for value in updated.values():
        assert value <= 10_000
