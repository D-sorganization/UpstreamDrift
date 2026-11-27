"""Ledger entries must bind to the rendered inputs, not just the engine name (#14756 P3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.render_best_matches import (
    DEFAULT_LEDGER,
    TARGETS,
    bind_ledger_entry,
    render_summary,
)

pytestmark = pytest.mark.unit

ENTRY = {
    "engine": "opensim",
    "model": "Rajagopal OS9",
    "approach": "Kinematic marker IK",
    "key_parameters": "Address calibration 19.1 mm",
    "marker_rms_mm": 32.5,
    "joint_angle_rms_deg": None,
    "verified": True,
    "evidence": "receipt.json",
    "date": "2026-10-25",
    "verdict": "Kinematics only",
    "blocker": "#13991",
}


def _target(model_id: str):
    return next(t for t in TARGETS if t.model_id == model_id)


def _pinned(model_id: str) -> dict[str, str]:
    return {p.name: p.sha256 for p in _target(model_id).inputs}


def test_matching_binding_returns_the_entry() -> None:
    hashes = _pinned("opensim_os9")
    entry = {**ENTRY, "input_sha256": dict(hashes)}
    assert bind_ledger_entry(entry, _target("opensim_os9"), hashes) is entry


def test_same_engine_wrong_run_ledger_is_rejected() -> None:
    hashes = _pinned("opensim_os9")
    other_run = dict.fromkeys(hashes, "0" * 64)
    entry = {**ENTRY, "input_sha256": other_run}
    with pytest.raises(ValueError, match="different run"):
        bind_ledger_entry(entry, _target("opensim_os9"), hashes)


def test_ledger_entry_without_a_binding_is_rejected() -> None:
    with pytest.raises(ValueError, match="no input_sha256"):
        bind_ledger_entry(dict(ENTRY), _target("opensim_os9"), _pinned("opensim_os9"))


def test_partial_binding_is_rejected() -> None:
    hashes = _pinned("opensim_os9")
    first = next(iter(hashes))
    entry = {**ENTRY, "input_sha256": {first: hashes[first]}}
    with pytest.raises(ValueError, match="different run"):
        bind_ledger_entry(entry, _target("opensim_os9"), hashes)


@pytest.mark.parametrize("model_id", [t.model_id for t in TARGETS])
def test_committed_ledger_binds_every_render_target(model_id: str) -> None:
    target = _target(model_id)
    ledger = json.loads(DEFAULT_LEDGER.read_text(encoding="utf-8"))
    entry = next(e for e in ledger["engines"] if e["engine"] == target.ledger_engine)
    assert bind_ledger_entry(entry, target, _pinned(model_id)) is entry


def test_summary_states_the_verified_binding(tmp_path: Path) -> None:
    hashes = _pinned("opensim_os9")
    entry = {**ENTRY, "input_sha256": dict(hashes)}
    text = render_summary(
        _target("opensim_os9"), entry, hashes, {"a.mp4": "f" * 64}, "c0ffee"
    )
    assert "Ledger binding: verified" in text
