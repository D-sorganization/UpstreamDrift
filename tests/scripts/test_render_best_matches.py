"""Tests for the best-model render orchestrator (MATCH-3 #14445)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from scripts.render_best_matches import (
    TARGETS,
    build_command,
    load_ledger_entry,
    render_summary,
    verify_inputs,
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


def test_targets_cover_best_mujoco_and_opensim_with_pinned_hashes() -> None:
    assert {t.model_id for t in TARGETS} == {"mujoco_driver", "opensim_os9"}
    for pinned in (p for t in TARGETS for p in t.inputs):
        assert len(pinned.sha256) == 64
        int(pinned.sha256, 16)


def test_verify_inputs_fails_closed_and_accepts_pinned_hash(tmp_path: Path) -> None:
    target = _target("mujoco_driver")
    name = target.inputs[0].name
    blob = tmp_path / "bundle.npz"
    with pytest.raises(ValueError, match="missing"):
        verify_inputs(target, {name: blob})
    blob.write_bytes(b"pinned")
    with pytest.raises(ValueError, match="sha256"):
        verify_inputs(target, {name: blob})
    digest = hashlib.sha256(b"pinned").hexdigest()
    pinned = dataclasses.replace(target.inputs[0], sha256=digest)
    repinned = dataclasses.replace(target, inputs=(pinned,))
    assert verify_inputs(repinned, {name: blob}) == {name: digest}


def test_build_command_is_headless_and_deterministic(tmp_path: Path) -> None:
    target = _target("mujoco_driver")
    paths = {target.inputs[0].name: tmp_path / "bundle.npz"}
    first = build_command(target, paths, tmp_path / "out", "python3")
    assert first == build_command(target, paths, tmp_path / "out", "python3")
    assert "src.tools.native_viewer_export" in first
    assert "--engines" in first and "mujoco" in first
    os9 = _target("opensim_os9")
    os9_paths = {p.name: tmp_path / p.name for p in os9.inputs}
    assert build_command(os9, os9_paths, tmp_path, "python3")[0] == "xvfb-run"


def test_render_summary_reports_fit_and_hides_private_paths(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"engines": [ENTRY]}), encoding="utf-8")
    entry = load_ledger_entry(ledger, "opensim")
    text = render_summary(
        _target("opensim_os9"),
        entry,
        {"model": "a" * 64},
        {"os9_face_on_1x_120fps.mp4": "b" * 64},
        "0123456789abcdef0123456789abcdef01234567",
    )
    for needle in ("Rajagopal OS9", "32.5", "not measured", "#13991", "a" * 64):
        assert needle in text
    assert "/home/" not in text
    assert "<bundle>" in render_summary(_target("mujoco_driver"), entry, {}, {}, "x")
    with pytest.raises(ValueError, match="mujoco"):
        load_ledger_entry(ledger, "mujoco")


def test_mujoco_best_match_migrates_to_the_v2_trunk_but_opensim_does_not() -> None:
    assert _target("mujoco_driver").trunk_v2 is True
    assert _target("opensim_os9").trunk_v2 is False


def test_v2_summary_labels_the_migration_a_replay_not_a_refit() -> None:
    entry = {**ENTRY, "engine": "mujoco"}
    text = render_summary(_target("mujoco_driver"), entry, {}, {}, "x")
    assert "replay, not a refit" in text and "lumbar" in text
    assert "replay, not a refit" not in render_summary(
        _target("opensim_os9"), ENTRY, {}, {}, "x"
    )


# --- #14756 P3: bind ledger metrics to the rendered run -----------------------

HASHES = {"model": "a" * 64, "states": "b" * 64}


def test_bound_ledger_entry_matching_rendered_hashes_is_accepted() -> None:
    from scripts.render_best_matches import verify_ledger_binding

    entry = {**ENTRY, "input_sha256": dict(HASHES)}
    assert verify_ledger_binding(_target("opensim_os9"), entry, HASHES) == "bound"


def test_same_engine_wrong_run_ledger_entry_is_rejected() -> None:
    from scripts.render_best_matches import verify_ledger_binding

    stale = {**ENTRY, "input_sha256": {"model": "c" * 64, "states": "b" * 64}}
    with pytest.raises(ValueError, match="model"):
        verify_ledger_binding(_target("opensim_os9"), stale, HASHES)


def test_ledger_entry_missing_a_rendered_input_hash_is_rejected() -> None:
    from scripts.render_best_matches import verify_ledger_binding

    partial = {**ENTRY, "input_sha256": {"model": "a" * 64}}
    with pytest.raises(ValueError, match="states"):
        verify_ledger_binding(_target("opensim_os9"), partial, HASHES)


def test_ledger_entry_without_declared_hashes_is_unbound_not_silently_joined() -> None:
    from scripts.render_best_matches import verify_ledger_binding

    assert verify_ledger_binding(_target("opensim_os9"), ENTRY, HASHES) == "unbound"


def test_summary_states_binding_status_and_render_only_operation() -> None:
    unbound = render_summary(_target("opensim_os9"), ENTRY, HASHES, {}, "x")
    assert "UNBOUND" in unbound and "not joined to this render" in unbound
    assert "Operation: visual replay" in unbound
    bound = render_summary(
        _target("opensim_os9"),
        {**ENTRY, "input_sha256": dict(HASHES)},
        HASHES,
        {},
        "x",
    )
    assert "Ledger binding: bound" in bound and "UNBOUND" not in bound
