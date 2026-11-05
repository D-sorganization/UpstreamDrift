"""Tests for the WSCG best-match MuJoCo model selection record (Issue #14437)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("mujoco")

from scripts.research.proximal_distal_energy.wscg_model_selection import (
    FEATURES,
    RECORD_PATH,
    REPO_ROOT,
    build_record,
    initial_condition_digest,
    rank_candidates,
)
from src.sim_run.manifest import validate_run_manifest

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

_BEST = "mujoco/advanced_biomechanical_golf_swing"
_RUNNER_UP = "mujoco/golf_double_pendulum_params"

# Features each candidate is expected to reproduce, measured from its MjModel.
_EXPECTED_FEATURES = {
    _BEST: {"joint_torque_driven", "two_hand_closure", "flexible_shaft"},
    _RUNNER_UP: {"planar", "joint_torque_driven"},
    "mujoco/golf_triple_pendulum": {"planar", "joint_torque_driven"},
    "mujoco/golf_upper_body_swing": {"joint_torque_driven", "two_hand_closure"},
    "mujoco/golf_full_body_swing": {"joint_torque_driven", "two_hand_closure"},
    "mujoco/canonical_golf_humanoid": {"joint_torque_driven"},
}


def _checked_in() -> dict:
    return json.loads((REPO_ROOT / RECORD_PATH).read_text(encoding="utf-8"))


def test_features_are_measured_from_each_model() -> None:
    measured = {
        score.model_id: {f for f in FEATURES if score.features[f]}
        for score in rank_candidates()
    }
    assert measured == _EXPECTED_FEATURES


def test_best_match_and_runner_up() -> None:
    ranking = rank_candidates()
    assert ranking[0].model_id == _BEST
    assert ranking[0].score == 3
    assert ranking[1].model_id == _RUNNER_UP
    assert set(ranking[1].missing) == {"two_hand_closure", "flexible_shaft"}
    assert ranking[0].missing == ("planar",)


def test_checked_in_selection_matches_regenerated_ranking() -> None:
    record = _checked_in()
    regenerated = build_record(
        ud_commit=record["manifest"]["ud_commit"],
        created_at=record["manifest"]["created_at"],
    )
    for key in ("metric", "ranking", "selected", "runner_up"):
        assert record[key] == regenerated[key], key
    assert "two_hand_closure" in record["runner_up"]["why_it_lost"]
    assert "not empirical validation" in record["scope"]


def test_manifest_pins_model_revision_and_initial_condition() -> None:
    record = _checked_in()
    manifest = validate_run_manifest(record["manifest"])
    assert manifest.model.model_id == _BEST
    assert manifest.model.revision.startswith("sha256:")
    assert manifest.engine.name == "mujoco"
    ic = record["initial_condition"]
    assert len(ic["qpos"]) == ic["nq"]
    assert manifest.input_data.sha256 == initial_condition_digest(
        ic["qpos"], [0.0] * ic["nv"]
    )


def test_initial_condition_digest_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="qpos"):
        initial_condition_digest([], [0.0])
