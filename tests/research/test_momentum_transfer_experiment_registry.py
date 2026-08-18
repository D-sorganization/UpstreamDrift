from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/momentum_transfer_experiment_registry.json"
)


def _registry() -> dict[str, object]:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_every_question_has_an_executable_experiment() -> None:
    experiments = _registry()["experiments"]
    covered = {question for item in experiments for question in item["questions"]}
    assert covered == {f"Q{index}" for index in range(1, 8)}
    assert len({item["id"] for item in experiments}) == len(experiments)


def test_experiments_are_controlled_and_falsifiable() -> None:
    for experiment in _registry()["experiments"]:
        assert len(experiment["interventions"]) >= 3
        assert len(experiment["controls"]) >= 3
        assert len(experiment["outcomes"]) >= 3
        assert experiment["uncertainty"] and experiment["required_data"]
        assert experiment["falsifier"] and experiment["execution_status"]


def test_key_failure_controls_are_explicit() -> None:
    experiments = {item["id"]: item for item in _registry()["experiments"]}
    assert "same_state_all_driver_killswitch" in experiments["MT-E01"]["interventions"]
    assert "reversed_moment_arm" in experiments["MT-E02"]["controls"]
    assert "alternative_casting_definitions" in experiments["MT-E03"]["controls"]
    assert "delayed_observer_policy" in experiments["MT-E04"]["interventions"]
    assert "interior_optimum" in experiments["MT-E05"]["outcomes"]
    assert "one_slack_class_at_a_time" in experiments["MT-E06"]["controls"]
    assert "manufactured_axial_null_mode" in experiments["MT-E07"]["controls"]
    assert "bilateral_six_axis_grip_wrenches" in experiments["MT-E07"]["required_data"]


def test_human_stage_is_participant_held_out_and_fail_closed() -> None:
    record = _registry()
    human = next(item for item in record["experiments"] if item["id"] == "MT-H01")
    assert "participant_holdout" in human["interventions"]
    assert "bilateral_six_axis_grip_wrenches" in human["required_data"]
    assert human["execution_status"] == "blocked_qualifying_dataset_not_acquired"
    assert "cannot establish a human strategy" in record["evidence_rule"]
