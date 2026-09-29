from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/momentum_transfer_human_registration.json"
)


def _registration() -> dict[str, object]:
    return json.loads(REGISTRATION.read_text(encoding="utf-8"))


def test_registration_is_participant_held_out_and_fail_closed() -> None:
    record = _registration()
    assert record["registration_status"] == "frozen_before_governed_human_outcomes"
    assert record["dataset_status"] == "not_acquired"
    assert record["split_unit"] == "participant"
    assert "bilateral_six_axis_grip_wrenches" in record["required_modalities"]
    assert "synthetic dry runs" in record["completion_gate"]


def test_every_question_has_null_falsifier_and_sensitivities() -> None:
    tests = _registration()["tests"]
    covered = {question for test in tests for question in test["questions"]}
    assert covered == {f"Q{index}" for index in range(1, 8)}
    for test in tests:
        assert test["estimand"] and test["null"] and test["falsifier"]
        assert len(test["sensitivities"]) >= 5


def test_registration_preserves_null_adverse_and_identity_controls() -> None:
    record = _registration()
    assert "participant_label_permutation" in record["negative_controls"]
    assert "adverse_load_test" in record["negative_controls"]
    assert (
        record["adverse_margins"]["peak_bilateral_grip_wrench"]
        == "frozen_before_outcome_access"
    )
    assert record["identity_policy"] == "pseudonym_only_no_identity_inference"
    assert record["missing_data_contract"]["primary_window_imputation"] == "prohibited"
