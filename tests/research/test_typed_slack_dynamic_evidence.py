from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "docs/research/proximal_distal_energy_transfer/data/typed_slack_dynamic_study.json"
)


def test_dynamic_slack_evidence_is_complete_and_neutral() -> None:
    record = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert record["schema_version"] == "typed-slack-dynamic-study/v1"
    assert set(record["classes"]) == {
        "contact_disengagement",
        "transmission_backlash",
        "structural_preload",
        "biological_series_compliance",
        "control_deadband",
    }
    assert set(record["excitations"]) == {"slow_sine", "multisine_reversal"}
    assert record["claims"]["global_slack_benefit"] == "unsupported"
    assert record["claims"]["human_intentionality"] == "untested"
    assert record["claims"]["class_identification_from_one_channel"] != "established"
    mechanical = record["passivity_summary"]["mechanical_classes"]
    assert mechanical["all_pass"]
    assert mechanical["maximum_abs_energy_residual_j"] < 2e-5
    assert record["control_boundary"]["passivity_applicable"] is False
    assert record["identifiability"]["multisine_reversal"]
